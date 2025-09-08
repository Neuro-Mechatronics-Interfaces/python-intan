import os
import time
import glob
import pickle
import random
import platform
import json
import torch
from typing import Any, Dict, Iterable, Optional, List
import joblib
import logging
import numpy as np
from collections import Counter
from intan.io import load_config_file
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    classification_report,
)


def setup_logger():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


setup_logger()


def _seed_everything(seed=42):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _jsonify(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def _is_classification(y):
    return y.dtype.kind in {'i', 'u', 'O', 'S', 'U'} and len(np.unique(y)) < 1000


def load_training_metadata(file_path: str) -> Dict:
    """Load model metadata saved at training time."""
    if file_path.endswith(".json"):
        root_dir = os.path.dirname(file_path)
    else:
        root_dir = file_path
    meta_path = os.path.join(root_dir, "model", "metadata.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing metadata.json at {meta_path}")
    with open(meta_path, "r") as f:
        return json.load(f)


def write_training_metadata(root_dir: str, *, window_ms: int, step_ms: int, envelope_cutoff_hz: float,
    channel_names: Optional[Iterable[str]] = None, selected_channels: Optional[Iterable[int]] = None,
    sample_rate_hz: Optional[float] = None, n_features: Optional[int] = None, feature_set: Optional[Iterable[str]] = None,
    require_complete: bool = True, required_fraction: float = 1.0, channel_wait_timeout_sec: float = 15.0) -> None:
    """
    Merge/update fields in model/metadata.json so real-time ZMQ prediction has a single source of truth.
    Non-destructive: preserves any existing keys unless overridden here.
    """
    meta_dir = os.path.join(root_dir, "model")
    os.makedirs(meta_dir, exist_ok=True)
    meta_path = os.path.join(meta_dir, "metadata.json")

    # Load existing metadata if present
    meta = load_training_metadata(meta_path) if os.path.isfile(meta_path) else {}

    # Core pipeline params (training-time truth)
    meta.update({
        "window_ms": int(window_ms),
        "step_ms": int(step_ms),
        "envelope_cutoff_hz": float(envelope_cutoff_hz),
        "require_complete": bool(require_complete),
        "required_fraction": float(required_fraction),
        "channel_wait_timeout_sec": float(channel_wait_timeout_sec),
    })

    if channel_names is not None:
        meta["channel_names"] = list(channel_names)
    if selected_channels is not None:
        meta["selected_channels"] = [int(i) for i in selected_channels]
    if sample_rate_hz is not None:
        meta["sample_rate_hz"] = float(sample_rate_hz)
    if n_features is not None:
        meta["n_features"] = int(n_features)
    if feature_set is not None:
        meta["feature_set"] = list(feature_set)

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Metadata updated at {meta_path}")


def _find_first(path_glob: str) -> str | None:
    try:
        return next(p for p in sorted(glob.glob(path_glob)))
    except StopIteration:
        return None





class ModelManager:
    def __init__(self, root_dir=None, label=None, model_cls=None, input_dim=None, output_dim=None, config=None, seed=42,
                 verbose=False):
        self.root_dir = root_dir
        self.label = label
        self.model_cls = model_cls
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eval_metrics = {}
        self.verbose = verbose

        if config and isinstance(config, str):
            config = load_config_file(config)
        self.config = config or {}

        # Logging
        level = logging.DEBUG if self.config.get('verbose', False) else logging.INFO
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=level)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Seed for reproducibility
        _seed_everything(self.config.get('seed', 42))
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.get("use_cuda", True) else "cpu")

        # Directories
        self.root_dir = root_dir
        self.model_dir = os.path.join(root_dir, 'model')
        os.makedirs(self.model_dir, exist_ok=True)

        # Paths
        prefix = f"{self.label}_" if label else ""
        self.scaler_path = os.path.join(self.model_dir, f"{prefix}scaler.pkl")
        self.model_path = os.path.join(self.model_dir, f"{prefix}model.pth")
        self.pca_path = os.path.join(self.model_dir, f"{prefix}pca.pkl") if self.config.get('use_pca', False) else None
        self.encoder_path = os.path.join(self.model_dir, f"{prefix}label_encoder.pkl")
        self.metadata_path = os.path.join(self.model_dir, f"{prefix}metadata.json")
        self.metrics_path = os.path.join(self.model_dir, f"{prefix}metrics.json")

        # Model initialization
        self.model_cls = model_cls
        self.model = None
        self.pca = None
        self.encoder = None
        self.scaler = None
        self.y_val_pred_ids = None
        self.y_val_pred = None
        self.y_val_true_ids = None
        self.y_val_true = None
        self.eval_metrics = {}


    def _build_dataset(self, X, y, validation_data=None):
        # Optional class-imbalance warning
        if self.config.get('task', 'classification') == 'classification':
            counts = Counter(y)
            min_pct = min(counts.values()) / len(y)
            if min_pct < 0.05:
                self.logger.warning(f"Detected class imbalance (<5%): {counts}")

        # Scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        joblib.dump(self.scaler, self.scaler_path)
        self.logger.info(f"Scaler saved to {self.scaler_path}")

        # PCA
        if self.config.get('use_pca', False):
            self.pca = PCA(n_components=self.config.get('pca_variance', 0.95))
            X_scaled = self.pca.fit_transform(X_scaled)
            with open(self.pca_path, 'wb') as f:
                pickle.dump(self.pca, f)
            self.logger.info(f"PCA saved to {self.pca_path}")

        # Label encoding
        is_class = self.config.get('task', 'classification') == 'classification'
        if is_class:
            self.encoder = LabelEncoder()
            y_enc = self.encoder.fit_transform(y)
            joblib.dump(self.encoder, self.encoder_path)
            self.logger.info(f"Label encoder saved to {self.encoder_path}")
        else:
            y_enc = np.array(y, dtype=np.float32)

        # Train/val split
        if validation_data:
            X_train, y_train = X_scaled, y_enc
            X_val, y_val = validation_data
            X_val = self.scaler.transform(X_val)
            if self.pca: X_val = self.pca.transform(X_val)
            if is_class: y_val = self.encoder.transform(y_val)
            train_idx = None
            val_idx = None
        else:
            test_size = self.config.get('test_size', 0.2)
            idx = np.arange(len(y_enc))
            #X_train, X_val, y_train, y_val = train_test_split(
            #    X_scaled, y_enc, test_size=test_size,
            #    stratify=y_enc if is_class else None,
            #    random_state=self.config.get('seed', 42)
            #)
            #self.logger.info(f"Train/Val split: {len(y_train)} / {len(y_val)} samples")
            train_idx, val_idx = train_test_split(
                idx, test_size=test_size,
                stratify=y_enc if is_class else None,
                random_state=self.config.get('seed', 42)
            )
            self.logger.info(f"Train/Val split: {len(train_idx)} / {len(val_idx)} samples")
            X_train, y_train = X_scaled[train_idx], y_enc[train_idx]
            X_val, y_val = X_scaled[val_idx], y_enc[val_idx]

        return X_train, y_train, X_val, y_val, val_idx

    def _resolve_metadata_path(self) -> str:
        # prefer label-prefixed path if it exists
        if os.path.isfile(self.metadata_path):
            return self.metadata_path
        # then unprefixed
        base = os.path.join(self.model_dir, "metadata.json")
        if os.path.isfile(base):
            return base
        # then any *_metadata.json
        any_prefixed = _find_first(os.path.join(self.model_dir, "*_metadata.json"))
        return any_prefixed or self.metadata_path

    def train(self, X, y, num_epochs=3000, stop_patience=5, learning_rate=1e-3, val_interval=20, validation_data=None, save_model=True, save_metrics=True):

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()

        X_train, y_train, X_val, y_val, val_idx = self._build_dataset(X, y, validation_data)
        if self.verbose:
            print(f"Training set shape: {X_train.shape}, {y_train.shape}")
            print(f"Testing set shape: {X_val.shape}, {y_val.shape}")

        # Instantiate model
        self.model = self.model_cls(
            input_dim=X_train.shape[1],
            output_dim=np.unique(y_train).shape[0] if self.encoder else y_train.shape[1]
        )

        # Training hyperarameters
        hp = self.config.get('hyperparameters', {})
        lr = hp.get('learning_rate', learning_rate)
        num_epochs = hp.get('num_epochs', num_epochs)
        stop_patience = hp.get('stop_patience', stop_patience)
        val_interval = hp.get('val_interval', val_interval)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss() if self.encoder else torch.nn.MSELoss()

        # Tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long if self.encoder else torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long if self.encoder else torch.float32)
        if self.verbose:
            print("Starting training...")

        loss_curve = []
        best_val_loss = np.inf
        epochs_no_improve = 0
        best_state = None
        for epoch in range(num_epochs):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(X_train_t)
            loss = criterion(out, y_train_t)
            loss.backward()
            optimizer.step()

            loss_curve.append(loss.item())

            if (epoch + 1) % val_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(X_val_t)
                    val_loss = criterion(val_out, y_val_t).item()
                logging.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {loss.item():.8f} | Val Loss: {val_loss:.8f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= stop_patience:
                        logging.info("Early stopping triggered.")
                        break

        # Save model
        print(f" Training complete")
        if best_state is not None:
            self.model.load_state_dict(best_state)
        if save_model:
            print(f" Saving model to {self.model_path}")
            torch.save(self.model.state_dict(), self.model_path)
            print("Model saved")

        # Evaluate on validation dataset
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_val_t)

            if self.encoder:
                pred_labels = torch.argmax(y_pred, dim=1).numpy()
                true_labels = y_val

                # keep ids
                self.y_val_true_ids = np.asarray(true_labels, dtype=int)
                self.y_val_pred_ids = np.asarray(pred_labels, dtype=int)

                # and human-readable strings
                self.y_val_true = self.encoder.inverse_transform(self.y_val_true_ids)
                self.y_val_pred = self.encoder.inverse_transform(self.y_val_pred_ids)

                self.eval_metrics = {
                    'classification_report': classification_report(true_labels, pred_labels, output_dict=True),
                }
            else:
                pred_vals = y_pred.numpy().ravel()
                self.y_val_true = np.asarray(y_val)
                self.y_val_pred = np.asarray(pred_vals)
                self.eval_metrics = {
                    'mse': float(mean_squared_error(y_val, pred_vals)),
                    'mae': float(mean_absolute_error(y_val, pred_vals)),
                    'r2': float(r2_score(y_val, pred_vals))
                }

        print("Training complete. Validation metrics:")
        print(self.eval_metrics)


        if save_metrics:
            with open(self.metrics_path, 'w') as f:
                json.dump(self.eval_metrics, f, indent=2)
            self.logger.info(f"Evaluation metrics saved to {self.metrics_path}")

    def load_model(self):
        with open(self._resolve_metadata_path(), 'r') as f:
            meta = json.load(f)
        self.model = self.model_cls(input_dim=meta['input_dim'], output_dim=meta['output_dim'])
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        self.scaler = joblib.load(self.scaler_path)
        if self.pca_path and os.path.exists(self.pca_path):
            with open(self.pca_path, 'rb') as f:
                self.pca = pickle.load(f)
        if os.path.exists(self.encoder_path):
            self.encoder = joblib.load(self.encoder_path)
        self.logger.info(f"Loaded model weights from {self.model_path}")
        return self.model

    def predict(self, X):
        """
        Predict using the loaded model and scalar.
        If scalar is not yet loaded, attempt to load it from disk.

        Parameters:
            X (np.ndarray): Input data to predict.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.model is None:
            #raise ValueError("Model must be loaded before prediction.")
            self.load_model()

        # Load scalar, pca, encoder if needed
        if self.scaler is None:
            self.scaler = joblib.load(self.scaler_path)

        X_scaled = self.scaler.transform(X)

        if self.pca:
            X_scaled = self.pca.transform(X_scaled)

        if X_scaled.shape[1] != self.model.input_dim:
            raise ValueError(f"Feature dim {X_scaled.shape[1]} != model.input_dim {self.model.input_dim}")

        # Predict
        self.model.eval()
        with torch.no_grad():
            #predictions = self.model(X_tensor).numpy()
            #X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            #logits = self.model(X_tensor)
            out = self.model(torch.tensor(X_scaled, dtype=torch.float32))
            if self.encoder:
                idx = torch.argmax(out, axis=1).numpy()
                return self.encoder.inverse_transform(idx)
            else:
                return out.numpy().ravel()
            #    predictions = out.numpy()
            #if self.label_encoder is not None:
            #    # Classification, return string
            #    y_pred = torch.argmax(logits, axis=1).numpy()
            #    predictions = self.label_encoder.inverse_transform(y_pred)

        #return predictions

    def grid_search(self, X, y, param_grid, cv=5, scoring='accuracy'):
        # Build a sklearn Pipeline for search
        from sklearn.pipeline import Pipeline
        steps = [('scaler', StandardScaler())]
        if self.config.get('use_pca', False):
            steps.append(('pca', PCA()))
        if self.encoder:
            le = LabelEncoder()
            y = le.fit_transform(y)
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        est = RandomForestClassifier() if self.encoder else RandomForestRegressor()
        steps.append(('estimator', est))
        pipe = Pipeline(steps)

        gs = GridSearchCV(pipe, param_grid, cv=cv, scoring=scoring)
        gs.fit(X, y)
        self.logger.info(f"GridSearch best params: {gs.best_params_}")
        return gs.best_estimator_, gs.best_params_

    def cross_validate(self, X, y, k=5):
        #is_class = self.config.get('task', 'classification') == 'classification'
        kf = KFold(n_splits=k, shuffle=True, random_state=self.config.get('seed', 42))
        metrics = []
        for train_idx, val_idx in kf.split(X):
            self.train(X[train_idx], y[train_idx], validation_data=(X[val_idx], y[val_idx]),
                       save_model=False, save_metrics=False)
            metrics.append(self.eval_metrics)
        return metrics

    def build_metadata(self, *, sample_rate_hz: float, window_ms: int, step_ms: int, envelope_cutoff_hz: float,
            selected_channels: Optional[List[int]], channel_names: Optional[List[str]], feature_spec: Dict[str, Any],
            n_features: int, label_classes: List[str], scaler_mean: Optional[List[float]] = None,
            scaler_scale: Optional[List[float]] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Build a self-describing metadata dict the prediction paths can trust.
        """
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        meta = {
            "schema_version": "1.1.0",
            "created_by": "pyoephys.ml.ModelManager",
            "system": {"platform": platform.platform()},
            "model": {
                "class": type(self.model).__name__ if getattr(self, "model", None) is not None else self.model_cls.__name__,
                "path": os.path.relpath(self.model_path, self.root_dir),
                "label_encoder_path": os.path.relpath(self.encoder_path, self.root_dir) if hasattr(self,
                                                                                                         "label_encoder_path") else None,
                "scaler_path": os.path.relpath(self.scaler_path, self.root_dir) if hasattr(self,
                                                                                           "scaler_path") else None,
            },
            "input_dim": int(self.model.input_dim),
            "output_dim": int(self.model.output_dim),
            "data": {
                "sample_rate_hz": float(sample_rate_hz),
                "window_ms": int(window_ms),
                "step_ms": int(step_ms),
                "envelope_cutoff_hz": float(envelope_cutoff_hz),
                "selected_channels": selected_channels,  # indices in raw order used for training (may be None)
                "channel_names": channel_names,  # full list (raw order) if available
            },
            "features": {
                "n_features": int(n_features),
                # feature_spec describes how the feature vector was built
                # (names, per-channel vs global, order, etc.)
                "spec": feature_spec or {},
            },
            "labels": {
                "classes": list(map(str, label_classes)),
            },
            "scaler": {
                "mean": scaler_mean,
                "scale": scaler_scale,
            },
        }
        if extra:
            # attach any training-time metrics or notes
            meta["extra"] = extra
        return meta

    def save_metadata(self, meta: Dict[str, Any]) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(_jsonify(meta), f, indent=2)
        if self.config.get("verbose"):
            print(f"[ModelManager] wrote metadata → {self.metadata_path}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.load_model()
        if self.scaler is None:
            self.scaler = joblib.load(self.scaler_path)

        X_scaled = self.scaler.transform(X)
        if self.pca:
            X_scaled = self.pca.transform(X_scaled)

        if X_scaled.shape[1] != self.model.input_dim:
            raise ValueError(f"Feature dim {X_scaled.shape[1]} != model.input_dim {self.model.input_dim}")

        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X_scaled, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict_topk(self, X: np.ndarray, k: int = 3):
        probs = self.predict_proba(X)
        topk_idx = np.argsort(-probs, axis=1)[:, :k]
        if self.encoder:
            topk_labels = np.array([self.encoder.inverse_transform(idx) for idx in topk_idx])
        else:
            topk_labels = topk_idx
        topk_probs = np.take_along_axis(probs, topk_idx, axis=1)
        return topk_labels, topk_probs
