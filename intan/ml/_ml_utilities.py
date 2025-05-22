# intan/ml_utilities.py

import numpy as np
import time
import threading
import queue

class EMGRealTimePredictor:
    def __init__(
        self,
        device,
        model,
        pca,
        mean,
        std,
        label_names,
        cue_df=None,
        buffer_sec=1,
        window_ms=250,
        step_sec=1,
        window_offset_samples=0,
        feature_list=None
    ):
        self.device = device
        self.model = model
        self.pca = pca
        self.mean = mean
        self.std = std
        self.label_names = label_names
        self.cue_df = cue_df
        self.buffer_sec = buffer_sec
        self.window_ms = window_ms
        self.step_sec = step_sec
        self.window_offset_samples = window_offset_samples
        self.feature_list = feature_list or [
            'mean_absolute_value',
            'zero_crossings',
            'slope_sign_changes',
            'waveform_length',
            'root_mean_square'
        ]
        self.fs = self.device.sample_rate
        self.total = 0
        self.correct = 0
        self.stop_event = threading.Event()
        self.print_queue = queue.Queue()

    def apply_filters(self, emg):
        from intan.processing import notch_filter, bandpass_filter, lowpass_filter, rectify
        emg = notch_filter(emg, self.fs, 60)
        emg = bandpass_filter(emg, 20, 450, self.fs)
        emg = rectify(emg)
        return lowpass_filter(emg, 5, self.fs)

    def extract_and_classify(self, window, already_filtered=True):
        from intan.processing import extract_features
        if not already_filtered:
            window = self.apply_filters(window)
        feats = extract_features(window, self.feature_list)
        feats = (feats - self.mean) / self.std
        if np.any(np.isnan(feats)):
            return "NaN", 0.0, -1
        feats = self.pca.transform(feats.reshape(1, -1))
        pred = self.model.predict(feats, verbose=0)
        gesture_idx = np.argmax(pred)
        gesture_name = self.label_names[gesture_idx]
        confidence = np.max(pred)
        return gesture_name, confidence, gesture_idx

    def get_label_for_sample(self, sample_idx):
        if self.cue_df is None:
            return None
        for i in range(len(self.cue_df) - 1):
            start = self.cue_df.loc[i, "Sample"]
            end = self.cue_df.loc[i + 1, "Sample"]
            label = str(self.cue_df.loc[i, "Label"])
            if label.lower() != "none" and start <= sample_idx < end:
                return label
        return None

    def run_prediction_loop(self, background=True):
        if background:
            thread = threading.Thread(target=self._prediction_worker, daemon=True)
            thread.start()
            return thread
        else:
            self._prediction_worker()

    def _prediction_worker(self):
        buffer_samples = int(self.buffer_sec * self.fs)
        window_samples = int(self.window_ms * self.fs / 1000)
        while not self.stop_event.is_set():
            time.sleep(self.step_sec)
            full_buffer = self.device.get_latest_window(self.buffer_sec * 1000)
            if full_buffer.shape[1] != buffer_samples:
                self.print_queue.put(f"Buffer size mismatch: expected {buffer_samples}, got {full_buffer.shape[1]}")
                continue

            filtered_buffer = self.apply_filters(full_buffer)
            start = self.window_offset_samples
            end = start + window_samples

            if start < 0 or end > buffer_samples:
                self.print_queue.put(f"Window out of bounds: start {start}, end {end}, buffer size {buffer_samples}")
                continue

            window = filtered_buffer[:, start:end]

            if window.shape[1] != window_samples:
                self.print_queue.put(f"Window size mismatch: expected {window_samples}, got {window.shape[1]}")
                continue

            if np.any(np.isnan(window)):
                self.print_queue.put("NaN detected in window, skipping prediction.")
                continue

            gesture_name, confidence, gesture_idx = self.extract_and_classify(window, already_filtered=True)
            if gesture_name == "NaN":
                self.print_queue.put("NaN detected in prediction, skipping.")
                continue

            # === Ground truth label from notes ===
            label = self.get_label_for_sample(start)
            msg = ""
            if label and str(label).lower() != "nan":
                label = label.strip().lower()
                self.total += 1
                try:
                    true_idx = next(i for i, name in enumerate(self.label_names) if label in name.lower())
                except StopIteration:
                    true_idx = None

                if true_idx is not None and gesture_idx == true_idx:
                    self.correct += 1
                    status = "✅"
                else:
                    status = "❌"
                msg = (f"{status} [{start / self.fs:.2f}-{end / self.fs:.2f}s] → {gesture_name} | True: "
                       f"{self.label_names[true_idx] if true_idx is not None else label} | Conf: {confidence:.2f}")
            else:
                msg = (f"⚠️  [{start / self.fs:.2f}-{end / self.fs:.2f}s] → {gesture_name} | No ground truth | Conf: {confidence:.2f}")

            self.print_queue.put(msg)
        # On exit
        if self.total > 0:
            self.print_queue.put(f"\nFinal accuracy: {self.correct}/{self.total} ({100 * self.correct / self.total:.2f}%)")

    def stop(self):
        self.stop_event.set()

    def get_message(self, timeout=0.5):
        try:
            return self.print_queue.get(timeout=timeout)
        except queue.Empty:
            return None
