Gesture Classification Pipeline
================================

This example demonstrates a complete machine learning pipeline for training gesture classifiers from EMG data. The workflow includes data preprocessing, feature extraction using PCA, and model training with TensorFlow/Keras.

**Requirements:**

- TensorFlow 2.19.0
- scikit-learn (for PCA)
- CUDA-compatible GPU (recommended for faster training)
- EMG data files (`.rhd`, `.npz`, or `.csv` formats supported)

**Overview:**

The gesture classification pipeline consists of several stages:

1. **Data Preparation**: Load and preprocess EMG data from multiple file formats
2. **Feature Extraction**: Apply PCA for dimensionality reduction
3. **Model Training**: Train a neural network classifier
4. **Real-time Prediction**: Use trained model for live gesture recognition

----

Stage 1: Building Training Datasets
-------------------------------------

**New in v0.1.0**: Unified dataset builder with GUI prompts and support for all file formats.

The pipeline uses a single script ``1_build_dataset.py`` that supports RHD, NPZ, CSV, and Poly5 formats.

**Interactive Mode (Recommended):**

.. code-block:: bash

    # Launch with interactive prompts
    python examples/gesture_classifier/1_build_dataset.py

    # You'll be prompted for:
    # - Project root directory
    # - Single file or multi-file mode
    # - Files to exclude (optional)

**Command-Line Mode:**

.. code-block:: bash

    # Single file with channel selection
    python 1_build_dataset.py \
        --root_dir /data \
        --file_type rhd \
        --file_path recording.rhd \
        --channels 0:64 \
        --overwrite

    # Multi-file with channel mapping
    python 1_build_dataset.py \
        --root_dir /data \
        --multi_file \
        --channel_map 8-8-L \
        --exclude_pattern test \
        --overwrite

    # CSV with IMU features
    python 1_build_dataset.py \
        --root_dir /data \
        --file_type csv \
        --multi_file \
        --modality both \
        --imu_features rich \
        --overwrite

**Configuration File:**

.. code-block:: bash

    # Use shared config file (.gesture_config)
    python 1_build_dataset.py --config_file config.json --overwrite

**Key Features:**

- Automatic event file discovery (checks ``events/`` subdirectory)
- Smart file filtering with exclude patterns
- HD-EMG grid support with spatial transforms
- Multi-modal EMG+IMU feature extraction
- Paper-style preprocessing mode (120Hz highpass, RMS only)
- Poly5 format support

**Legacy Scripts:**

Previous format-specific scripts (1a-1e) are still available but deprecated:

- ``1a_build_training_dataset_rhd.py`` - Single RHD file
- ``1b_build_training_dataset_multi_rhd.py`` - Multiple RHD files
- ``1c_build_training_dataset_npz.py`` - Single NPZ file
- ``1d_build_training_dataset_multi_npz.py`` - Multiple NPZ files
- ``1e_build_training_dataset_any.py`` - Auto-detect format

----

Stage 2: Training the Model
-----------------------------

Once you have prepared your training dataset, train a classifier:

.. code-block:: bash

    # examples/gesture_classifier/2_train_model.py
    python 2_train_model.py --data_path training_data.npz --model_output model.keras

**Example code:**

.. code-block:: python

    """
    Train a gesture classification model
    """
    from intan.ml import train_classifier, ModelManager

----

Stage 3: Making Predictions
---------------------------

**New in v0.1.0**: Unified prediction CLI with 4 modes.

The ``3_predict.py`` script provides a single entry point for all prediction workflows:

**File Mode** - Offline prediction from single RHD file:

.. code-block:: bash

    python 3_predict.py file \
        --root_dir /data \
        --file_path recording.rhd \
        --label 128ch \
        --verbose

**Batch Mode** - Process multiple files with aggregated metrics:

.. code-block:: bash

    python 3_predict.py batch \
        --root_dir /data \
        --rhd_glob "raw/**/*.rhd" \
        --events_dir events/ \
        --label 128ch \
        --save_eval

**Record Mode** - Fixed-duration device recording:

.. code-block:: bash

    python 3_predict.py record \
        --root_dir /data \
        --label 128ch \
        --seconds 10 \
        --verbose

**Stream Mode** - Real-time continuous prediction:

.. code-block:: bash

    python 3_predict.py stream \
        --root_dir /data \
        --label 128ch \
        --infer_hz 20 \
        --smooth_k 5 \
        --use_lsl

**Configuration:**

Uses shared ``.gesture_config`` file for default values. Interactive prompts when arguments missing.

**Legacy Scripts:**

Previous prediction scripts are still available but deprecated:

- ``3a_predict_from_rhd.py`` - Single file prediction
- ``3b_batch_predict_from_rhd.py`` - Batch processing
- ``3c_predict_from_device_record.py`` - Device recording
- ``3d_predict_from_device_realtime.py`` - Real-time streaming

.. code-block:: python

    import numpy as np
    from intan.ml import ModelManager

    # Load training data
    data = np.load('training_data.npz')
    X_train = data['features']
    y_train = data['labels']

    # Initialize model manager
    manager = ModelManager()

    # Train model (automatically applies PCA and normalization)
    model, pca, scaler = manager.train_model(
        X_train, y_train,
        model_type='CNN',
        n_components=20,
        epochs=50,
        batch_size=32,
    )

    # Save trained model
    manager.save_model('model.keras')
    manager.save_pca('pca_model.pkl')
    manager.save_normalization('norm_params.npz')

**GPU Training:**

For faster training, ensure CUDA is installed:

.. code-block:: bash

    # Verify GPU availability
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

See the `gesture_classifier README <https://github.com/Neuro-Mechatronics-Interfaces/python-intan/tree/main/examples/gesture_classifier>`_ for detailed GPU setup instructions.

----

Stage 3: Prediction
--------------------

**3a. Predict from .rhd file:**

.. code-block:: bash

    # examples/gesture_classifier/3a_predict_from_rhd.py
    python 3a_predict_from_rhd.py --file data.rhd --model model.keras

**3b. Batch predict from multiple .rhd files:**

.. code-block:: bash

    # examples/gesture_classifier/3b_batch_predict_from_rhd.py
    python 3b_batch_predict_from_rhd.py --data_dir /path/to/rhd/files

**3c. Predict from recorded device data:**

.. code-block:: python

    """
    Predict gestures from a recorded RHX session
    Script: examples/gesture_classifier/3c_predict_from_device_record.py
    """
    from intan.interface import IntanRHXDevice
    from intan.ml import ModelManager
    import tensorflow as tf

    # Load model
    model = tf.keras.models.load_model('model.keras')
    manager = ModelManager(model=model)
    manager.load_pca('pca_model.pkl')
    manager.load_normalization('norm_params.npz')

    # Load recorded data and predict
    result = load_rhd_file('recorded_session.rhd')
    predictions = manager.predict(result['amplifier_data'])

**3d. Real-time prediction from live stream:**

.. code-block:: python

    """
    Real-time gesture prediction from streaming device
    Script: examples/gesture_classifier/3d_predict_from_device_realtime.py
    """
    from intan.interface import IntanRHXDevice
    from intan.ml import EMGRealTimePredictor
    import tensorflow as tf

    # Initialize device
    device = IntanRHXDevice(num_channels=128, buffer_duration_sec=1)
    device.enable_wide_channel(range(128))
    device.start_streaming()

    # Load model and create predictor
    model = tf.keras.models.load_model('model.keras')
    predictor = EMGRealTimePredictor(
        device=device,
        model=model,
        pca=pca,
        mean=mean,
        std=std,
        label_names=label_names,
        window_ms=250,
        buffer_sec=1
    )

    # Run prediction loop
    predictor.run_prediction_loop()

    try:
        while True:
            prediction = predictor.get_prediction()
            if prediction:
                print(f"Detected: {prediction}")
    except KeyboardInterrupt:
        predictor.stop()
        device.close()

----

Additional Tools
-----------------

**Cross-orientation analysis:**

Analyze model performance across different arm orientations:

.. code-block:: bash

    # examples/gesture_classifier/plot_cross_orientation.py
    python plot_cross_orientation.py --results_dir /path/to/results

**Alternative training script:**

.. code-block:: bash

    # examples/gesture_classifier/2b_train_model.py
    python 2b_train_model.py --config config.json

----

Expected Outcomes
------------------

- **Stage 1**: `training_data.npz` file containing preprocessed EMG features and labels
- **Stage 2**: Trained model files (`model.keras`, `pca_model.pkl`, `norm_params.npz`)
- **Stage 3**: Gesture predictions with confidence scores

**Performance Tips:**

1. Use GPU acceleration for training (20-50x speedup)
2. Experiment with different PCA components (10-50 typically work well)
3. Balance your training dataset across gesture classes
4. Use data augmentation for small datasets
5. Fine-tune window size (200-300ms works well for EMG)

----

See Also
---------

- :doc:`realtime_predict` - Real-time prediction details
- :doc:`../info/signal_processing` - Signal processing techniques
- API Reference: :mod:`intan.ml`
