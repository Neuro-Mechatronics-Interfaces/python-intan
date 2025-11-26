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

The pipeline supports multiple data sources. Choose the appropriate script based on your data format:

**From single .rhd file:**

.. code-block:: python

    """
    Build training dataset from a single .rhd file
    Script: examples/gesture_classifier/1a_build_training_dataset_rhd.py
    """
    from intan.io import load_rhd_file
    from intan.processing import filter_emg, notch_filter, extract_windows
    import numpy as np

    # Load EMG data
    result = load_rhd_file('path/to/file.rhd')
    emg_data = result['amplifier_data']
    fs = result['frequency_parameters']['amplifier_sample_rate']

    # Preprocess: filter and segment by gesture labels
    emg_filtered = notch_filter(emg_data, fs, f0=60)
    emg_filtered = filter_emg(emg_filtered, 'bandpass', fs, lowcut=20, highcut=500)

    # Extract windows and labels (assumes you have a labels file)
    # See the full script for complete implementation

**From multiple .rhd files:**

.. code-block:: bash

    # examples/gesture_classifier/1b_build_training_dataset_multi_rhd.py
    python 1b_build_training_dataset_multi_rhd.py --data_dir /path/to/rhd/files

**From .npz files:**

.. code-block:: bash

    # examples/gesture_classifier/1c_build_training_dataset_npz.py
    python 1c_build_training_dataset_npz.py --data_dir /path/to/npz/files

**From .csv files:**

.. code-block:: bash

    # examples/gesture_classifier/1e_build_training_dataset_multi_csv.py
    python 1e_build_training_dataset_multi_csv.py --data_dir /path/to/csv/files

**From any format (automatic detection):**

.. code-block:: bash

    # examples/gesture_classifier/1e_build_training_dataset_any.py
    python 1e_build_training_dataset_any.py --data_dir /path/to/files

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
    import numpy as np

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
        batch_size=32
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
