Realtime gesture prediction from streaming data
=========================================================

This example demonstrates how to use a pre-trained model to predict gestures in real-time from streaming data. The model is trained on EMG data and can classify different gestures based on the input features.

**Requirements:**
- A pre-trained model file (e.g., `model.keras`)
- A recent `.rhd` data file recorded with the Intan RHX system
- A PCA model created from the training data
- Normalization parameters from the training data

**You’ll see how to:**
- Load a pre-trained model, PCA model, and normalization parameters
- Connect to the RHX device and stream data
- Preprocess the streamed data
- Use the model to predict gestures in real-time

----

**Example code:**

.. code-block:: python

    """
    Example: Predict gestures in real-time using a pre-trained model.
    """
    from intan.interface import IntanRHXDevice
    from intan.io import load_labeled_file
    import joblib
    import tensorflow as tf
    import numpy as np
    import os

    from intan.ml import EMGRealTimePredictor

    # === Config (update as needed) ===
    DATA_PATH = r"G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan"
    FILE_PATH = os.path.join(DATA_PATH, r"2024_10_22\raw\RingFlexion_241022_144153\RingFlexion_241022_144153.rhd")
    MODEL_PATH = os.path.join(DATA_PATH, "model.keras")
    PCA_PATH = os.path.join(DATA_PATH, "training_data_pca.pkl")
    NORM_PATH = os.path.join(DATA_PATH, "training_data_norm.npz")
    LABEL_PATH = os.path.join(DATA_PATH, "training_data.npz")
    NOTES_PATH = os.path.join(os.path.dirname(FILE_PATH), "notes.txt")

    # === Load model, PCA, normalization ===
    cue_df = load_labeled_file(NOTES_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    pca = joblib.load(PCA_PATH)
    norm = np.load(NORM_PATH)
    label_names = np.load(LABEL_PATH, allow_pickle=True)["label_names"]
    mean, std = norm["mean"], norm["std"]
    std[std == 0] = 1

    BUFFER_SEC = 1
    WINDOW_MS = 250
    STEP_SEC = 1
    WINDOW_OFFSET_SAMPLES = 200  # <-- Offset from start of buffer (in samples)

    device = IntanRHXDevice(num_channels=128, buffer_duration_sec=1)
    device.enable_wide_channel(range(128))
    device.start_streaming()

    predictor = EMGRealTimePredictor(
        device=device,
        model=model,
        pca=pca,
        mean=mean,
        std=std,
        label_names=label_names,
        cue_df=cue_df,  # optional for ground-truth
        buffer_sec=1,
        window_ms=250,
        step_sec=1,
        window_offset_samples=200
    )

    predictor.run_prediction_loop(background=True)

    try:
        while True:
            msg = predictor.get_message()
            if msg:
                print(msg)
    except KeyboardInterrupt:
        print("Stopping...")
        predictor.stop()
    finally:
        device.stop_streaming()
        device.close()

----
