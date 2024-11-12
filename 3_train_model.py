import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from tensorflow.keras import layers, models
from tensorflow.keras.layers import Normalization

import utilities.emg_processing as emg_proc
import utilities.models as emg_models


def train_emg_classifier(processed_filepath, gesture_label_filepath='gesture_labels.csv', model_type='rf', do_kfold=True, num_folds=5, epochs=100, batch_size=32):

    # Load processed data file
    feature_data = pd.read_csv(processed_filepath)
    print(f"Loaded processed data from {processed_filepath} with shape: {feature_data.shape}")

    # Split features (X) and labels (y)
    X = feature_data.drop(columns=['Gesture']).values
    y = feature_data['Gesture'].values

    if len(np.unique(y)) < 2:
        print("Not enough unique gestures to train the model. Exiting...")
        return None, None
    
    # Encode labels to numerical values
    y_encoded, _ = pd.factorize(y)
    n_gestures = len(np.unique(y_encoded))

    # print out the gestures and corresponding numerical values
    print("Gesture labels and their corresponding numerical values:")
    for i, gesture in enumerate(np.unique(y)):
        print(f"{gesture} -> {i}")

    # Save gesture labels to a file
    gesture_labels = pd.DataFrame({'Gesture': np.unique(y), 'Numerical Value': np.unique(y_encoded)})
    gesture_labels.to_csv(gesture_label_filepath, index=False)

    # Not using one-hot encoding
    #y_one_hot = tf.keras.utils.to_categorical(y_encoded, num_classes=n_gestures)
    
    print(f"Training {model_type.upper()} model...")

    if not do_kfold:
        # We can split up the data 80%/20% for training/testing
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

        if model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"Random Forest accuracy: {acc * 100}%")

        elif model_type == 'rnn':
            # Reshape for RNN (samples, timesteps, features)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            model = emg_models.build_rnn_model((X_train.shape[1], 1), n_gestures)
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
            acc = model.evaluate(X_test, y_test, verbose=1)
            print(f"RNN accuracy: {acc[1] * 100}%")

        elif model_type == 'cnn':
            # Reshape for CNN (samples, features, channels)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            model = emg_models.build_cnn_model((X_train.shape[1], 1), n_gestures)
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
            acc = model.evaluate(X_test, y_test, verbose=1)
            print(f"CNN accuracy: {acc[1] * 100}%")

        elif model_type == 'grnn':
            # Just train a simple GRNN model
            model = emg_models.build_grnn_model()
            model.fit(X_train, y_train)
            y_test_labels = np.argmax(y_test, axis=1)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test_labels, y_pred)
            print(f"GRNN accuracy: {acc * 100}%")

        elif model_type == 'intan':
            # Just Intan cnn model
            model = emg_models.build_intan_nn_model((X_train.shape[1],), n_gestures)
            model.summary()
            print(model.input_shape)
            print(model.output_shape)
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
            #y_test_labels = np.argmax(y_test, axis=1)
            test_loss, acc = model.evaluate(X_test, y_test)
            print(f"Intan CNN accuracy: {acc * 100}%")

    else:
        # Or we can implement K-Fold cross validation to ensure the model's performance is stable across different subsets and avoid overfitting!
        kfold_training(model=model_type, X=X, y_one_hot=y_one_hot, num_folds=num_folds, epochs=epochs, batch_size=batch_size)

    return model, acc

if __name__ == "__main__":

    # Grab the paths from the config file, returning dictionary of paths
    cfg = emg_proc.read_config_file('CONFIG.txt')

    # Train the model
    best_model, best_acc = train_emg_classifier(
                                processed_filepath=cfg['processed_file_path'], # Feature data file name
                                gesture_label_filepath=cfg['gesture_label_file_path'], # Gesture label file name
                                model_type='intan',                               # Model type to train
                                do_kfold=False,                                 # Whether to use K-Fold cross-validation
                                num_folds=2,                                    # Number of data splits to train for cross-validation
                                epochs=200,                                    # Total number of times model seees data
                                batch_size=128,                                # Training samples processed before update
                           )
    if best_model:
        best_model.save(cfg['model_path'])
        print(f"Best model saved at {cfg['model_path']} with accuracy: {best_acc * 100}%")
