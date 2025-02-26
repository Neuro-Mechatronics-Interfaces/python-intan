""" This script will train a machine learning model to classify EMG data into different gestures.

 Author: Jonathan Shulgach
 Last Modified: 11/15/2024
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from tensorflow.keras import layers, models
from tensorflow.keras.layers import Normalization

from utilities import ml_utilities as emg_models
from utilities import emg_processing as emg_proc



def train_emg_classifier(config_filepath, model_type, epochs=100, batch_size=32, do_kfold=False, num_folds=5, save_model=True):

    # Load processed data file
    cfg = emg_proc.read_config_file(config_filepath)
    processed_filepath = os.path.join(cfg['root_directory'], cfg['processed_data_filename'])
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
    gesture_label_filepath = os.path.join(cfg['root_directory'], cfg['gesture_label_filename'])
    gesture_labels.to_csv(gesture_label_filepath, index=False)

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

        elif model_type == 'rnn':
            # Reshape for RNN (samples, timesteps, features)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            model = emg_models.build_rnn_model((X_train.shape[1], 1), n_gestures)
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
            acc = model.evaluate(X_test, y_test, verbose=1)

        elif model_type == 'cnn':
            # Reshape for CNN (samples, features, channels)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            model = emg_models.build_cnn_model((X_train.shape[1], 1), n_gestures)
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
            acc = model.evaluate(X_test, y_test, verbose=1)

        elif model_type == 'grnn':
            # Just train a simple GRNN model
            model = emg_models.build_grnn_model()
            model.fit(X_train, y_train)
            y_test_labels = np.argmax(y_test, axis=1)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test_labels, y_pred)

        elif model_type == 'intan':
            # Just Intan cnn model
            model = emg_models.build_intan_nn_model((X_train.shape[1],), n_gestures)
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
            test_loss, acc = model.evaluate(X_test, y_test)

    else:
        # Or we can implement K-Fold cross validation to ensure the model's performance is stable across different subsets and avoid overfitting!
        kfold_training(model=model_type, X=X, y_one_hot=y_one_hot, num_folds=num_folds, epochs=epochs, batch_size=batch_size)

    print(f"Model accuracy: {acc * 100}%")
    with open(os.path.join(cfg['root_directory'], 'model_accuracy.txt'), 'w') as f:
        f.write(f"Model accuracy: {acc * 100}%")

    if save_model:
        model_filepath = os.path.join(cfg['root_directory'], cfg['model_filename'])
        model.save(model_filepath)
        print(f"Best model saved at {model_filepath} with accuracy: {acc * 100}%")

if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Preprocess EMG data to extract gesture timings.')
    parser.add_argument('--config_path', type=str, default='config.txt', help='Path to the config file containing the directory of .rhd files.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training the model.')
    parser.add_argument('--do_kfold', type=bool, default=False, help='Whether to use K-Fold cross-validation.')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of data splits to train for cross-validation.')
    args = parser.parse_args()

    # Train the model
    train_emg_classifier(args.config_path, 'intan', args.epochs, args.batch_size, args.do_kfold, args.num_folds, save_model=True)
    print("Step 3: Model training Done!")
