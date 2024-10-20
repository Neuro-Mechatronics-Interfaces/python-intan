import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models


def build_cnn_model(input_shape, num_classes):
    # https://pmc.ncbi.nlm.nih.gov/articles/PMC10669079/
    model = models.Sequential()

    # Input Layer
    model.add(layers.InputLayer(input_shape=input_shape))

    # Fully Connected Layer 1 with Dropout
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.2))

    # Fully Connected Layer 2 with Dropout
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.2))

    # Output Layer (10 classes with softmax)
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_emg_classifier(root_dir, feature_filename='processed_data.csv', num_folds=5, epochs=100, batch_size=32):

    # Load processed data file
    feature_filepath = os.path.abspath(os.path.join(root_dir, feature_filename))
    try:
        print(f"Loading feature data from: {feature_filepath}")
        feature_data = pd.read_csv(feature_filepath)
    except FileNotFoundError:
        print(f"File {feature_filepath} not found. Please check the path and file location.")
        return

    # Split features (X) and labels (y)
    X = feature_data.drop(columns=['Gesture'])
    y = feature_data['Gesture']
    
    # Encode labels to numerical values
    y_encoded, _ = pd.factorize(y)
    
    # Convert labels to one-hot encoding
    n_gestures = len(np.unique(y_encoded))
    y_one_hot = tf.keras.utils.to_categorical(y_encoded, num_classes=n_gestures)
    
    print("Training model...")
    # We can split up the data 80%/20% for training/testing, and fit the model, doing all this once...
    #X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
    #cnn_model = build_cnn_model((30,), n_gestures)
    #cnn_model.fit(X_train, y_train_one_hot, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    
    # Or we can implement K-Fold cross validation to ensure the model's performance is stable across different 
    # subsets and avoid overfitting!
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_no = 1
    accuracies = []

    # K-Fold Cross Validation
    best_model = None
    best_accuracy = 0
    for train_idx, test_idx in kfold.split(X, y_one_hot):
        print(f"Training on fold {fold_no}...")
        
        # Split the data into training and testing sets for this fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_one_hot[train_idx], y_one_hot[test_idx]

        # Build the CNN model
        model = build_cnn_model((X_train.shape[1],), n_gestures)

        # Train the model
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

        # Evaluate the model on the test set
        scores = model.evaluate(X_test, y_test, verbose=0)
        accuracy = scores[1] # Accuracy at index 1
        print(f"Fold {fold_no} - Test accuracy: {scores[1] * 100}%")

        # Store the accuracy for this fold
        accuracies.append(scores[1])

        # Save the model if it is the best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            
        fold_no += 1

    # Calculate and print the average accuracy across all folds
    avg_accuracy = np.mean(accuracies)
    print(f"Average test accuracy across {num_folds} folds: {avg_accuracy * 100}%")

    return best_model, best_accuracy
        
        
if __name__ == "__main__":

    # Define the path to the feature data
    #root_dir = '/mnt/g/if/using/google/drive'
    #root_dir = r'C:\absolute\path\to\your\folder'
    root_dir = '/home/nml/also/works/with/wsl2'

    # Train the model
    best_model, best_acc = train_emg_classifier(root_dir, 
                                feature_filename='processed_data.csv', # Feature data file name
                                num_folds=5,                           # Number of data splits to train for cross-validation
                                epochs=150,                            # Total number of times model seees data
                                batch_size=128,                        # Training samples processed before update
                           )

    if best_model:
        best_model_path = os.path.join(root_dir, 'best_cnn_model.h5')
        best_model.save(best_model_path)
        print(f"Best model saved at {best_model_path} with accuracy: {best_acc * 100}%")
