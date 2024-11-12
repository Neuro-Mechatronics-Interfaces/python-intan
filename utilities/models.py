import tensorflow as tf
import numpy as np
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping
from scipy.spatial.distance import cdist

class GRNN:
    def __init__(self, sigma=0.1):
        self.sigma = sigma
        self.train_X = None
        self.train_y = None

    def fit(self, X, y):
        # Store training data
        self.train_X = X
        self.train_y = y

    def predict(self, X):
        if self.train_X is None or self.train_y is None:
            raise ValueError("Model has not been trained yet. Please call `fit` before `predict`.")

        # Compute Euclidean distances between input and all training samples
        distances = cdist(X, self.train_X, metric='euclidean')
        # Compute Gaussian kernel (radial basis function)
        kernels = np.exp(-distances ** 2 / (2 * self.sigma ** 2))

        # Predict: weighted sum of the target values
        weighted_sum = np.dot(kernels, self.train_y)
        # Normalize by sum of kernel weights
        predictions = weighted_sum / np.sum(kernels, axis=1).reshape(-1, 1)

        # Convert predictions to discrete class labels using argmax
        return np.argmax(predictions, axis=1)


def build_grnn_model():
    #Build the GRNN model
    model = GRNN(sigma=0.5)
    return model

def build_cnn_model(input_shape, num_classes):

    # Build the model
    model = models.Sequential([

        # Add the pre-normalized data as the input layer
        layers.InputLayer(input_shape=input_shape),

        # Fully Connected Layer 1 with Dropout
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),

        # Fully Connected Layer 2 with Dropout
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),

        # Output Layer
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def build_new_cnn_model(input_shape, num_classes):
    """Build a 1D CNN model for EMG data classification.
    """
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def build_rnn_model(input_shape, num_classes):
    """https://ieeexplore.ieee.org/document/9669872"""
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def build_intan_nn_model(input_shape, num_classes):
    """Build a 1D CNN model for EMG data classification.
    """
    print(f"Input shape: {input_shape}")
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Dense(256, activation='relu'))  # Set input_dim properly here
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Assuming classification with multiple classes

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def kfold_training(model='cnn', num_folds=5, epochs=50, batch_size=32):
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_no = 1
    accuracies = []

    # K-Fold Cross Validation
    best_model = None
    best_accuracy = 0
    # for train_idx, test_idx in kfold.split(X, y_encoded):
    for train_idx, test_idx in kfold.split(X, y_one_hot):
        print(f"Training on fold {fold_no}...")

        # Split the data into training and testing sets for this fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_one_hot[train_idx], y_one_hot[test_idx]
        # y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        if model_type == 'cnn':
            # Build adn train the CNN model
            model = build_cnn_model((X_train.shape[1],), n_gestures, X_train)
            # model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
            model.fit(X_train, tf.keras.utils.to_categorical(y_train, num_classes=len(np.unique(y_encoded))),
                      epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
        if model_type == 'rnn':
            # Reshape X to have third dimension for RNN
            X_train = np.expand_dims(X_train, axis=-1)
            X_test = np.expand_dims(X_test, axis=-1)
            # Build and train the RNN model
            model = build_rnn_model((X_train.shape[1], X_train.shape[2]), n_gestures)
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

        # Evaluate the model on the test set
        scores = model.evaluate(X_test, y_test, verbose=0)
        accuracy = scores[1]  # Accuracy at index 1
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
