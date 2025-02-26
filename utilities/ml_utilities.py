import tensorflow as tf
import numpy as np
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


# ==============================
# Model building
# ==============================
class EMGCNN(nn.Module):
    def __init__(self, num_classes=7, input_channels=8):
        super(EMGCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(1, stride=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(1, stride=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        #self.fc = nn.Linear(64 * 5, num_classes)  # Adjust based on your final feature map size
        # ✅ Compute final feature map size dynamically (Adjust this based on pooling)
        self.fc = nn.Linear(64 * 1, num_classes)  # Adjusted for sequence_length = 1
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.activation(x)

    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"Saved trained model to {path}")

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
    https://www.nature.com/articles/s41598-024-64458-x
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
    model.summary()

    return model


# ==============================
# Training & Evaluation
# ==============================
def train_pytorch_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        # Validation
        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss / len(val_loader))
        scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    return train_losses, val_losses, val_accuracies


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


# ==============================
# Transformations
# ==============================
def convert_lists_to_tensors(X_list, y_list):
    """Converts lists of EMG features and labels into PyTorch tensors."""
    X_tensor = torch.tensor(np.vstack(X_list), dtype=torch.float32).unsqueeze(-1)
    y_tensor = torch.tensor(np.concatenate(y_list), dtype=torch.long)

    print(f"Final X tensor shape: {X_tensor.shape}, Final y tensor shape: {y_tensor.shape}")
    return X_tensor, y_tensor


# ==============================
# Plotting
# ==============================

def plot_training_metrics(train_losses, val_losses, val_accuracies, save_fig=False):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    if save_fig:
        plt.savefig("loss_plot.png")
    plt.show()

    plt.figure()
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.legend()
    if save_fig:
        plt.savefig("accuracy_plot.png")
    plt.show()
