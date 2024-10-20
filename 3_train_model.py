import pickle
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_model(input_shape):
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
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_emg_classifier(feature_filepath, training=True):

    # Load the feature data
    print(f"Loading feature data from: {feature_filepath}")
    feature_data = pd.read_csv(feature_filepath)

    # Split the data into training (80%) and testing (20%) sets
    X = feature_data.drop(columns=['Gesture'])
    y = feature_data['Gesture']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert labels to one-hot encoding
    y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=10)

    # Build and train the model
    cnn_model = build_cnn_model((30,))
    cnn_model.fit(X_train, y_train_one_hot, epochs=10, batch_size=32, validation_split=0.2)

    # Create a Random Forrest Classifier model
    if training:
        print("Training the model...")

        # Define the hyperparameters to search
        param_dist = {'n_estimators': randint(50, 200),
                  'max_depth': randint(1, 20)}

        # Create a random forest classifier
        rf = RandomForestClassifier(verbose=5)

        # Use random search to find the best hyperparameters
        rand_search = RandomizedSearchCV(rf,
                                     param_distributions=param_dist,
                                     n_iter=5,
                                     cv=5)

        # Fit the random search object to the data
        rand_search.fit(X_train, y_train)

        # Print the best hyperparameters
        print('Best hyperparameters:', rand_search.best_params_)

    else:
        # Load the best model
        f = open(model_path, 'rb')
        best_rf_model = pickle.load(f)
        f.close()

        # test the model
        print("Testing the model...")
        print(f"Model accuracy: {best_rf_model.score(X_test, y_test)}")




    # TO-DO: Try the prediction with the best model, then move on to using the model in the
    # publication: https://pmc.ncbi.nlm.nih.gov/articles/PMC10669079/

    #y_pred = rf.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred)
    #print("Accuracy:", accuracy)

    #rf_model = RandomForrestModel(n_estimators=100, verbose=3)
    #rf_model.train(X_train, y_train)

    # Evaluate the model
    #print(f"Model accuracy: {rf_model.score(X_test, y_test)}")

    return rand_search.best_estimator_

if __name__ == "__main__":
    # This script trains a gesture classifier model using the feature data extracted from the EMG data

    # Path to the feature data file
    #feature_data_path = 'dataset/processed_data.csv'
    feature_data_path = r'G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\intan_HDEMG_sleeve\processed_data.csv'

    # Path to save the trained model
    model_path = 'dataset/rf_model.pkl'

    # Train the model
    best_rf_model = train_emg_classifier(feature_data_path, training=False)

    # Save the best model
    with open(model_path, 'wb') as f:
        pickle.dump(best_rf_model, f)

