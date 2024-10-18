import pickle
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split


def train_emg_classifier(feature_filepath):

    # Load the feature data
    print(f"Loading feature data from: {feature_filepath}")
    feature_data = pd.read_csv(feature_filepath)

    # Split the data into training (80%) and testing (20%) sets
    X = feature_data.drop(columns=['Gesture'])
    y = feature_data['Gesture']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Random Forrest Classifier model
    print("Training the model...")

    # Define the hyperparameters to search
    param_dist = {'n_estimators': randint(50, 500),
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
    feature_data_path = 'dataset/processed_data.csv'

    # Path to save the trained model
    model_path = 'dataset/rf_model.pkl'

    # Train the model
    best_rf_model = train_emg_classifier(feature_data_path)

    # Save the best model
    with open(model_path, 'wb') as f:
        pickle.dump(best_rf_model, f)

