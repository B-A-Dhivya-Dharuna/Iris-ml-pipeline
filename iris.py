# ml_pipeline.py

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib
import os

# --- Configuration/Constants ---
TEST_SIZE = 0.3
RANDOM_STATE = 42
TRAIN_DATA_FILE = 'data/iris_train.csv'
TEST_DATA_FILE = 'data/iris_test.csv'
MODEL_DIR = 'models'
MODEL_FILE = f'{MODEL_DIR}/iris_knn_model.pkl'

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Part 1: Download/Split and Save Datasets

def download_and_split_data():
    """Loads the Iris dataset, splits it, and saves the train/test sets to CSV."""
    print("Step 1: Downloading and splitting Iris dataset...")

    # Load data from scikit-learn
    iris = load_iris(as_frame=True)
    df = iris.frame

    # Map the numeric target (0, 1, 2) to species names for clarity
    df['species'] = df['target'].map(lambda x: iris.target_names[x])
    df = df.drop(columns=['target'])

    # Separate features (X) and target (y)
    X = df.drop('species', axis=1)
    y = df['species']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Recombine features and target for saving to CSV
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Save the separate datasets (simulating separate downloads)
    train_df.to_csv(TRAIN_DATA_FILE, index=False)
    test_df.to_csv(TEST_DATA_FILE, index=False)

    print(f"✅ Training data saved to: {TRAIN_DATA_FILE} (Rows: {len(train_df)})")
    print(f"✅ Testing data saved to: {TEST_DATA_FILE} (Rows: {len(test_df)})")

# Execute this step
download_and_split_data()

# Part 2: Train, Evaluate, and Save Model

def train_and_evaluate_model():
    """Loads the split data from CSVs, trains a KNN model, and evaluates it."""
    print("\nStep 2: Training and evaluating ML Model...")

    # Load the separate train and test datasets from disk
    try:
        train_df = pd.read_csv(TRAIN_DATA_FILE)
        test_df = pd.read_csv(TEST_DATA_FILE)
    except FileNotFoundError:
        print("❌ Error: Data files not found. Ensure Part 1 ran successfully.")
        return

    # Prepare the data for scikit-learn
    X_train = train_df.drop('species', axis=1)
    y_train = train_df['species']
    X_test = test_df.drop('species', axis=1)
    y_test = test_df['species']

    # **Build and Train the ML Model (K-Nearest Neighbors)**
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    # Evaluate the Model
    y_pred = model.predict(X_test)

    print("\n--- Model Evaluation (K-Nearest Neighbors) ---")
    print(classification_report(y_test, y_pred))

    # Save the Model for future use (e.g., deployment)
    joblib.dump(model, MODEL_FILE)
    print(f"✅ Trained model saved as: {MODEL_FILE}")

# Execute this step
train_and_evaluate_model()

# Final output to run the full script
if __name__ == '__main__':
    # You can combine the execution for a single script run
    pass

