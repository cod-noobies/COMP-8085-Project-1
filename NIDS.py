import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report
import joblib
import os

def load_data(filepath, nrows=None):
    return pd.read_csv(filepath, low_memory=False, nrows=nrows)

def preprocess_data(df, target):
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = label_encoder.fit_transform(df[col].astype(str))
    X = df.drop([target], axis=1)
    y = df[target]
    return X, y

def select_features_via_rfe(X_train, y_train, num_features=10):
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = RFE(estimator, n_features_to_select=num_features, step=1)
    selector.fit(X_train, y_train)
    selected_features_mask = selector.support_
    selected_feature_names = np.array(X_train.columns)[selected_features_mask]
    return selected_feature_names

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print_classification_report(y_test, predictions)

def print_classification_report(y_test, predictions):
    try:
        report = classification_report(y_test, predictions, target_names=['Normal', 'Intrusion'], output_dict=True)
        print("\nDetailed Performance of the Model:")
        print(f"Accuracy: {report['accuracy']:.2%}")

        for label in ['Normal', 'Intrusion']:
            print(f"\nCategory '{label}':")
            print(f"  Precision: {report[label]['precision']:.2%}")
            print(f"  Recall: {report[label]['recall']:.2%}")
            print(f"  F1-Score: {report[label]['f1-score']:.2%}")
            print(f"  Support: {report[label]['support']}")

    except Exception as e:
        print("Failed to generate classification report:", str(e))

def main():
    if len(sys.argv) != 3:
        print("Usage: python NIDS.py <data.csv> <task>")
        sys.exit(1)

    data_path = sys.argv[1]
    task = sys.argv[2]

    df = load_data(data_path, nrows=10000)
    X, y = preprocess_data(df, task)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

    selected_feature_names = select_features_via_rfe(X_train_scaled, y_train, num_features=10)
    print("Features selected by RFE:", ', '.join(selected_feature_names))

    X_train_selected = X_train_scaled[selected_feature_names]
    X_test_selected = X_test_scaled[selected_feature_names]

    model_filename = f"RandomForest_{task}_model.joblib"
    if os.path.exists(model_filename):
        model = joblib.load(model_filename)
    else:
        model = train_model(X_train_selected, y_train)
        joblib.dump(model, model_filename)

    evaluate_model(model, X_test_selected, y_test)  # Make sure to use X_test_selected

if __name__ == "__main__":
    main()
