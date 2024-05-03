import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from argparse import ArgumentParser
from io import StringIO
import pydotplus
from IPython.display import Image
from tabulate import tabulate
from itertools import product


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, low_memory=False, nrows=1000)
    original_attack_cat = df['attack_cat'].copy()
    categorical_cols = df.select_dtypes(include=['object']).columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    df[categorical_cols] = df[categorical_cols].fillna('None')
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    print(df.head())
    return df, original_attack_cat, label_encoders


def feature_selection(df, label_col):
    """ This is for Part 1 of the assignment """
    X = df.drop(['attack_cat', 'Label'], axis=1)
    y = df[label_col]

    estimator = DecisionTreeClassifier()  # Placeholder, could be parameterized
    rfe = RFE(estimator=estimator, n_features_to_select=8)
    rfe.fit(X, y)
    features_df = pd.DataFrame({
        'Feature': X.columns,
        'Selected': rfe.support_,
        'Ranking': rfe.ranking_
    })
    print("\nFeature Selection and Ranking:")
    print(features_df)
    return X.columns[rfe.support_], rfe.transform(X), y


def grid_search(X_train, y_train, X_val, y_val, param_grid, model):
    """ This is for Part 2 of the assignment """
    best_accuracy = 0
    best_params = None
    best_model = None
    keys, values = zip(*param_grid.items())
    for params in product(*values):
        kwargs = dict(zip(keys, params))
        if model == 'DecisionTreeClassifier':
            model = DecisionTreeClassifier(**kwargs)
        elif model == 'RandomForestClassifier':
            model = RandomForestClassifier(**kwargs)
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = kwargs
            best_model = model

    return best_model, best_params


def visualize_decision_tree(model, feature_names):
    """ Only Creates the image if the classifier is DecisionTree"""
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data, filled=True, rounded=True,
                    special_characters=True, feature_names=feature_names)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('decision_tree.png')
    Image(graph.create_png())


def evaluate_model(model, X_test, y_test, column_name, classifier, label_encoders):
    y_predict = model.predict(X_test)
    print("Accuracy: {:.3f}%\n".format(accuracy_score(y_test, y_predict) * 100))
    mse = mean_squared_error(y_test, y_predict)
    rmse = mse ** 0.5
    print("Root Mean Squared Error:", rmse)

    cm = confusion_matrix(y_test, y_predict)
    if column_name == 'Label':
        print(classifier, "Report:")
        print(classification_report(y_test, y_predict, zero_division=0))
        print("Confusion Matrix:")
        cm_df = pd.DataFrame(cm, columns=["Normal", "Attack"], index=["Normal", "Attack"])
        print(tabulate(cm_df, headers='keys', tablefmt='psql'))
    elif column_name == 'attack_cat':
        print("Confusion Matrix:")
        # cm_df = pd.DataFrame(cm, columns=["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic",
        #                                    "Normal", "Reconnaissance", "Shellcode", "Worms"],
        #                      index=["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic", "Normal",
        #                             "Reconnaissance", "Shellcode", "Worms"])
        # print(tabulate(cm_df, headers='keys', tablefmt='psql'))
        y_test_str = preprocess_labels(y_test, label_encoders['attack_cat'])
        y_predict_str = preprocess_labels(y_predict, label_encoders['attack_cat'])

        print(classification_report(y_test_str, y_predict_str, zero_division=0))


def preprocess_labels(labels, label_encoder):
    preprocessed_labels = []
    for label in labels:
        original_label = label_encoder.inverse_transform([label])[0]
        preprocessed_label = original_label.strip()
        if preprocessed_label == "Backdoor":
            preprocessed_label = preprocessed_label.rstrip('s')
        preprocessed_labels.append(str(preprocessed_label))
    return preprocessed_labels


def main():
    parser = ArgumentParser(description='NIDS System')
    parser.add_argument('filepath', help='File path of the dataset')
    parser.add_argument('classifier', choices=['DecisionTreeClassifier', 'RandomForestClassifier'], help='Classifier to use')
    parser.add_argument('task', choices=['Label', 'attack_cat'], help='Prediction task')
    args = parser.parse_args()

    df, original_attack_cat, label_encoders = load_and_preprocess_data(args.filepath)
    selected_features, X, y = feature_selection(df, args.task)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

    if args.classifier == 'DecisionTreeClassifier':
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 3, 5, 10, 15, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 6],
            'max_features': [None,'sqrt', 'log2'],
            'min_impurity_decrease': [0.0, 0.05, 0.1],
            'max_leaf_nodes': [None, 10, 20, 30]
        }
        best_model, best_params = grid_search(X_train, y_train, X_val, y_val, param_grid, args.classifier)
        print("Best Parameters:", best_params)
        visualize_decision_tree(best_model, selected_features)

    elif args.classifier == 'RandomForestClassifier':
        param_grid = {
             'n_estimators': [100, 200, 300],
             'criterion': ['gini', 'entropy'],
             'max_depth': [None, 3, 5, 10, 15, 20, 30],
             'max_features': [None, 'sqrt', 'log2'],
             'min_samples_split': [2, 5, 10],
             'min_samples_leaf': [1, 2, 4, 6],
             'bootstrap': [True, False],
             'min_impurity_decrease': [0.0, 0.01, 0.05],
        }
        best_model, best_params = grid_search(X_train, y_train, X_val, y_val, param_grid, args.classifier)
        print("Best Parameters:", best_params)
    evaluate_model(best_model, X_test, y_test, args.task, args.classifier, label_encoders)


if __name__ == "__main__":
    main()
