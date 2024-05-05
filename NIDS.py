import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, \
    precision_score, recall_score, f1_score
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
    df = pd.read_csv(filepath, low_memory=False)
    categorical_cols = df.select_dtypes(include=['object']).columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    df[categorical_cols] = df[categorical_cols].fillna('None')
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    if 'attack_cat' in categorical_cols:
        df['attack_cat'] = df['attack_cat'].apply(lambda x: x.strip())
        df['attack_cat'] = df['attack_cat'].replace({'Backdoor': 'Backdoors'})

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    print(df.head())
    return df, label_encoders


def feature_selection(df, label_col):
    """ This is for Part 1 of the assignment """
    X = df.drop(['attack_cat', 'Label'], axis=1)
    y = df[label_col]

    estimator = RandomForestClassifier()
    rfe = RFE(estimator=estimator, n_features_to_select=8)
    rfe.fit(X, y)
    features_df = pd.DataFrame({
        'Feature': X.columns,
        'Selected': rfe.support_,
        'Ranking': rfe.ranking_
    })
    print("\nFeature Selection and Ranking:")
    print(features_df)
    print("\nSelected Features:", X.columns[rfe.support_])
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
            current_model = DecisionTreeClassifier(**kwargs)
        elif model == 'RandomForestClassifier':
            current_model = RandomForestClassifier(**kwargs)
        current_model.fit(X_train, y_train)
        predictions = current_model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = kwargs
            best_model = current_model

    return best_model, best_params


def visualize_decision_tree(model, feature_names, task):
    """ Only Creates the image if the classifier is DecisionTree"""
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data, filled=True, rounded=True,
                    special_characters=True, feature_names=feature_names)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(f'decision_tree_{task}.png')
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
        y_test_str = preprocess_labels(y_test, label_encoders['attack_cat'])
        y_predict_str = preprocess_labels(y_predict, label_encoders['attack_cat'])
        print(classifier, "Report:")
        print(classification_report(y_test_str, y_predict_str, zero_division=0))
        micro_precision = precision_score(y_test, y_predict, average='micro')
        micro_recall = recall_score(y_test, y_predict, average='micro')
        micro_f1 = f1_score(y_test, y_predict, average='micro')
        print(f"Micro Average Precision: {micro_precision:.6f}")
        print(f"Micro Average Recall: {micro_recall:.6f}")
        print(f"Micro Average F1-score: {micro_f1:.6f}")
        print("Confusion Matrix:")
        cm_df = pd.DataFrame(cm, columns=["Analysis", "Backdoors", "DoS", "Exploits", "Fuzzers", "Generic",
                                          "Normal", "Reconnaissance", "Shellcode", "Worms"],
                             index=["Analysis", "Backdoors", "DoS", "Exploits", "Fuzzers", "Generic", "Normal",
                                    "Reconnaissance", "Shellcode", "Worms"])
        print(tabulate(cm_df, headers='keys', tablefmt='psql'))

    if model == 'DecisionTreeClassifier':
        visualize_decision_tree(model, X_test.columns, column_name)


def preprocess_labels(labels, label_encoder):
    original_labels = []
    for label in labels:
        original_label = label_encoder.inverse_transform([label])[0]
        original_labels.append(original_label)
    return original_labels


def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def main():
    parser = ArgumentParser(description='NIDS System')
    parser.add_argument('test_file', help='File path of the dataset')
    parser.add_argument('classifier', choices=['DecisionTreeClassifier', 'RandomForestClassifier'], help='Classifier to use')
    parser.add_argument('task', choices=['Label', 'attack_cat'], help='Prediction task')
    parser.add_argument('model_file', nargs='?', help='Optional model file to load', default='')
    args = parser.parse_args()

    df, label_encoders = load_and_preprocess_data(args.test_file)
    y = df[args.task]
    # selected_features, X, y = feature_selection(df, args.task)
    selected_features = ['srcip', 'dstip', 'sttl', 'Dload', 'smeansz', 'dmeansz', 'tcprtt', 'ct_state_ttl']
    X = df[selected_features]
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)
    if args.model_file and os.path.exists(args.model_file):
        best_model = load_model(args.model_file)
    else:
        if args.classifier == 'DecisionTreeClassifier':
            '''Decision making Tree Label'''
            # # For Part 2
            # if args.task == 'Label':
            #     best_model = DecisionTreeClassifier(criterion='gini',
            #                                         splitter='best',
            #                                         max_depth=20,
            #                                         min_samples_split=10,
            #                                         min_samples_leaf=4,
            #                                         max_features='log2',
            #                                         min_impurity_decrease=0.0,
            #                                         max_leaf_nodes=20)
            # # For Part 3
            # elif args.task == 'attack_cat':
            #     best_model = DecisionTreeClassifier(criterion='gini',
            #                                         splitter='best',
            #                                         max_depth=15,
            #                                         min_samples_split=10,
            #                                         min_samples_leaf=1,
            #                                         max_features='None',
            #                                         min_impurity_decrease=0.0,
            #                                         max_leaf_nodes=None)
            # best_model.fit(X_train, y_train)
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
        elif args.classifier == 'RandomForestClassifier':
            # # For Part 2
            # if args.task == 'Label':
            #     best_model = RandomForestClassifier(criterion='entropy',
            #                                         n_estimators=50,
            #                                         max_depth=50,
            #                                         max_features='sqrt',
            #                                         min_samples_split=2,
            #                                         min_samples_leaf=1,
            #                                         bootstrap=True,
            #                                         min_impurity_decrease=0.0,
            #                                         random_state=70)
            # # For Part 3
            # elif args.task == 'attack_cat':
            #     best_model = RandomForestClassifier(criterion='gini',
            #                                         n_estimators=50,
            #                                         max_depth=30,
            #                                         max_features='log2',
            #                                         min_samples_split=10,
            #                                         min_samples_leaf=1,
            #                                         bootstrap=False,
            #                                         min_impurity_decrease=0.0,
            #                                         random_state=70)
            # best_model.fit(X_train, y_train)
            param_grid = {
                 'n_estimators': [10, 50, 100],
                 'max_depth': [10, 30, 50],
                 'max_features': ['sqrt', 'log2'],
                 'min_samples_split': [2, 5, 10],
                 'min_samples_leaf': [1, 2, 6],
                 'bootstrap': [True, False],
                 'min_impurity_decrease': [0.0, 0.01, 0.05],
            }
        best_model, best_params = grid_search(X_train, y_train, X_val, y_val, param_grid, args.classifier)
        print("Best Parameters:", best_params)

        with open(f"{args.classifier}_{args.task}.pkl", 'wb') as file:
            pickle.dump(best_model, file)
    evaluate_model(best_model, X_test, y_test, args.task, args.classifier, label_encoders)


if __name__ == "__main__":
    main()
