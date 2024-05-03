import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
from tabulate import tabulate


## this only tries the first 100000 rows of the dataset
df = pd.read_csv('UNSW-NB15-BALANCED-TRAIN.csv', low_memory=False, nrows=100000)

categorical_cols = df.select_dtypes(include=['object']).columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

df[categorical_cols] = df[categorical_cols].fillna('Missing')
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
# removing attack_cat and label columns from the x
X = df.drop(['attack_cat', 'Label'], axis=1)
# for the part 1 and 2 we will use label and after we will try to use attack_cat
y = df['Label']

print(df.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# tested esimators: RandomForestClassifier, DecisionTreeClassifier, LinearRegression, SVC, GradientBoostingClassifier
# the most realistic results are obtained with LinearRegression
estimator = DecisionTreeClassifier()
rfe = RFE(estimator=estimator, n_features_to_select=8)
rfe.fit(X_train, y_train)

features_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Selected': rfe.support_,
    'Ranking': rfe.ranking_
})
print("\nFeature Selection and Ranking:")
print(features_df)

X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
model = model.fit(X_train_rfe, y_train)

y_pred = model.predict(X_test_rfe)
print("Accuracy: {:.2f}%\n".format(accuracy_score(y_test, y_pred) * 100))

# Printing the confusion matrix
print("Formatted Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, columns=["Normal", "Attack"], index=["Normal", "Attack"])
print(tabulate(cm_df, headers='keys', tablefmt='psql'))

# Printing classification report
print(classification_report(y_test, y_pred))
print("Number of nodes:", model.tree_.node_count)
print("Selected Features:", X.columns[rfe.support_].tolist())

dot_data = StringIO()
export_graphviz(model, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=X.columns[rfe.support_].tolist(), class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree.png')
Image(graph.create_png())

