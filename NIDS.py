import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('UNSW-NB15-BALANCED-TRAIN.csv', low_memory=False, nrows=10000)

categorical_cols = df.select_dtypes(include=['object']).columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

df[categorical_cols] = df[categorical_cols].fillna('Missing')
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

X = df.drop(['Label'], axis=1)
y = df['Label']

print(df.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = RandomForestClassifier()
rfe = RFE(estimator=model, n_features_to_select=6)
rfe.fit(X_train, y_train)

features_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Selected': rfe.support_,
    'Ranking': rfe.ranking_
})
print("\nFeature Selection and Ranking:")
print(features_df)

X_train_rfe = rfe.transform(X_train)
X_val_rfe = rfe.transform(X_test)

model.fit(X_train_rfe, y_train)

y_val_pred = model.predict(X_val_rfe)
val_accuracy = metrics.accuracy_score(y_test, y_val_pred)

print("\nValidation Accuracy:", val_accuracy)
print("Selected Features:", X.columns[rfe.support_].tolist())
report = classification_report(y_test, y_val_pred, target_names=['0', '1'])


print(f"Classifier: {model.__class__.__name__}\n")
print(report)

importances = model.feature_importances_
feature_importances = pd.DataFrame({
    'Feature': X.columns[rfe.support_],
    'Importance': importances
})
feature_importances = feature_importances.sort_values('Importance', ascending=False)

# plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances')
plt.show()