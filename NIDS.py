import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn import metrics
import matplotlib.pyplot as plt

df = pd.read_csv('UNSW-NB15-BALANCED-TRAIN.csv', low_memory=False, nrows=100000)
for column in df.select_dtypes(include=[object]):
    df[column] = df[column].factorize()[0]

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