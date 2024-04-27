import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('UNSW-NB15-BALANCED-TRAIN.csv', low_memory=False, nrows=100000)

label_encoders = {}
for column in df.columns:
    if df[column].dtype == type(object):
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

print(df.head())
X = df.drop(['Label'], axis=1)
y = df['Label']

X, y = make_classification(n_samples=10000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=3)
model = DecisionTreeClassifier()  # Instantiate a model
pipeline = Pipeline(steps=[('s', rfe), ('m', model)])

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

print('Cross-validation Accuracy: %.2f (%.2f)' % (np.mean(n_scores), np.std(n_scores)))

