import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction import DictVectorizer

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)

train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S', inplace=True)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels =  train_data['Survived']
test_features = test_data[features]

dvec = DictVectorizer(sparse=False)
train_features = dvec.fit_transform(train_features.to_dict(orient='record'))
test_features = dvec.transform(test_features.to_dict(orient='record'))
print(dvec.feature_names_)

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(train_features, train_labels)
pred_labels = clf.predict(test_features)

acc_decision_tree = round(clf.score(train_features, train_labels), 6)
print('score accuracy: %.4lf' % acc_decision_tree)

print('cross_val_score accuracy: %.4lf' %
      np.mean(cross_val_score(clf, train_features, train_labels, cv=10)))