import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# 填充nan值
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S', inplace=True)
# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]

# 显示特征之间的相关系数
plt.figure(figsize=(10, 10))
plt.title('Pearson Correlation between Features', y=1.05, size=15)
train_data_hot_encode = train_features.drop('Embarked', 1)\
    .join(train_features.Embarked.str.get_dummies())
train_data_hot_encode = train_data_hot_encode.drop('Sex', 1)\
    .join(train_data_hot_encode.Sex.str.get_dummies())
sns.heatmap(train_data_hot_encode.corr(), linewidths=0.1, vmax=1.0,
            fmt='.2f', square=True, linecolor='white', annot=True)
plt.show()

# 使用饼图来进行Survived取值的可视化
train_data['Survived'].value_counts().plot(kind='pie', label='Survived')
plt.show()
# 不同Pclass，幸存人数
sns.barplot(x='Pclass', y='Survived', data=train_data)
plt.show()
# 不同Embarked，幸存人数
sns.barplot(x='Embarked', y='Survived', data=train_data)
plt.show()

# 训练并显示特征向量的重要程度
def train(train_features, train_labels):
    clf = DecisionTreeClassifier()
    clf.fit(train_features, train_labels)
    coeffs = clf.feature_importances_
    df_co = pd.DataFrame(coeffs, columns=['importance_'])
    df_co.index = train_features.columns
    df_co.sort_values("importance_", ascending=True, inplace=True)
    df_co.importance_.plot(kind='barh')
    plt.title('Feature Importance')
    plt.show()
    return clf

clf = train(train_data_hot_encode, train_data['Survived'])

import pydotplus
from io import StringIO
from sklearn.tree import export_graphviz

def show_tree(clf):
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('titanic_tree.pdf')
show_tree(clf)