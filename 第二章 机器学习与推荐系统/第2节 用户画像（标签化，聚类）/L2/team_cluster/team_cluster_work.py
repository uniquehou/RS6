from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np
import pandas as pd

data = pd.read_csv('team_cluster_data.csv', encoding='gbk')
train_x = data[['2019国际排名', '2018世界杯排名', '2015亚洲杯排名']]
kmeans = KMeans(n_clusters=3)
min_max_scaler = preprocessing.MinMaxScaler()
train_x = min_max_scaler.fit_transform(train_x)
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)
result = pd.concat((data, pd.DataFrame(predict_y)), axis=1)
result.rename({0:u'聚类结果'}, axis=1, inplace=True)
print(result)
result.to_csv("team_cluster_result.csv")