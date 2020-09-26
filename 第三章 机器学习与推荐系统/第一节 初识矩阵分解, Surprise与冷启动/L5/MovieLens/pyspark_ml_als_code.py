from pyspark.ml.recommendation import ALS
from pyspark import SparkContext
from pyspark.sql import SQLContext
import pandas as pd

sc = SparkContext()
sql_sc = SQLContext

pd_df_ratings = pd.read_csv('ratings_small.csv')
pyspark_df_ratings = sql_sc.createDataFrame(pd_df_ratings)
pyspark_df_ratings = pyspark_df_ratings.drop('Timestamp')

# data format
# userId,movieId,rating,timestamp
# 1,31,2.5,1260759144
# 1,1029,3.0,1260759179
# 1,1061,3.0,1260759182
als = ALS(rank=3, maxIter=10, regParam=0.1,
          userCol='userId', itemCol='movieId', ratingCol='rating')
model = als.fit(pyspark_df_ratings)
recommendations = model.recommendForAllUsers(5)
print(recommendations.where(recommendations.userId==100).collect())
