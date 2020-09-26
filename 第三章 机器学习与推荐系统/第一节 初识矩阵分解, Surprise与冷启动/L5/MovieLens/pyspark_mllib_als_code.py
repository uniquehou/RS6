from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS

# data format
# 1,31,2.5,1260759144
# 1,1029,3.0,1260759179
# 1,1061,3.0,1260759182
# 1,1129,2.0,1260759185
sc = SparkContext('local', 'MovieRec')
rawUserData = temp = sc.textFile('ratings_small_without_header.csv')
print(rawUserData.count(), rawUserData.first())

rawRatings = rawUserData.map(lambda line: line.split(',')[:3])
print(rawRatings.take(5))
training_RDD = rawRatings.map(lambda x: (x[0], x[1], x[2]))

rank = 3    # 隐特征数量
model = ALS.train(training_RDD, rank, seed=5, iterations=10, lambda_=0.2)
# 针对user_id=100的用户进行top-N推荐
print(model.recommendProducts(100, 5))