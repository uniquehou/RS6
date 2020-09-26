from surprise import Dataset
from surprise import Reader, BaselineOnly, accuracy
from surprise.model_selection import KFold

# data format
# userId,movieId,rating,timestamp
# 1,31,2.5,1260759144
# 1,1029,3.0,1260759179
# 1,1061,3.0,1260759182

reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('ratings.csv', reader=reader)
train_set = data.build_full_trainset()

# ASL 优化
asl_options = {
    'method': 'als',
    'n_epochs': 5,
    'reg_u': 12,    # 对用户的正则化参数
    'reg_i': 5,    # 对商品的正则化参数
}
sgd_option = {
    'method': 'sgd',
    'n_epochs': 5
}
algo = BaselineOnly(bsl_options=asl_options)
# K 折交叉验证
kf = KFold(n_splits=3)
for train_set, test_set in kf.split(data):
    algo.fit(train_set)
    predictions = algo.test(test_set)
    accuracy.rmse(predictions, verbose=True)

uid, iid = "196", "302"
pre = algo.predict(uid, iid, r_ui=4, verbose=True)

# result
# Estimating biases using als...
# RMSE: 0.8638
# Estimating biases using als...
# RMSE: 0.8640
# Estimating biases using als...
# RMSE: 0.8642
# user: 196        item: 302        r_ui = 4.00   est = 4.29   {'was_impossible': False}
# 4.288910833816449