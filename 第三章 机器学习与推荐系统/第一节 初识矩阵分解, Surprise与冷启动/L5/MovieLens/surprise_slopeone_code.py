from surprise import Dataset
from surprise import Reader, SlopeOne
import pandas as pd

# data format
# movieId,title,genres
# 1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
# 2,Jumanji (1995),Adventure|Children|Fantasy
# 3,Grumpier Old Men (1995),Comedy|Romance
def read_item_names():
    data = pd.read_csv('movies.csv')
    rid_to_name, name_to_rid = {}, {}
    for i in range(len(data['movieId'])):
        rid_to_name[data['movieId'][i]] = data['title'][i]
        name_to_rid[data['title'][i]] = data['movieId'][i]
    return rid_to_name, name_to_rid

reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('./ratings.csv', reader=reader)
train_set = data.build_full_trainset()

algo = SlopeOne()
algo.fit(train_set)
uid, iid = "196", "302"
pred = algo.predict(uid, iid, r_ui=4, verbose=True)

# result
# user: 196        item: 302        r_ui = 4.00   est = 4.32   {'was_impossible': False}
# 4.322991780575918