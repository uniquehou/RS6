import random
import operator
import pandas as pd
import numpy as np
from copy import deepcopy

# 初始化时，总的数据结构时导入训练集的数据
# 将测试集的数据另存一份，在测试推荐的时候使用
file_path = "user_taggedbookmarks-timestamps.dat"
# file_path = "user_tagged_10000.csv"
records = {}
train_data = dict()
test_data = dict()
user_tags = dict()
tag_items = dict()
user_items = dict()
test_user_tags, test_tag_items, test_user_items = dict(), dict(), dict() # 测试集推荐所使用的数据结构

def load_data(file_path):
    print("Start load data...")
    data = pd.read_csv(file_path, sep='\t')
    # data = pd.read_csv(file_path)
    for _, row in data.iterrows():
        uid, iid, tid = row['userID': 'tagID']
        records.setdefault(uid, {})
        records[uid].setdefault(iid, [])
        records[uid][iid].append(tid)
    print("Load complete, data(user) size: %d" % len(records))

def split_train_test(train_size=0.8, seed=20):
    random.seed(10)
    for uid in records.keys():
        if random.random()<=train_size:
            train_data[uid] = deepcopy( records[uid] )
        else:
            test_data[uid] = deepcopy( records[uid] )
    print(f"train size: {len(train_data)}, test size: {len(test_data)}")


def initState():
    def addValueToMat(mat:dict, index, item, value=1):
        if index not in mat:
            mat.setdefault(index, {})
            mat[index].setdefault(item, value)
        elif item not in mat[index]:
            mat[index].setdefault(item, value)
        else:
            mat[index][item] += value

    # 初始化时，总的数据结构时导入训练集的数据
    for uid in train_data.keys():
        for iid, tag in train_data[uid].items():
            for tid in tag:
                addValueToMat(user_tags, uid, tid, 1)
                addValueToMat(user_items, uid, iid, 1)
                addValueToMat(tag_items, tid, iid, 1)
    # 将测试集的数据另存一份，在测试推荐的时候使用
    for uid in test_data.keys():
        for iid, tag in test_data[uid].items():
            for tid in tag:
                addValueToMat(test_user_tags, uid, tid, 1)
                addValueToMat(test_user_items, uid, iid, 1)
    print(f"user_tags size: {len(user_tags)}, user_items size: {len(user_items)}, tag_items size: {len(tag_items)}")


def precision_recall(n):
    hit, h_precision, h_recall = 0, 0, 0
    # 测试时只对测试集中的用户进行测试
    for uid in test_data.keys():
        rank = recommend(uid, n)
        for iid, _ in rank:
            if iid in test_user_items[uid].keys():
                hit += 1
        h_precision += len(test_user_items[uid])
        h_recall += len(rank)
    return hit/h_precision, hit/h_recall

def recommend(uid, n):
    recommend_item = {}
    for tid, nt in test_user_tags[uid].items():
        if tag_items.get(tid):
            for iid, ni in tag_items[tid].items():
                # if iid not in tes[uid].keys():
                if iid not in recommend_item.keys():
                    recommend_item[iid] = nt * ni
                else:
                    recommend_item[iid] += nt * ni
    return sorted(recommend_item.items(), key=operator.itemgetter(1), reverse=True)[:n]


def ttestrecommend():
    print("Evalutation of recommend result")
    print(f"{'n':>4s} {'precision':>10s} {'recall':>10s}")
    for n in [3, 5, 10, 30, 50, 100]:
        precision, recall = precision_recall(n)
        print(f"{n:4d} {precision*100:9.4f}% {recall*100:9.4f}%")

if __name__ == '__main__':
    load_data(file_path)
    split_train_test(train_size=0.9)
    initState()
    ttestrecommend()


# Start load data...
# Load complete, data(user) size: 1867
# train size: 1689, test size: 178
# user_tags size: 1867, user_items size: 1689, tag_items size: 38512
# Evalutation of recommend result
#    n  precision     recall
#    3    0.2789%    5.1136%
#    5    0.4339%    4.7782%
#   10    0.6508%    3.5918%
#   30    1.2397%    2.2840%
#   50    1.7252%    1.9077%
#  100    2.7893%    1.5441%