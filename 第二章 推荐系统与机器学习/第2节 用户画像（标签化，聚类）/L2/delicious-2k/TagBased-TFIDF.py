import random
import operator
import pandas as pd
import numpy as np
from copy import deepcopy
from math import log

file_path = "user_taggedbookmarks-timestamps.dat"
# file_path = "user_tagged_10000.csv"
records = {}
train_data = dict()
test_data = dict()
user_tags = dict()
tag_items = dict()
user_items = dict()
tag_users = dict()  # 某标签在所有用户里出现的次数
test_user_tags, test_tag_items, test_user_items, test_tag_users = dict(), dict(), dict(), dict()

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

    for uid in train_data.keys():
        tag_users_flag = 0
        for iid, tag in train_data[uid].items():
            for tid in tag:
                addValueToMat(user_tags, uid, tid, 1)
                addValueToMat(user_items, uid, iid, 1)
                addValueToMat(tag_items, tid, iid, 1)
                if tid not in tag_users.keys():
                    tag_users.setdefault(tid, 1)
                    tag_users_flag = 1
                elif tag_users_flag==0:
                    tag_users[tid] += 1
                    tag_users_flag = 1
    for uid in test_data.keys():
        for iid, tag in test_data[uid].items():
            for tid in tag:
                addValueToMat(test_user_tags, uid, tid, 1)
                addValueToMat(test_user_items, uid, iid, 1)

    print(f"user_tags size: {len(user_tags)}, user_items size: {len(user_items)}, tag_items size: {len(tag_items)}")


def precision_recall(n):
    hit, h_precision, h_recall = 0, 0, 0
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
                if iid not in recommend_item.keys():
                    recommend_item[iid] = (nt/log(tag_users[tid]+1)) * ni
                else:
                    recommend_item[iid] += (nt/log(tag_users[tid]+1)) * ni
    return sorted(recommend_item.items(), key=operator.itemgetter(1), reverse=True)[:n]


def ttestrecommend():
    print("Evalutation of recommend result")
    print(f"{'n':>4s} {'precision':>10s} {'recall':>10s}")
    for n in [3, 5, 10, 30, 50, 100]:
        precision, recall = precision_recall(n)
        print(f"{n:4d} {precision:9.4f}% {recall:9.4f}%")

if __name__ == '__main__':
    load_data(file_path)
    split_train_test(0.9)
    initState()
    ttestrecommend()


# Evalutation of recommend result
#    n  precision     recall
#    3    0.0029%    0.0545%
#    5    0.0048%    0.0537%
#   10    0.0076%    0.0431%
#   30    0.0151%    0.0285%
#   50    0.0203%    0.0231%
#  100    0.0302%    0.0172%