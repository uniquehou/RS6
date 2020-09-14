import random
import operator
import pandas as pd
import numpy as np
from copy import deepcopy

file_path = "user_taggedbookmarks-timestamps.dat"
records = {}
train_data = dict()
test_data = dict()
user_tags = dict()
tag_items = dict()
user_items = dict()

def load_data(file_path):
    print("Start load data...")
    data = pd.read_csv(file_path, sep='\t')
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

    for uid in records.keys():
        for iid, tag in records[uid].items():
            user_tags.setdefault(uid, {})
            for tid in tag:
                addValueToMat(user_tags, uid, tid, 1)
                addValueToMat(user_items, uid, iid, 1)
                addValueToMat(tag_items, tid, iid, 1)
    print(f"user_tags size: {len(user_tags)}, user_items size: {len(user_items)}, tag_items size: {len(tag_items)}")


def precision_recall(n):
    hit, h_precision, h_recall = 0, 0, 0
    for uid in test_data.keys():
        rank = recommend(uid, n)
        for iid, _ in rank:
            if iid in user_items[uid].keys():
                hit += 1
        h_precision += len(user_items[uid])
        h_recall += len(rank)
    return hit/h_precision, hit/h_recall

def recommend(uid, n):
    recommend_item = {}
    for tid, nt in user_tags[uid].items():
        for iid, ni in tag_items[tid].items():
            if iid not in user_items[uid].keys():
                if iid not in recommend_item.keys():
                    recommend_item[iid] = nt * ni
                else:
                    recommend_item[iid] += nt * ni
    return sorted(recommend_item.items(), key=operator.itemgetter(1), reverse=True)[:n]


def ttestrecommend():
    print("Evalutation of recommend result")
    print(f"{'n':>4s} {'precision':>10s} {'recall':>10s}")
    for n in [3, 5, 10, 30, 50, 100][:2]:
        precision, recall = precision_recall(n)
        print(f"{n:4d} {precision:9.4f}% {recall:9.4f}%")

if __name__ == '__main__':
    load_data(file_path)
    split_train_test()
    initState()
    ttestrecommend()