import pandas as pd
import time

data = pd.read_csv('BreadBasket_DMS.csv')
data['Item'] = data['Item'].str.lower()
data = data.drop(data[data.Item == 'none'].index)

def rule1():
    from efficient_apriori import apriori
    start = time.time()
    order_series = data.set_index('Transaction')['Item']
    transactions = []
    temp_index = 0
    for i, v in order_series.items():
        if i!=temp_index:
            temp_set = set()
            temp_index = i
            temp_set.add(v)
            transactions.append(temp_set)
        else:
            temp_set.add(v)

    itemsets, rules = apriori(transactions, min_support=0.02, min_confidence=0.5)
    print("频繁项集：", itemsets)
    print("关联规则：", rules)
    end = time.time()
    print("用时：", end-start)

def rule2():
    def encode_units(x):
        return int(x>=1)
    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import association_rules
    pd.options.display.max_columns = 100
    start = time.time()
    hot_encoded_df = data.groupby(['Transaction', 'Item'])['Item']\
        .count().unstack().reset_index().fillna(0).set_index('Transaction')
    hot_encoded_df = hot_encoded_df.applymap(encode_units)
    frequent_itemsets = apriori(hot_encoded_df, min_support=0.02, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=0.5)
    print("频繁项集：", frequent_itemsets)
    print("关联规则：", rules[ (rules['lift']>=1) & (rules['confidence']>0.5)])
    end = time.time()
    print("用时：", end-start)


if __name__ == '__main__':
    rule1()
    print('-' * 100)
    rule2()