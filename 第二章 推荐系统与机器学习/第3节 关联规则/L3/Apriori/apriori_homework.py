import pandas as pd
import numpy as np
from efficient_apriori import apriori


data = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transcation = np.array([data.loc[i].dropna().unique() for i in range(data.shape[0])])
items, rules = apriori(transcation, min_support=0.1, min_confidence=0.5)
print("频繁项集：", items)
print("关联规则：", rules)

