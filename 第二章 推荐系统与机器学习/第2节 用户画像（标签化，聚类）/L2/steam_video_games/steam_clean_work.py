import numpy as np
import pandas as pd
import random

path = './steam-200k.csv'
df = pd.read_csv(path, header=None,
                 names=['UserID', 'Game', 'Action', 'Hours', 'Not Needed'])
df['Hours_Played'] = df['Hours'].astype('float32')
df.loc[(df['Action']=='purchase') & (df['Hours']==1.0), 'Hours_Played'] = 0

df.UserID = df.UserID.astype(np.int)
df = df.sort_values(['UserID', 'Game', 'Hours_Played'], ascending=True)

clean_df = df.drop_duplicates(['UserID', 'Game'], keep='last')
clean_df = clean_df.drop(['Action', 'Hours', 'Not Needed'], axis=1)

n_users = len(clean_df.UserID.unique())
n_games = len(clean_df.Game.unique())

sparsity = clean_df.shape[0] / float(n_users * n_games)
print(f"用户行为矩阵的稀疏性（填充比例）为{sparsity:.2%}")