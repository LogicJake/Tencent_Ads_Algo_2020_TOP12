#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
from sklearn.model_selection import GroupKFold
import gc
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import math
import pickle

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

seed = 2020

# 读取数据集
df_train_ad = pd.read_csv('../../raw_data/train_preliminary/ad.csv')
df_train_log = pd.read_csv('../../raw_data/train_preliminary/click_log.csv')
df_train_user = pd.read_csv('../../raw_data/train_preliminary/user.csv')

df_test_ad = pd.read_csv('../../raw_data/test/ad.csv')
df_test_log = pd.read_csv('../../raw_data/test/click_log.csv')

df_train_semi_final_ad = pd.read_csv('../../raw_data/train_semi_final/ad.csv')
df_train_semi_final_log = pd.read_csv('../../raw_data/train_semi_final/click_log.csv')
df_train_semi_final_user = pd.read_csv('../../raw_data/train_semi_final/user.csv')

df_train_user = df_train_user.append(df_train_semi_final_user)
df_train_log = df_train_log.append(df_train_semi_final_log)
df_train_ad = df_train_ad.append(df_train_semi_final_ad)


# 提取所有用户
df_test_user = df_test_log[['user_id']]
df_test_user.drop_duplicates(inplace=True)
df_feature = pd.concat([df_train_user, df_test_user], sort=False)

# 日志数据
df_ad = pd.concat([df_train_ad, df_test_ad], sort=False)
df_ad.drop_duplicates(inplace=True)

df_log = pd.concat([df_train_log, df_test_log], sort=False)
df_log.sort_values(['user_id', 'time'], inplace=True)

df_log = df_log.merge(df_ad, how='left', on='creative_id')

df_feature.to_pickle('../../data/torch/feature.pkl')
df_log.to_pickle('../../data/torch/log.pkl')
