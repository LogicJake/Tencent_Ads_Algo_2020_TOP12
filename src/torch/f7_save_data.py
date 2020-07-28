#!/usr/bin/env python
# coding: utf-8


##########################################################################
# 生成 transformer_v2 所需的数据文件
##########################################################################


import pandas as pd
import warnings
from sklearn.model_selection import GroupKFold
import gc
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import math
import pickle
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

warnings.filterwarnings('ignore')


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
df_train_ad.drop_duplicates(inplace=True)


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


# Function to reduce the memory usage
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in tqdm(df.columns):
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(
                        np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(
                        np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(
                        np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(
                        np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(
                        np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(
                        np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


df_log = reduce_mem_usage(df_log)

# 把 id 拼接成字符串
for col in tqdm(['creative_id', 'ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry', 'click_times', 'time']):
    df_log[col]  = df_log[col].astype(str)

    tmp = df_log.sort_values(['user_id', 'time']).groupby('user_id')[col].agg(list).reset_index()
    tmp[col] = tmp[col].map(lambda x: ' '.join(x))
    df_feature = df_feature.merge(tmp, how='left')
    del tmp
    gc.collect()



df_feature.to_pickle('../../data/torch/data.pkl')


