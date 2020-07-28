#!/usr/bin/env python
# coding: utf-8

########################################################################
# 将用户的点击日志按 user_id, time, click_times 聚合排列生成点击序列
# 生成 TF-IDF 特征和 COUNTVEC 特征
########################################################################

import warnings
warnings.filterwarnings("ignore")

import pandas as pd 
import numpy as np

import os
import json
import gc
from tqdm import tqdm

from sklearn.model_selection import train_test_split 
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler as std
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score

import time
import datetime 
from datetime import datetime, timedelta

from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy import stats
from scipy import sparse
import scipy.spatial.distance as dist

from collections import Counter 
from statistics import mode 

import math
from itertools import product
import ast


# 降低数据内存使用的函数
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

path = '../../raw_data/'

train_log1 = pd.read_csv(path+'train_preliminary/click_log.csv')
train_log2 = pd.read_csv(path+'train_semi_final/click_log.csv')
train_log = pd.concat([train_log1, train_log2])
train_ad1 = pd.read_csv(path+'train_preliminary/ad.csv')
train_ad2 = pd.read_csv(path+'train_semi_final/ad.csv')
train_ad = pd.concat([train_ad1, train_ad2])
train_log = pd.merge(train_log, train_ad, on='creative_id')
train_log = train_log.drop_duplicates()

test_log = pd.read_csv(path+'test/click_log.csv')
test_ad = pd.read_csv(path+'test/ad.csv')
test_log = pd.merge(test_log, test_ad, on='creative_id')

df_log = pd.concat([train_log, test_log])
df_log = df_log.drop_duplicates()
data = df_log.sort_values(by=['user_id', 'time','click_times'], ascending=[True,True,True]).reset_index(drop=True)


data['user_items']='tim'+data['time'].astype(str)+','+'crea'+data['creative_id'].astype(str)+','+'ad'+data['ad_id'].astype(str)+','+'prodd'+data['product_id'].astype(str)+','+'proc'+data['product_category'].astype(str)+','+'adv'+data['advertiser_id'].astype(str)+','+'ind'+data['industry'].astype(str)+','


trian_user1 = pd.read_csv(path+'train_preliminary/user.csv')
trian_user2 =  pd.read_csv(path+'train_semi_final/user.csv')
trian_user = trian_user1.append(trian_user2).reset_index(drop=True)
trian_user = trian_user.drop_duplicates(['user_id'])
trian_user = trian_user.sort_values(by=['user_id'])


df_tmp = data.groupby('user_id')['user_items'].agg(list)
df_tmp = pd.DataFrame(df_tmp)
df_tmp['user_id'] = df_tmp.index
df_tmp = df_tmp.reset_index(drop=True)


train_uid = trian_user[['user_id','age','gender']]
test_uid = pd.DataFrame(list(set(test_log['user_id'])))
test_uid.columns=['user_id']

# #### 构建Tfidf 特征

os.system('mkdir -pv ../../data/lgb/tf_idf_feats')

df_tmp['text'] = df_tmp['user_items'].apply(lambda x: " ".join([str(i) for i in x]))
txt1 = data.groupby('user_id')['user_items'].apply(lambda x: " ".join(x)).reset_index()['user_items']
X = list(txt1.values)
tfv = TfidfVectorizer(min_df=35)
tfv.fit(X)

train_uid = train_uid.merge(df_tmp, on='user_id', how='left')
test_uid = test_uid.merge(df_tmp, on='user_id', how='left')

traintext_tfidf = tfv.transform(train_uid['text'].values)
sparse.save_npz('./tf_idf_feats/traintext_tfidf3.npz',traintext_tfidf)
testtext_tfidf = tfv.transform(test_uid['text'].values)
sparse.save_npz('./tf_idf_feats/testtext_tfidf3.npz', testtext_tfidf)

# #### 构建COUNTVEC特征

os.system('mkdir -pv ../../data/lgb/countvec_feats')

cv = CountVectorizer(min_df=30)
cv.fit(df_tmp['text'])

train_ta = cv.transform(train_uid['text'])
sparse.save_npz('../../data/lgb/countvec_feats/traintext_countvec2.npz',train_ta)
test_ta = cv.transform(test_uid['text'])
sparse.save_npz('../../data/lgb/countvec_feats/testtext_countvec2.npz',test_ta)

