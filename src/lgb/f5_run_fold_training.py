#!/usr/bin/env python
# coding: utf-8

##################################################################################
# LGB 模型训练
# 所使用的特征:
# 1. TF-IDF
# 2. COUNTVEC
# 3. LR model 概率
# 输出:
# AGE 概率
# GENDER 概率并没有使用, 如需要可取消掉对应的注释

# 训练时长:  ~ 5 days
##################################################################################

import warnings
warnings.filterwarnings("ignore")

import pandas as pd 
import numpy as np

import sys
import os
import json
import gc
from tqdm import tqdm

from sklearn.model_selection import train_test_split 
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler as std
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

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


# 减少内存函数
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


# ### 模型初始化

lgb_model = lgb.LGBMClassifier(
        boosting_type="gbdt", num_leaves=15, reg_alpha=0, reg_lambda=0.,
    max_depth=-1, n_estimators=1500, objective='multiclass',metric= 'multi_error',
    subsample=0.95, colsample_bytree=0.95, subsample_freq=1,
    learning_rate=0.1, random_state=2017
    )

lgb_model_binary = lgb.LGBMClassifier(
        boosting_type="gbdt", num_leaves=150, reg_alpha=0, reg_lambda=0.,
    max_depth=-1, n_estimators=500, objective='binary',metric= 'error',
    subsample=0.95, colsample_bytree=0.95, subsample_freq=1,
    learning_rate=0.025, random_state=2017
    )


# ### 读取数据

path = '../../raw_data/'
train_user1 = pd.read_csv(path+'train_preliminary/user.csv')
train_user2 =  pd.read_csv(path+'train_semi_final/user.csv')
train_user = train_user1.append(train_user2).reset_index(drop=True)
train_user = train_user.drop_duplicates(['user_id'])
train_user = train_user.sort_values(by=['user_id'])
test_click = pd.read_csv(path+'test/click_log.csv')


train_uid = train_user[['user_id','age','gender']]
test_uid = pd.DataFrame(list(set(test_click['user_id'])))
test_uid.columns=['user_id']

# ###  target_encoding 特征

target_encoding = pd.read_pickle('../../data/lgb/te_feats/df_user_target_encoding.pickle')


target_feats = [i for i in target_encoding.columns if i not in ['user_id']]
train_uid = train_uid.merge(target_encoding,on='user_id',how='left')
test_uid = test_uid.merge(target_encoding,on='user_id',how='left')


# ### tfidf_stacking特征

stackingfeats = pd.read_csv('../../data/lgb/stacking_feats/tfidf_classfiy_package.csv')
train_uid = train_uid.merge(stackingfeats, on='user_id', how='left')
test_uid = test_uid.merge(stackingfeats, on='user_id', how='left')

stackingfeats_vec = pd.read_csv('../../data/lgb/stacking_feats/tfidf_classfiy_age_package.csv')
train_uid = train_uid.merge(stackingfeats_vec,on='user_id',how='left')
test_uid = test_uid.merge(stackingfeats_vec,on='user_id',how='left')


# ### countvec stacking 特征 

normal_feats = [i for i in train_uid.columns if i not in ['user_id', 'age', 'gender']]


train_x = train_uid[normal_feats]
test_x = test_uid[normal_feats]


train_tfidf= sparse.load_npz('../../data/lgb/tf_idf_feats/traintext_tfidf3.npz')
test_tfidf= sparse.load_npz('../../data/lgb/tf_idf_feats/testtext_tfidf3.npz')

train_vec= sparse.load_npz('../../data/lgb/countvec_feats/traintext_countvec2.npz')
test_vec= sparse.load_npz('../../data/lgb/countvec_feats/testtext_countvec2.npz')

train_data = sparse.hstack((train_x, train_tfidf), 'csr')
test_data = sparse.hstack((test_x, test_tfidf), 'csr')

train_data = sparse.hstack((train_data, train_vec), 'csr')
test_data = sparse.hstack((test_data, test_vec), 'csr')


y = train_uid[['age']]
y2 = train_uid[['gender']]



model = lgb.LGBMClassifier(
        boosting_type="gbdt", num_leaves=176, reg_alpha=0.1, reg_lambda=0.1,
    max_depth=-1, n_estimators=55, objective='multiclass',metric= 'multi_error',
    subsample=0.95, colsample_bytree=0.95, subsample_freq=1,
    learning_rate=0.1, random_state=2017
)


train_uid['agepre'] = 0
train_uid['genpre'] = 0

testrs = test_uid[['user_id']]
testrs['age'] = 0
testrs['gender'] = 0


# #### 二分类模型定义

#fold = 0
#n_splits = 5
#testrs['genpre'] = 0
#train_uid['genpre'] = 0
#kfold = KFold(n_splits=5, shuffle=True, random_state=42)
#def run_model_gender():
#    for train_idx, val_idx in kfold.split(train_data):
#        train_x = train_data[train_idx]
#        train_y = y2.loc[train_idx]
#        test_xt = train_data[val_idx]
#        test_yt = y2.loc[val_idx]
#
#        lgb_model_binary.fit(train_x, train_y, eval_set=[(train_x,train_y),(test_xt, test_yt)], early_stopping_rounds=100,
#                          eval_metric='error',
#                          verbose=5)
#
#        train_uid.loc[val_idx, 'genpre'] = lgb_model_binary.predict_proba(test_xt)[:,1]
#        testrs['genpre']+= lgb_model_binary.predict_proba(test_data)[:,1]/n_splits
#
## #### 运行二分类模型
#
#run_model_gender()


# #### 多分类模型定义

for i in range(1,11):
    train_uid['age_prob_'+str(i)]=0
    testrs['age_prob_'+str(i)]=0

foldi = 0
n_splits = 5
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
seed = 2020
folds = StratifiedKFold(n_splits=5, random_state=seed, shuffle=False)
def run_model_age():
    for train_idx, val_idx in kfold.split(train_data):
        train_x = train_data[train_idx]
        train_y = y.loc[train_idx]
        test_x = train_data[val_idx]
        test_y = y.loc[val_idx]
        
        print('modelrun_fold_{}'.format(fold))
        model.fit(train_x, train_y, eval_set=[(train_x,train_y),(test_x, test_y)], early_stopping_rounds=100,
                          eval_metric='multi_error',
                          verbose=5)
        train_uid.loc[val_idx, ['age_prob_'+str(i) for i in range(1,11)]] = model.predict_proba(test_x)
        testrs[['age_prob_'+str(i) for i in range(1,11)]] += model.predict_proba(test_data)/5


# #### 运行多分类模型

run_model_age()


# 保存 AGE 概率结果
train_uid[['user_id']+['age_prob_'+str(i) for i in range(1,11)]+['genpre']].to_csv('../../probs/oof_age_lgb.csv', index=False)
testrs[['user_id']+['age_prob_'+str(i) for i in range(1,11)]+['genpre']].to_csv('../../probs/sub_age_lgb.csv', index=False)

