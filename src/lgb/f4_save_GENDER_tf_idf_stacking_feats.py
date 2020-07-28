#!/usr/bin/env python
# coding: utf-8

##########################################################################
# 用 TF-IDF 特征训练 LR model 生成概率当做后面 LGB 的 stacking 特征
##########################################################################

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

train_user1 = pd.read_csv(path+'train_preliminary/user.csv')
train_user2 = pd.read_csv(path+'train_semi_final/user.csv')
train_user = train_user1.append(train_user2).reset_index(drop=True)
train_user = train_user.drop_duplicates(['user_id'])
train_user = train_user.sort_values(by=['user_id'])
test_click = pd.read_csv(path+'test/click_log.csv')

train_uid= train_user[['user_id','age','gender']]
test_uid= pd.DataFrame(list(set(test_click['user_id'])))
test_uid.columns=['user_id']


train_x = pd.DataFrame()
test_x = pd.DataFrame()


train_tfidf= sparse.load_npz('./tf_idf_feats/traintext_tfidf3.npz')
test_tfidf= sparse.load_npz('./tf_idf_feats/testtext_tfidf3.npz')


train_data = sparse.hstack((train_x, train_tfidf), 'csr')
test_data = sparse.hstack((test_x, test_tfidf), 'csr')



y2 = train_uid[['gender']]
all_id = train_uid.append(test_uid).reset_index(drop=True)


print('开始进行一些前期处理')
train_feature = train_data
test_feature = test_data
# 五则交叉验证
n_folds = 5
print('处理完毕')
df_stack = pd.DataFrame()
df_stack['user_id']=all_id['user_id']
seed = 1017
folds = StratifiedKFold(n_splits=5, random_state=seed, shuffle=False)

for label in ['gender']:
    score = y2[label]-1
    
    ########################### lr(LogisticRegression) ################################
    print('lr stacking')
    stack_train = np.zeros((len(train_uid), 1))
    stack_test = np.zeros((len(test_uid), 1))
    score_va = 0
    for i, (tr, va) in enumerate(folds.split(train_feature,score)):
        print('stack:%d/%d' % ((i + 1), n_folds))
        clf = LogisticRegression(random_state=1017, C=8)
        clf.fit(train_feature[tr], score[tr])
        score_va = clf.predict_proba(train_feature[va])[:,1]
        
        score_te = clf.predict_proba(test_feature)[:,1]
        print('得分' + str(mean_squared_error(score[va], clf.predict(train_feature[va]))))
        stack_train[va,0] = score_va
        stack_test[:,0]+= score_te
    stack_test /= n_folds
    
    stack = np.vstack([stack_train, stack_test])
    
    
    df_stack['pack_tfidf_vec_lr_classfiy_{}'.format(label)] = stack[:, 0]


########################### SGD(随机梯度下降) ################################
print('sgd stacking')
stack_train = np.zeros((len(train_uid), 1))
stack_test = np.zeros((len(test_uid), 1))
score_va = 0

for i, (tr, va) in enumerate(folds.split(train_feature,score)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    sgd = SGDClassifier(random_state=1017, loss='log')
    sgd.fit(train_feature[tr], score[tr])
    score_va = sgd.predict_proba(train_feature[va])[:,1]
    score_te = sgd.predict_proba(test_feature)[:,1]
    print('得分' + str(mean_squared_error(score[va], sgd.predict(train_feature[va]))))
    stack_train[va,0] = score_va
    stack_test[:,0]+= score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])

df_stack['pack_tfidf_vec_sgd_classfiy_{}'.format(label)] = stack[:, 0]

########################### pac(PassiveAggressiveClassifier) ################################
print('sgd stacking')
stack_train = np.zeros((len(train_uid), 1))
stack_test = np.zeros((len(test_uid), 1))
score_va = 0
for i, (tr, va) in enumerate(folds.split(train_feature,score)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    pac = PassiveAggressiveClassifier(random_state=1017)
    pac.fit(train_feature[tr], score[tr])
    score_va = pac._predict_proba_lr(train_feature[va])[:,1]
    score_te = pac._predict_proba_lr(test_feature)[:,1]
    print(score_va)
    print('得分' + str(mean_squared_error(score[va], pac.predict(train_feature[va]))))
    stack_train[va,0] += score_va
    stack_test[:,0] += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])

df_stack['pack_tfidf_vec_pac_classfiy_{}'.format(label)] = stack[:, 0]


########################### ridge(RidgeClassfiy) ################################
print('RidgeClassfiy stacking')
stack_train = np.zeros((len(train_uid), 1))
stack_test = np.zeros((len(test_uid), 1))
score_va = 0

for i, (tr, va) in enumerate(folds.split(train_feature,score)):
    ridge = RidgeClassifier(random_state=1017)
    ridge.fit(train_feature[tr], score[tr])
    score_va = ridge._predict_proba_lr(train_feature[va])[:,1]
    score_te = ridge._predict_proba_lr(test_feature)[:,1]
    print(score_va)
    print('得分' + str(mean_squared_error(score[va], ridge.predict(train_feature[va]))))
    stack_train[va,0] += score_va
    stack_test[:,0] += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])


df_stack['pack_tfidf_vec_ridge_classfiy_{}'.format(label)] = stack[:, 0]

########################### bnb(BernoulliNB) ################################
print('BernoulliNB stacking')
stack_train = np.zeros((len(train_uid), 1))
stack_test = np.zeros((len(test_uid), 1))
score_va = 0

for i, (tr, va) in enumerate(folds.split(train_feature,score)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    bnb = BernoulliNB()
    bnb.fit(train_feature[tr], score[tr])
    score_va = bnb.predict_proba(train_feature[va])[:,1]
    score_te = bnb.predict_proba(test_feature)[:,1]
    print(score_va)
    print('得分' + str(mean_squared_error(score[va], bnb.predict(train_feature[va]))))
    stack_train[va,0] += score_va
    stack_test[:,0] += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])


df_stack['pack_tfidf_vec_bnb_classfiy_{}'.format(label)] = stack[:, 0]

########################### mnb(MultinomialNB) ################################
print('MultinomialNB stacking')
stack_train = np.zeros((len(train_uid), 1))
stack_test = np.zeros((len(test_uid), 1))
score_va = 0

for i, (tr, va) in enumerate(folds.split(train_feature,score)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    mnb = MultinomialNB()
    mnb.fit(train_feature[tr], score[tr])
    score_va = mnb.predict_proba(train_feature[va])[:,1]
    score_te = mnb.predict_proba(test_feature)[:,1]
    print(score_va)
    print('得分' + str(mean_squared_error(score[va], mnb.predict(train_feature[va]))))
    stack_train[va,0] += score_va
    stack_test[:,0] += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])


df_stack['pack_tfidf_vec_mnb_classfiy_{}'.format(label)] = stack[:, 0]
    

############################ Linersvc(LinerSVC) ################################
print('LinerSVC stacking')
stack_train = np.zeros((len(train_uid), 1))
stack_test = np.zeros((len(test_uid), 1))
score_va = 0

for i, (tr, va) in enumerate(folds.split(train_feature,score)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    lsvc = LinearSVC(random_state=1017)
    lsvc.fit(train_feature[tr], score[tr])
    score_va = lsvc._predict_proba_lr(train_feature[va])[:,1]
    score_te = lsvc._predict_proba_lr(test_feature)[:,1]
    print(score_va)
    print('得分' + str(mean_squared_error(score[va], lsvc.predict(train_feature[va]))))
    stack_train[va,0] += score_va
    stack_test[:,0] += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])


df_stack['pack_tfidf_vec_lsvc_classfiy_{}'.format(label)] = stack[:, 0]
    

############################################# save ############################################### 
os.system('mkdir -pv ../../data/lgb/stacking_feats')
df_stack.to_csv('../../data/lgb/stacking_feats/tfidf_classfiy_package.csv', index=None, encoding='utf8')

