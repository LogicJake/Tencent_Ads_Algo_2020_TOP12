#!/usr/bin/env python
# coding: utf-8

#####################################################################
# 生成目标编码 Target Encoding 特征
#####################################################################

import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np

pd.set_option('max_columns', 500)
pd.set_option('max_rows', 500)
pd.set_option('display.float_format',lambda x : '%.4f' % x)

from tqdm import tqdm
tqdm.pandas()

import os
import gc
from sklearn.model_selection import KFold


user_train_1 = pd.read_csv('../../raw_data/train_preliminary/user.csv')
user_train_2 = pd.read_csv('../../raw_data/train_semi_final/user.csv')
user_train = pd.concat([user_train_1, user_train_2])

click_train_1 = pd.read_csv('../../raw_data/train_preliminary/click_log.csv')
click_train_2 = pd.read_csv('../../raw_data/train_semi_final/click_log.csv')
click_train = pd.concat([click_train_1, click_train_2])
click_test = pd.read_csv('../../raw_data/test/click_log.csv')

ad_train_1 = pd.read_csv('../../raw_data/train_preliminary/ad.csv')
ad_train_2 = pd.read_csv('../../raw_data/train_semi_final/ad.csv')
ad_train = pd.concat([ad_train_1, ad_train_2])
ad_test = pd.read_csv('../../raw_data/test/ad.csv')

del user_train_1, user_train_2
del click_train_1, click_train_2
del ad_train_1, ad_train_2
gc.collect()


df_train = pd.merge(click_train, ad_train, on='creative_id')
df_test = pd.merge(click_test, ad_test, on='creative_id')

df_train = df_train.drop_duplicates()
df_test = df_test.drop_duplicates()

del click_train, click_test, ad_train, ad_test
gc.collect()


df_train = pd.merge(df_train, user_train, on='user_id')

df = pd.concat([df_train, df_test])

del df_train, df_test
gc.collect()


df = df.sort_values(by=['user_id', 'time', 'click_times'], ascending=[True, True, True]).reset_index(drop=True)

# 目标编码

from sklearn.model_selection import KFold

def stat(df, df_merge, group_by, agg):
    group = df.groupby(group_by).agg(agg)

    columns = []
    for on, methods in agg.items():
        for method in methods:
            columns.append('{}_{}_{}'.format('_'.join(group_by), on, method))
    group.columns = columns
    group.reset_index(inplace=True)
    df_merge = df_merge.merge(group, on=group_by, how='left')

    del (group)
    gc.collect()
    return df_merge
    

def statis_feat(df_know, df_unknow):
    df_unknow = stat(df_know, df_unknow, ['ad_id'], {'age': ['mean', 'std'], 'gender': ['mean', 'std']})
    df_unknow = stat(df_know, df_unknow, ['creative_id'], {'age': ['mean', 'std'], 'gender': ['mean', 'std']})
    df_unknow = stat(df_know, df_unknow, ['advertiser_id'], {'age': ['mean', 'std'], 'gender': ['mean', 'std']})
    df_unknow = stat(df_know, df_unknow, ['product_id'], {'age': ['mean', 'std'], 'gender': ['mean', 'std']})
    df_unknow = stat(df_know, df_unknow, ['industry'], {'age': ['mean', 'std'], 'gender': ['mean', 'std']})
    df_unknow = stat(df_know, df_unknow, ['product_category'], {'age': ['mean', 'std'], 'gender': ['mean', 'std']})

    return df_unknow
    

df_train = df[~df['age'].isnull()]
df_train = df.reset_index(drop=True)
df_test = df[df['age'].isnull()]

df_stas_feat = None
kf = KFold(n_splits=5, random_state=2020, shuffle=True)
for train_index, val_index in kf.split(df_train):
    df_fold_train = df_train.iloc[train_index]
    df_fold_val = df_train.iloc[val_index]

    df_fold_val = statis_feat(df_fold_train, df_fold_val)
    df_stas_feat = pd.concat([df_stas_feat, df_fold_val], axis=0)

    del(df_fold_train)
    del(df_fold_val)
    gc.collect()

    
df_test = statis_feat(df_train, df_test)
df = pd.concat([df_stas_feat, df_test], axis=0)

del(df_stas_feat)
del(df_train)
del(df_test)
gc.collect()


df_ = df[[col for col in df.columns if col not in ['time', 'creative_id', 'click_times',
                                                   'ad_id', 'product_id', 'product_category',
                                                   'advertiser_id', 'industry', 'age', 'gender']]]

gc.collect()

for col in tqdm(['ad_id', 'creative_id', 'advertiser_id',
                 'product_id', 'industry', 'product_category']):
    for method in ['mean', 'std']: 
        df_[f'{col}_age_{method}_mean'] = df_.groupby('user_id')[f'{col}_age_{method}'].transform('mean')
        df_[f'{col}_gender_{method}_mean'] = df_.groupby('user_id')[f'{col}_gender_{method}'].transform('mean')


cols = [col for col in df_.columns if col.endswith('_mean_mean')] + [col for col in df_.columns if col.endswith('_std_mean')]

df_ = df_[['user_id', 'ad_id_age_mean_mean', 'ad_id_gender_mean_mean', 'ad_id_age_std_mean',
       'ad_id_gender_std_mean', 'creative_id_age_mean_mean',
       'creative_id_gender_mean_mean', 'creative_id_age_std_mean',
       'creative_id_gender_std_mean', 'advertiser_id_age_mean_mean',
       'advertiser_id_gender_mean_mean', 'advertiser_id_age_std_mean',
       'advertiser_id_gender_std_mean', 'product_id_age_mean_mean',
       'product_id_gender_mean_mean', 'product_id_age_std_mean',
       'product_id_gender_std_mean', 'industry_age_mean_mean',
       'industry_gender_mean_mean', 'industry_age_std_mean',
       'industry_gender_std_mean', 'product_category_age_mean_mean',
       'product_category_gender_mean_mean', 'product_category_age_std_mean',
       'product_category_gender_std_mean']].drop_duplicates(subset=['user_id'])

os.system('mkdir -pv ../../data/lgb/te_feats')

df_.to_pickle('../../data/lgb/te_feats/df_user_target_encoding.pickle')

