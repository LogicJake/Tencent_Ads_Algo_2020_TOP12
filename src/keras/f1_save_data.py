#!/usr/bin/env python
# coding: utf-8



import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('max_rows', 1000)
pd.set_option('float_format', lambda x: '%.6f' % x)

import pickle
import gc
import logging

from tqdm.autonotebook import *

import gensim
from gensim.models import FastText, Word2Vec


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


df.to_pickle('../../data/keras/df_user_click_ad.pickle')

