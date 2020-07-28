#!/usr/bin/env python
# coding: utf-8

#############################################################################
# 对各不同的 id 生成序列数据
#############################################################################

import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
pd.set_option('max_columns', 1000)
pd.set_option('max_rows', None)
pd.set_option('display.float_format',lambda x : '%.6f' % x)

from pandarallel import pandarallel
pandarallel.initialize()

from gensim.models import Word2Vec

from tqdm import tqdm
tqdm.pandas()

import gc


df = pd.read_pickle('../../data/keras/df_user_click_ad.pickle')


# click_times
df_clicktimes = df.groupby('user_id')['click_times'].agg(clks=list)
df_clicktimes = df_clicktimes.reset_index(drop=True)

df_clicktimes.to_pickle('../../data/keras/df_clicktimes_sequence.pickle')


# creative_id
df_creative = df.groupby('user_id')['creative_id'].agg(cids=list)
df_creative = df_creative.reset_index(drop=True)

df_creative.to_pickle('../../data/keras/df_creative_sequence.pickle')


# advertiser_id
df_advertiser = df.groupby('user_id')['advertiser_id'].agg(advids=list)
df_advertiser = df_advertiser.reset_index(drop=True)

df_advertiser.to_pickle('../../data/keras/df_advertiser_sequence.pickle')


# ad_id
df_ad = df.groupby('user_id')['ad_id'].agg(aids=list)
df_ad = df_ad.reset_index(drop=True)

df_ad.to_pickle('../../data/keras/df_ad_sequence.pickle')


# industry
df_industry = df.groupby('user_id')['industry'].agg(industry=list)
df_industry = df_industry.reset_index(drop=True)

df_industry.to_pickle('../../data/keras/df_industry_sequence.pickle')


# product
df_product = df.groupby('user_id')['product_id'].agg(pids=list)
df_product = df_product.reset_index(drop=True)

df_product.to_pickle('../../data/keras/df_product_sequence.pickle')

