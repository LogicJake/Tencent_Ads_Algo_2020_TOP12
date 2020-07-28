#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupKFold
import gc
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import math
import pickle
from gensim.models import Word2Vec
import logging
import os
from gensim.models import KeyedVectors

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


seed = 2020

df_log = pd.read_pickle('../../data/torch/log.pkl')

def emb(df, f1, f2, emb_size):
    print(
        '====================================== {} {} ======================================'
        .format(f1, f2))
    tmp = df.groupby(f1, as_index=False)[f2].agg(
        {'{}_{}_list'.format(f1, f2): list})
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]

    words = []
    for i in range(len(sentences)):
        words += [x for x in sentences[i]]
        sentences[i] = [str(x) for x in sentences[i]]

    model = Word2Vec(sentences,
                     size=emb_size,
                     window=10,
                     min_count=1,
                     sg=1,
                     hs=1,
                     workers=30,
                     seed=seed)

    emb_matrix = []
    words = list(set(words))
    for w in tqdm(words):
        if str(w) in model:
            emb_matrix.append(model[str(w)])
        else:
            emb_matrix.append([0] * emb_size)

    df_emb = pd.DataFrame(emb_matrix)
    df_emb.columns = [
        '{}_{}_w2v_{}'.format(f1, f2, i) for i in range(emb_size)
    ]
    df_emb[f2] = words

    return df_emb


for f1, f2, dim in [['user_id', 'industry', 128], ['user_id', 'ad_id', 128], ['user_id', 'product_id', 128]]:
    df_emb = emb(df_log, f1, f2, dim)

    df_emb.to_pickle('../../w2v_models/w2v_{}_{}.pkl'.format(f2, dim))

    del df_emb
    gc.collect()
