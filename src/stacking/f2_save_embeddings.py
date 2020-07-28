#!/usr/bin/env python
# coding: utf-8

####################################################################
# 生成用于 stacking 用的序列特征
# 目的主要是为了与参与 stacking 的模型有区别
# 且限制过拟合
####################################################################

import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('max_rows', 1000)

import pickle
import gc
import logging

from tqdm.autonotebook import *

import gensim
from gensim.models import FastText, Word2Vec

from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences


window = 100
max_len = 100
min_count = 1
iter_ = 20
emb_dim_cid = 64
emb_dim_advid = 32


def set_tokenizer(docs, split_char=' '):
    tokenizer = Tokenizer(lower=False, char_level=False, split=split_char)
    tokenizer.fit_on_texts(docs)
    X = tokenizer.texts_to_sequences(docs)
    maxlen = max_len
    X = pad_sequences(X, maxlen=maxlen, value=0)
    word_index = tokenizer.word_index
    return X, word_index


def trian_save_word2vec(sentences, emb_dim, save_name='w2v.txt', split_char=' '):
    input_docs = []
    for i in sentences:
        input_docs.append([ii for ii in i])
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO
    )
    w2v = Word2Vec(input_docs, 
                   size=emb_dim, 
                   sg=1,
                   window=window, 
                   seed=2020, 
                   workers=18, 
                   min_count=min_count, 
                   iter=iter_)
    w2v.wv.save_word2vec_format(save_name)
    return w2v


def get_embedding_matrix(word_index, embed_size=128, Emed_path="w2v_300.txt"):
    embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(
        Emed_path, binary=False)
    nb_words = len(word_index)+1
    embedding_matrix = np.zeros((nb_words, embed_size))
    count = 0
    for word, i in tqdm(word_index.items()):
        if i >= nb_words:
            continue
        try:
            embedding_vector = embeddings_index[word]
        except:
            embedding_vector = np.zeros(embed_size)
            count += 1
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector    
    print("null cnt", count)
    return embedding_matrix



# creative_od
df = pd.read_pickle('data/df_creative_sequence.pickle')
cid_list = list(df['cids'])

for i in tqdm(range(0, len(cid_list))):
    cid_list[i] =[str(ii) for ii in cid_list[i]]
    
x_cid, index_cid = set_tokenizer(cid_list, split_char=',')
trian_save_word2vec(cid_list, 
                    emb_dim_cid, 
                    save_name=f'../../w2v_models/cid_w2v_{emb_dim_cid}_win{window}_iter{iter_}_mincount{min_count}.txt', 
                    split_char=',')

del df
gc.collect()


# advertiser_id
df = pd.read_pickle('../../data/keras/df_advertiser_sequence.pickle')
advid_list = list(df['advids'])

for i in tqdm(range(0, len(advid_list))):
    advid_list[i] =[str(ii) for ii in advid_list[i]]
    
x_advid, index_advid = set_tokenizer(advid_list, split_char=',')
trian_save_word2vec(advid_list, 
                    emb_dim_advid, 
                    save_name=f'../../w2v_models/advid_w2v_{emb_dim_advid}_win{window}_iter{iter_}_mincount{min_count}.txt', 
                    split_char=',')

del df
gc.collect()

