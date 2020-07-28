#!/usr/bin/env python
# coding: utf-8


#####################################################################################
# GENDER model 2: Keras transformer+LSTM 3 inputs 
# score: 
# 五折: 0.94910 (线下)
# 五折: 0.95014 (线上)
# 训练时长: ~ 2.5 days
#####################################################################################

import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('max_rows', 1000)

import sys
import time
import pickle
import gc
import logging

from tqdm.autonotebook import *

import gensim
from gensim.models import FastText, Word2Vec

import keras
from keras import layers
from keras import callbacks

from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

from keras_multi_head import MultiHead, MultiHeadAttention
from keras_self_attention import SeqSelfAttention
from keras_position_wise_feed_forward import FeedForward
from keras_layer_normalization import LayerNormalization



fold = sys.argv[1]

max_len = 100
emb_dim_cid = 128
emb_dim_aid = 128
emb_dim_advid = 128

batch_size = 512


def set_tokenizer(docs, split_char=' '):
    tokenizer = Tokenizer(lower=False, char_level=False, split=split_char)
    tokenizer.fit_on_texts(docs)
    X = tokenizer.texts_to_sequences(docs)
    maxlen = max_len
    X = pad_sequences(X, maxlen=maxlen, value=0)
    word_index = tokenizer.word_index
    return X, word_index


def get_embedding_matrix(word_index, embed_size=128, Emed_path="w2v_300.txt"):
    embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(
        Emed_path, binary=False)
    nb_words = len(word_index)+1
    embedding_matrix = np.zeros((nb_words, embed_size))
    count = 0
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        try:
            embedding_vector = embeddings_index[word]
        except:
            embedding_vector = np.zeros(embed_size)
            count += 1
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


print("loading sequence data and embedding")
start_time = time.time()


print("loading creative id")
df = pd.read_pickle('../../data/keras/df_creative_sequence.pickle')
cid_list = list(df['cids'])
for i in range(0, len(cid_list)):
    cid_list[i] =[str(ii) for ii in cid_list[i]]
    
x_cid, index_cid = set_tokenizer(cid_list, split_char=',')
emb_cid = get_embedding_matrix(index_cid,
                               embed_size=emb_dim_cid,
                               Emed_path='../../w2v_models/cid_w2v_128_win8_iter10_mincount3.txt')
del df, cid_list, index_cid
gc.collect()


print("loading advertiser id")
df = pd.read_pickle('../../data/keras/df_advertiser_sequence.pickle')
advid_list = list(df['advids'])
for i in range(0, len(advid_list)):
    advid_list[i] =[str(ii) for ii in advid_list[i]]

x_advid, index_advid = set_tokenizer(advid_list, split_char=',')
emb_advid = get_embedding_matrix(index_advid,
                                 embed_size=emb_dim_advid,
                                 Emed_path='../../w2v_models/advid_w2v_128_win8_iter10_mincount3.txt')
del df, advid_list, index_advid
gc.collect()


print("loading ad id")
df = pd.read_pickle('../../data/df_ad_sequence.pickle')
aid_list = list(df['aids'])
for i in range(0, len(aid_list)):
    aid_list[i] =[str(ii) for ii in aid_list[i]]

x_aid, index_aid = set_tokenizer(aid_list, split_char=',')
emb_aid = get_embedding_matrix(index_aid,
                               embed_size=emb_dim_aid,
                               Emed_path='../../w2v_models/adid_w2v_128_win8_iter10_mincount3.txt')
del df, aid_list, index_aid
gc.collect()


used_minutes = (time.time() - start_time) / 60
print(f"done, used {used_minutes} minutes")


print("loading labels")
start_time = time.time()

labels_1 = pd.read_csv('../../raw_data/train_preliminary/user.csv')
labels_2 = pd.read_csv('../../raw_data/train_semi_final/user.csv')
labels = pd.concat([labels_1, labels_2])
labels['age'] = labels['age'] - 1
labels['gender'] = labels['gender'] - 1

used_minutes = (time.time() - start_time) / 60
print(f"done, used {used_minutes} minutes")

print("split train, valid and test data")
start_time = time.time()

y = labels['gender'].astype(int)

if fold == "fold0":
    train_cid = x_cid[:2400000]
    valid_cid = x_cid[2400000:3000000]
    train_advid = x_advid[:2400000]
    valid_advid = x_advid[2400000:3000000]
    train_aid = x_aid[:2400000]
    valid_aid = x_aid[2400000:3000000]
    y_train = y[:2400000]
    y_valid = y[2400000:]
elif fold == "fold1":
    train_cid = np.concatenate((x_cid[:1800000], x_cid[2400000:3000000]), axis=0)
    valid_cid = x_cid[1800000:2400000]
    train_advid = np.concatenate((x_advid[:1800000], x_advid[2400000:3000000]), axis=0)
    valid_advid = x_advid[1800000:2400000]
    train_aid = np.concatenate((x_aid[:1800000], x_aid[2400000:3000000]), axis=0)
    valid_aid = x_aid[1800000:2400000]
    y_train = np.concatenate((y[:1800000], y[2400000:3000000]))
    y_valid = y[1800000:2400000]
elif fold == "fold2":
    train_cid = np.concatenate((x_cid[:1200000], x_cid[1800000:3000000]), axis=0)
    valid_cid = x_cid[1200000:1800000]
    train_advid = np.concatenate((x_advid[:1200000], x_advid[1800000:3000000]), axis=0)
    valid_advid = x_advid[1200000:1800000]
    train_aid = np.concatenate((x_aid[:1200000], x_aid[1800000:3000000]), axis=0)
    valid_aid = x_aid[1200000:1800000]
    y_train = np.concatenate((y[:1200000], y[1800000:3000000]))
    y_valid = y[1200000:1800000]
elif fold == "fold3":
    train_cid = np.concatenate((x_cid[:600000], x_cid[1200000:3000000]), axis=0)
    valid_cid = x_cid[600000:1200000]
    train_advid = np.concatenate((x_advid[:600000], x_advid[1200000:3000000]), axis=0)
    valid_advid = x_advid[600000:1200000]
    train_aid = np.concatenate((x_aid[:600000], x_aid[1200000:3000000]), axis=0)
    valid_aid = x_aid[600000:1200000]
    y_train = np.concatenate((y[:600000], y[1200000:3000000]))
    y_valid = y[600000:1200000]
elif fold == "fold4":
    train_cid = x_cid[600000:3000000]
    valid_cid = x_cid[:600000]
    train_advid = x_advid[600000:3000000]
    valid_advid = x_advid[:600000]
    train_aid = x_aid[600000:3000000]
    valid_aid = x_aid[:600000]
    y_train = y[600000:3000000]
    y_valid = y[:600000]
else:
    pass

test_cid = x_cid[3000000:]
test_advid = x_advid[3000000:]
test_aid = x_aid[3000000:]

del x_cid, x_advid, x_aid
del y
gc.collect()

print(train_cid.shape, valid_cid.shape, test_cid.shape)
print(train_advid.shape, valid_advid.shape, test_advid.shape)
print(train_aid.shape, valid_aid.shape, test_aid.shape)
print(y_train.shape, y_valid.shape)

used_minutes = (time.time() - start_time) / 60
print(f"done, used {used_minutes} minutes")


print("building model")

start_time = time.time()

def build_model(emb_cid, emb_advid, emb_aid):
    
    inp1 = layers.Input(shape=(max_len,))
    inp2 = layers.Input(shape=(max_len,))
    inp3 = layers.Input(shape=(max_len,))
    
    emb1 = layers.Embedding(
        input_dim=emb_cid.shape[0],
        output_dim=emb_cid.shape[1],
        input_length=max_len,
        weights=[emb_cid],
        trainable=False
    )(inp1)
    emb2 = layers.Embedding(
        input_dim=emb_advid.shape[0],
        output_dim=emb_advid.shape[1],
        input_length=max_len,
        weights=[emb_advid],
        trainable=False
    )(inp2)
    emb3 = layers.Embedding(
        input_dim=emb_aid.shape[0],
        output_dim=emb_aid.shape[1],
        input_length=max_len,
        weights=[emb_aid],
        trainable=False
    )(inp3)
    
    sdrop = layers.SpatialDropout1D(rate=0.2)
    
    emb1 = sdrop(emb1)
    emb2 = sdrop(emb2)
    emb3 = sdrop(emb3)
    
    content = layers.Concatenate()([emb1, emb2, emb3])
    
    mha1 = MultiHeadAttention(head_num=32)(content)
    mha1 = layers.Dropout(0.01)(mha1)
    mha1 = layers.Add()([content, mha1])
    mha1 = LayerNormalization()(mha1)
    mha1 = layers.Dropout(0.01)(mha1)
    mha1_ff = FeedForward(256)(mha1)
    mha1_out = layers.Add()([mha1, mha1_ff])
    mha1_out = LayerNormalization()(mha1_out)
    
    lstm = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(mha1_out)

    avg_pool = layers.GlobalAveragePooling1D()(lstm)
    max_pool = layers.GlobalMaxPool1D()(lstm)
    
    x = layers.Concatenate()([avg_pool, max_pool])
    
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    out = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=[inp1, inp2, inp3], outputs=out)
    model.compile(loss='binary_crossentropy', 
                  optimizer=keras.optimizers.Adam(2e-4), 
                  metrics=['accuracy'])
    
    return model


model = build_model(emb_cid, emb_advid, emb_aid)

used_minutes = (time.time() - start_time) / 60
print(f"done, used {used_minutes} minutes")



checkpoint = callbacks.ModelCheckpoint(f'../../models/gender_m2_{fold}.h5',
                                       monitor='val_accuracy',
                                       verbose=1,
                                       save_best_only=True,
                                       mode='max',
                                       save_weights_only=True)

reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                        factor=0.1,
                                        min_lr=2e-7,
                                        patience=2,
                                        verbose=1,
                                        mode='max',
                                        epsilon=1e-6)

early_stop = callbacks.EarlyStopping(monitor='val_accuracy',
                                     mode='max',
                                     patience=4)


hist = model.fit([train_cid, train_advid, train_aid],
                 y_train, 
                 batch_size=batch_size, 
                 epochs=100, 
                 validation_data=([valid_cid, valid_advid, valid_aid], y_valid),
                 callbacks=[checkpoint, reduce_lr, early_stop], 
                 verbose=1,
                 shuffle=True)



acc = max(hist.history['val_accuracy'])
print(acc)

print("predict start")
start_time = time.time()

model.load_weights(f'../../models/gender_m2_{fold}.h5')
preds = model.predict([test_cid, test_advid, test_aid],
                      batch_size=batch_size,
                      verbose=1)

np.save(f'../../probs/sub_gender_m2_keras_{fold}', preds)

used_minutes = (time.time() - start_time) / 60
print(f"done, used {used_minutes} minutes")



print("save oof start")
start_time = time.time()

valid_preds = model.predict([valid_cid, valid_advid, valid_aid],
                            batch_size=batch_size,
                            verbose=1)

np.save(f'../../probs/oof_gender_m2_keras_{fold}', valid_preds)

used_minutes = (time.time() - start_time) / 60
print(f"done, used {used_minutes} minutes")

