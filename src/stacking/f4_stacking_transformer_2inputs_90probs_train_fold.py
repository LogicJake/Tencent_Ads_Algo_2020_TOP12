#!/usr/bin/env python
# coding: utf-8

#####################################################################################
# AGE Stacking: Keras transformer 2 inputs with 90 prob feats 
# score: 
# 五折: 0.52018 (线下)
# 五折: 0.52279 (线上)
# 训练时长: ~ 5 hours
#####################################################################################


import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd

import sys
import time
import pickle
import gc
import logging

from tqdm import tqdm

import gensim
from gensim.models import FastText, Word2Vec

from scipy.special import softmax

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

max_len = 120
emb_dim_cid = 64
emb_dim_advid = 32

batch_size = 1024


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

print("loading advertiser id")
df = pd.read_pickle('../../data/keras/df_advertiser_sequence.pickle')
advid_list = list(df['advids'])
for i in range(0, len(advid_list)):
    advid_list[i] =[str(ii) for ii in advid_list[i]]

x_advid, index_advid = set_tokenizer(advid_list, split_char=',')
emb_advid = get_embedding_matrix(index_advid,
                                 embed_size=emb_dim_advid,
                                 Emed_path='../../w2v_models/advid_w2v_32_win100_iter20_mincount1.txt')
del df, advid_list, index_advid
gc.collect()


print("loading creative id")
df = pd.read_pickle('../../data/keras/df_creative_sequence.pickle')
cid_list = list(df['cids'])
for i in range(0, len(cid_list)):
    cid_list[i] =[str(ii) for ii in cid_list[i]]
    
x_cid, index_cid = set_tokenizer(cid_list, split_char=',')
emb_cid = get_embedding_matrix(index_cid,
                               embed_size=emb_dim_cid,
                               Emed_path='../../w2v_models/cid_w2v_64_win100_iter20_mincount1.txt')
del df, cid_list, index_cid
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

y = keras.utils.to_categorical(labels['age'])

x_stacking = np.load('../../probs/x_stacking_120probs.npy')[:, :90]

if fold == "fold0":
    train_cid = x_cid[:2400000]
    valid_cid = x_cid[2400000:3000000]
    train_advid = x_advid[:2400000]
    valid_advid = x_advid[2400000:3000000]
    train_stacking = x_stacking[:2400000]
    valid_stacking = x_stacking[2400000:3000000]
    y_train = y[:2400000]
    y_valid = y[2400000:]
elif fold == "fold1":
    train_cid = np.concatenate((x_cid[:1800000], x_cid[2400000:3000000]), axis=0)
    valid_cid = x_cid[1800000:2400000]
    train_advid = np.concatenate((x_advid[:1800000], x_advid[2400000:3000000]), axis=0)
    valid_advid = x_advid[1800000:2400000]
    train_stacking = np.concatenate((x_stacking[:1800000], x_stacking[2400000:3000000]), axis=0)
    valid_stacking = x_stacking[1800000:2400000]
    y_train = np.concatenate((y[:1800000], y[2400000:3000000]))
    y_valid = y[1800000:2400000]
elif fold == "fold2":
    train_cid = np.concatenate((x_cid[:1200000], x_cid[1800000:3000000]), axis=0)
    valid_cid = x_cid[1200000:1800000]
    train_advid = np.concatenate((x_advid[:1200000], x_advid[1800000:3000000]), axis=0)
    valid_advid = x_advid[1200000:1800000]
    train_stacking = np.concatenate((x_stacking[:1200000], x_stacking[1800000:3000000]), axis=0)
    valid_stacking = x_stacking[1200000:1800000]
    y_train = np.concatenate((y[:1200000], y[1800000:3000000]))
    y_valid = y[1200000:1800000]
elif fold == "fold3":
    train_cid = np.concatenate((x_cid[:600000], x_cid[1200000:3000000]), axis=0)
    valid_cid = x_cid[600000:1200000]
    train_advid = np.concatenate((x_advid[:600000], x_advid[1200000:3000000]), axis=0)
    valid_advid = x_advid[600000:1200000]
    train_stacking = np.concatenate((x_stacking[:600000], x_stacking[1200000:3000000]), axis=0)
    valid_stacking = x_stacking[600000:1200000]
    y_train = np.concatenate((y[:600000], y[1200000:3000000]))
    y_valid = y[600000:1200000]
elif fold == "fold4":
    train_cid = x_cid[600000:3000000]
    valid_cid = x_cid[:600000]
    train_advid = x_advid[600000:3000000]
    valid_advid = x_advid[:600000]
    train_stacking = x_stacking[600000:3000000]
    valid_stacking= x_stacking[:600000]
    y_train = y[600000:3000000]
    y_valid = y[:600000]
else:
    pass

test_cid = x_cid[3000000:]
test_advid = x_advid[3000000:]
test_stacking = x_stacking[3000000:]

del x_cid, x_advid
del x_stacking
del y
gc.collect()

print(train_cid.shape, valid_cid.shape, test_cid.shape)
print(train_advid.shape, valid_advid.shape, test_advid.shape)
print(train_stacking.shape, valid_stacking.shape, test_stacking.shape)
print(y_train.shape, y_valid.shape)

used_minutes = (time.time() - start_time) / 60
print(f"done, used {used_minutes} minutes")


print("building model")

stacking_shape = train_stacking.shape[1]

start_time = time.time()

def build_model(emb_cid, emb_advid):

    inp1 = layers.Input(shape=(max_len,))
    inp2 = layers.Input(shape=(max_len,))
    inp_stacking = layers.Input(shape=(stacking_shape,))

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

    sdrop = layers.SpatialDropout1D(rate=0.1)

    emb1 = sdrop(emb1)
    emb2 = sdrop(emb2)

    content = layers.Concatenate()([emb1, emb2])

    mha1 = MultiHeadAttention(head_num=8)(content)
#     mha1 = layers.Dropout(0.01)(mha1)
    mha1 = layers.Add()([content, mha1])
    mha1 = LayerNormalization()(mha1)
#     mha1 = layers.Dropout(0.01)(mha1)
    mha1_ff = FeedForward(128)(mha1)
    mha1_out = layers.Add()([mha1, mha1_ff])
    mha1_out = LayerNormalization()(mha1_out)

#     mha2 = MultiHeadAttention(head_num=8)(mha1_out)
#     mha2 = layers.Dropout(0.01)(mha2)
#     mha2 = layers.Add()([mha1_out, mha2])
#     mha2 = LayerNormalization()(mha2)
#     mha2 = layers.Dropout(0.01)(mha2)
#     mha2_ff = FeedForward(128)(mha2)
#     mha2_out = layers.Add()([mha2, mha2_ff])
#     mha2_out = LayerNormalization()(mha2_out)

#     mha3 = MultiHeadAttention(head_num=8)(mha2_out)
#     mha3 = layers.Dropout(0.01)(mha3)
#     mha3 = layers.Add()([mha2_out, mha3])
#     mha3 = LayerNormalization()(mha3)
#     mha3 = layers.Dropout(0.01)(mha3)
#     mha3_ff = FeedForward(128)(mha3)
#     mha3_out = layers.Add()([mha3, mha3_ff])
#     mha3_out = LayerNormalization()(mha3_out)

#     avg_pool = layers.GlobalAveragePooling1D()(mha3_out)
    max_pool = layers.GlobalMaxPool1D()(mha1_out)

    x = layers.Concatenate()([max_pool, inp_stacking])

    x = layers.Dense(128, activation='relu')(x)
#     x = layers.BatchNormalization()(x)

    x = layers.Dense(64, activation='relu')(x)
#     x = layers.BatchNormalization()(x)

    x = layers.Dense(32, activation='relu')(x)
#     x = layers.BatchNormalization()(x)
    
#     x = layers.Dropout(0.1)(x)

    out = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=[inp1, inp2, inp_stacking], outputs=out)
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(5e-4),
                  metrics=['accuracy'])

    return model


model = build_model(emb_cid, emb_advid)

used_minutes = (time.time() - start_time) / 60
print(f"done, used {used_minutes} minutes")


checkpoint = callbacks.ModelCheckpoint(f'../../models/age_transformer_stacking_{fold}.h5',
                                       monitor='val_accuracy',
                                       verbose=1,
                                       save_best_only=True,
                                       mode='max',
                                       save_weights_only=True)

reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                        factor=0.1,
                                        patience=4,
                                        verbose=1,
                                        mode='max',
                                        epsilon=1e-6)

early_stop = callbacks.EarlyStopping(monitor='val_accuracy',
                                     mode='max',
                                     patience=10)


hist = model.fit([train_cid, train_advid, train_stacking],
                 y_train,
                 batch_size=batch_size,
                 epochs=100,
                 validation_data=([valid_cid, valid_advid, valid_stacking], y_valid),
                 callbacks=[
                     checkpoint, 
                     reduce_lr, 
                     early_stop],
                 verbose=1,
                 shuffle=True)


acc = max(hist.history['val_accuracy'])
acc


print("predict start")
start_time = time.time()

model.load_weights(f'../../models/age_transformer_stacking_{fold}.h5')
preds = model.predict([test_cid, test_advid, test_stacking],
                      batch_size=batch_size,
                      verbose=1)

np.save(f'../../probs/sub_age_transformer_stacking_{fold}', preds)

used_minutes = (time.time() - start_time) / 60
print(f"done, used {used_minutes} minutes")
