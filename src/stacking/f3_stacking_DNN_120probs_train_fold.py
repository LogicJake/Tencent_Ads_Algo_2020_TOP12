#!/usr/bin/env python
# coding: utf-8

#####################################################################################
# AGE Stacking: DNN 双路分层残差
# score: 
# 五折: 0.52131 (线下)
# 五折: 0.52446 (线上)
# 训练时长: ~ 2 hours
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

import keras
from keras import layers
from keras import callbacks



fold = sys.argv[1]
batch_size = 20480


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

x_stacking = np.load('../../probs/x_stacking_120probs.npy')

x1  = x_stacking[:,0:10]        # LGB
x2  = x_stacking[:,10:20]       # torch 6inputs lstm+attention 
x3  = x_stacking[:,20:30]       # torch 6inputs transformer 
x4  = x_stacking[:,30:40]       # keras 4inputs lstm+attention
x5  = x_stacking[:,40:50]       # keras 4inputs transformer
x6  = x_stacking[:,50:60]       # keras 3inputs transformer
x7  = x_stacking[:,60:70]       # keras 2inputs transformer+lstm
x8  = x_stacking[:,70:80]       # torch 3inputs lstm+attention
x9  = x_stacking[:,80:90]       # keras 3inputs transformer+lstm
x10 = x_stacking[:,90:100]      # torch 3inputs transformer+lstm
x11 = x_stacking[:,100:110]     # keras 4inputs transformer+lstm
x12 = x_stacking[:,110:120]     # keras 5inputs transformer+lstm

x_lgb = x1
x_keras = np.concatenate((x4, x5, x6, x7, x9, x11, x12), axis=1)
x_torch = np.concatenate((x2, x3, x8, x10), axis=1)

del x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12
gc.collect()

if fold == "fold0":
    train_lgb = x_lgb[:2400000]
    valid_lgb = x_lgb[2400000:3000000]
    train_keras = x_keras[:2400000]
    valid_keras = x_keras[2400000:3000000]
    train_torch = x_torch[:2400000]
    valid_torch = x_torch[2400000:3000000]
    y_train = y[:2400000]
    y_valid = y[2400000:]
elif fold == "fold1":
    train_lgb = np.concatenate((x_lgb[:1800000], x_lgb[2400000:3000000]), axis=0)
    valid_lgb = x_lgb[1800000:2400000]
    train_keras = np.concatenate((x_keras[:1800000], x_keras[2400000:3000000]), axis=0)
    valid_keras = x_keras[1800000:2400000]
    train_torch = np.concatenate((x_torch[:1800000], x_torch[2400000:3000000]), axis=0)
    valid_torch = x_torch[1800000:2400000]
    y_train = np.concatenate((y[:1800000], y[2400000:3000000]))
    y_valid = y[1800000:2400000]
elif fold == "fold2":
    train_lgb = np.concatenate((x_lgb[:1200000], x_lgb[1800000:3000000]), axis=0)
    valid_lgb = x_lgb[1200000:1800000]
    train_keras = np.concatenate((x_keras[:1200000], x_keras[1800000:3000000]), axis=0)
    valid_keras = x_keras[1200000:1800000]
    train_torch = np.concatenate((x_torch[:1200000], x_torch[1800000:3000000]), axis=0)
    valid_torch = x_torch[1200000:1800000]
    y_train = np.concatenate((y[:1200000], y[1800000:3000000]))
    y_valid = y[1200000:1800000]
elif fold == "fold3":
    train_lgb = np.concatenate((x_lgb[:600000], x_lgb[1200000:3000000]), axis=0)
    valid_lgb = x_lgb[600000:1200000]
    train_keras = np.concatenate((x_keras[:600000], x_keras[1200000:3000000]), axis=0)
    valid_keras = x_keras[600000:1200000]
    train_torch = np.concatenate((x_torch[:600000], x_torch[1200000:3000000]), axis=0)
    valid_torch = x_torch[600000:1200000]
    y_train = np.concatenate((y[:600000], y[1200000:3000000]))
    y_valid = y[600000:1200000]
elif fold == "fold4":
    train_lgb = x_lgb[600000:3000000]
    valid_lgb = x_lgb[:600000]
    train_keras = x_keras[600000:3000000]
    valid_keras = x_keras[:600000]
    train_torch = x_torch[600000:3000000]
    valid_torch = x_torch[:600000]
    y_train = y[600000:3000000]
    y_valid = y[:600000]
else:
    pass

test_lgb = x_lgb[3000000:]
test_keras = x_keras[3000000:]
test_torch = x_torch[3000000:]

del x_stacking, x_lgb, x_keras, x_torch
del y
gc.collect()

print(train_lgb.shape, valid_lgb.shape, test_lgb.shape)
print(train_keras.shape, valid_keras.shape, test_keras.shape)
print(train_torch.shape, valid_torch.shape, test_torch.shape)
print(y_train.shape, y_valid.shape)

used_minutes = (time.time() - start_time) / 60
print(f"done, used {used_minutes} minutes")



print("building model")

lgb_shape = train_lgb.shape[1]
keras_shape = train_keras.shape[1]
torch_shape = train_torch.shape[1]

start_time = time.time()

def build_model():

    inp1 = layers.Input(shape=(lgb_shape,))
    inp2 = layers.Input(shape=(keras_shape,))
    inp3 = layers.Input(shape=(torch_shape,))

    x1 = layers.Concatenate()([inp1, inp2])
    x1 = layers.Dense(40, activation='relu')(inp1)
    x1 = layers.BatchNormalization()(x1)

    x2 = layers.Concatenate()([x1, inp3])
    x2 = layers.Dense(40, activation='relu')(x2)
    x2 = layers.BatchNormalization()(x2)
    
    x3 = layers.Dense(40, activation='relu')(x2)
    x3 = layers.BatchNormalization()(x3)
    
    x_all = layers.Concatenate()([inp1, inp2, inp3])
    x_all = layers.Dense(120, activation='relu')(x_all)
    x_all = layers.BatchNormalization()(x_all)
    
    x_all = layers.Dense(80, activation='relu')(x_all)
    x_all = layers.BatchNormalization()(x_all)
    
    x_all = layers.Dense(60, activation='relu')(x_all)
    x_all = layers.BatchNormalization()(x_all)
    
    x = layers.Concatenate()([x3, x_all])
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    out = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=[inp1, inp2, inp3], outputs=out)
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model


model = build_model()

used_minutes = (time.time() - start_time) / 60
print(f"done, used {used_minutes} minutes")


# In[6]:


model.summary()


# In[7]:


checkpoint = callbacks.ModelCheckpoint(f'../../models/age_dnn_stacking_{fold}.h5',
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


hist = model.fit([train_lgb, train_keras, train_torch],
                 y_train,
                 batch_size=batch_size,
                 epochs=100,
                 validation_data=([valid_lgb, valid_keras, valid_torch], y_valid),
                 callbacks=[checkpoint, reduce_lr, early_stop],
                 verbose=1,
                 shuffle=True)

acc = max(hist.history['val_accuracy'])
print(acc)


print("predict start")
start_time = time.time()

model.load_weights(f'../../models/age_dnn_stacking_{fold}.h5')
preds = model.predict([test_lgb, test_keras, test_torch],
                      batch_size=batch_size,
                      verbose=1)

np.save(f'../../probs/sub_age_dnn_stacking_{fold}', preds)

used_minutes = (time.time() - start_time) / 60
print(f"done, used {used_minutes} minutes")

