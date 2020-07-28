#!/usr/bin/env python
# coding: utf-8


#################################################################################
# AGE model 13: Torch huggingface transformer 4 inputs
# score: 
# 五折: 0.50699 (线下)
# 五折: 0.51866 (线上)
# GENDER model 3:
# score:
# 五折: 0.94678 (线下) 

# 训练时长: ~ 5 days
#################################################################################

import pandas as pd
from keras.preprocessing import text, sequence
import torch
import random
import numpy as np
import os
from gensim.models import Word2Vec
import warnings
from pytorchtools import EarlyStopping
from tqdm import tqdm
import torch_optimizer as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import gc
from m13_transformer_4inputs import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

warnings.filterwarnings('ignore')

torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed = 2020
fix_seed(seed)


df_data = pd.read_pickle('../../data/torch/data.pkl')
df_data['age'] = df_data['age'] - 1
df_data['gender'] = df_data['gender'] - 1


test = False
df_data = df_data.reset_index(drop=True)


seq_len = 128

def sequence_processing(df_data, col, embedding_dim):
    print('Generate {} seqs'.format(col))
    os.makedirs('../../data/torch/seqs', exist_ok=True)
    seq_path = '../../data/torch/seqs/seqs_{}_{}.npy'.format(col, seq_len)
    word_index_path = '../../data/torch/seqs/word_index_{}_{}.npy'.format(col, seq_len)
    if test or not os.path.exists(seq_path) or not os.path.exists(word_index_path):
        tokenizer = text.Tokenizer(lower=False)
        tokenizer.fit_on_texts(df_data[col].values.tolist())
        seqs = sequence.pad_sequences(tokenizer.texts_to_sequences(df_data[col].values.tolist()), maxlen=seq_len,
                                        padding='post', truncating='pre')
        word_index = tokenizer.word_index
        
        if not test:
            np.save(seq_path, seqs)
            np.save(word_index_path, word_index)
    else:
        seqs = np.load(seq_path)
        word_index = np.load(word_index_path, allow_pickle=True).item()
    
    print('Generate {} embedding'.format(col))
    os.makedirs('../../data/torch/embedding', exist_ok=True)
    embedding_path = '../../data/torch/embedding/w2v_{}_{}.m'.format(col, embedding_dim)
    if test or not os.path.exists(embedding_path):
        print('Training {} w2v'.format(col))

        sentences = []
        for sentence in df_data[col].values:
            sentence = sentence.split(' ')
            sentences.append(sentence)

        model = Word2Vec(sentences, size=embedding_dim, window=20, workers=32, seed=seed, min_count=1, sg=1, hs=1)
        if not test:
            model.save(embedding_path)
    else:
        model = Word2Vec.load(embedding_path)

    embedding = np.zeros((len(word_index)+1, embedding_dim))
    for word, i in tqdm(word_index.items()):
        if word in model:
            embedding[i] = model[word]

    return seqs, embedding



creative_id_seqs, creative_id_embedding = sequence_processing(df_data, 'creative_id', 128)
ad_id_seqs, ad_id_embedding = sequence_processing(df_data, 'ad_id', 128)
advertiser_id_seqs, advertiser_id_embedding = sequence_processing(df_data, 'advertiser_id', 128)
product_id_seqs, product_id_embedding = sequence_processing(df_data, 'product_id', 128)


all_index = df_data[df_data['age'].notnull()].index.tolist()
test_index = df_data[df_data['age'].isnull()].index.tolist()


target = df_data[['user_id', 'age', 'gender']].copy(deep=True)
del df_data
gc.collect()


kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
for fold_id, (train_index, val_index) in enumerate(kfold.split(all_index)):
    model = Model(embeddings=[creative_id_embedding, ad_id_embedding, advertiser_id_embedding, product_id_embedding],
                  device=device).to(device).to(device)
    criterion_age = nn.CrossEntropyLoss().to(device)
    criterion_gender = nn.CrossEntropyLoss().to(device)

    optimizer = optim.Ranger(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           factor=0.1,
                                                           patience=1,
                                                           min_lr=1e-12,
                                                           verbose=True)
    early_stopping = EarlyStopping(
        file_name='../../models/age_m13_gender_m3_checkpoint{}.pt'.format(fold_id),
        patience=5,
        verbose=True,
        delta=0.00000001)

    model.set(criterion_age,criterion_gender, optimizer, scheduler, early_stopping)

    batch_size = 256
    # 6000
    epoches = 10000

    train_creative_id_seqs = creative_id_seqs[train_index]
    train_ad_id_seqs = ad_id_seqs[train_index]
    train_advertiser_id_seqs = advertiser_id_seqs[train_index]
    train_product_id_seqs = product_id_seqs[train_index]

    train_age = target.loc[train_index]['age'].values
    train_gender = target.loc[train_index]['gender'].values


    best_acc, best_age_acc, best_gender_acc = model.model_train(train_input=[train_creative_id_seqs,
                                                                             train_ad_id_seqs,
                                                                             train_advertiser_id_seqs,
                                                                             train_product_id_seqs],
                                                                val_input=[creative_id_seqs[val_index],
                                                                           ad_id_seqs[val_index], 
                                                                           advertiser_id_seqs[val_index],
                                                                           product_id_seqs[val_index]],
                                                                train_output=[train_age, train_gender],
                                                                val_output=[target.loc[val_index]['age'].values, target.loc[val_index]['gender'].values],
                                                                epoches=epoches, 
                                                                batch_size=batch_size)


batch_size = 256

oof_pred_age = np.zeros((len(all_index), 10))
test_pred_age = np.zeros((len(test_index), 10))

oof_pred_gender = np.zeros((len(all_index), 2))
test_pred_gender = np.zeros((len(test_index), 2))

test_creative_id_seqs = creative_id_seqs[test_index]
test_ad_id_seqs = ad_id_seqs[test_index]
test_advertiser_id_seqs = advertiser_id_seqs[test_index]
test_product_id_seqs = product_id_seqs[test_index]

kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
for fold_id, (train_index, val_index) in enumerate(kfold.split(all_index)):
    print('\nFold_{} Training ============================================\n'.
          format(fold_id + 1))

    model = Model(embeddings=[creative_id_embedding, ad_id_embedding, advertiser_id_embedding, product_id_embedding],
                  device=device).to(device).to(device)
    model.load_state_dict(torch.load('../../models/age_m13_gender_m3_checkpoint{}.pt'.format(fold_id)), strict=False)    
    model.eval()
    
    # 对训练集预测
    val_creative_id_seqs = creative_id_seqs[val_index]
    val_ad_id_seqs = ad_id_seqs[val_index]
    val_advertiser_id_seqs = advertiser_id_seqs[val_index]
    val_product_id_seqs = product_id_seqs[val_index]
    
    pred_age, pred_gender = model.model_predict([val_creative_id_seqs, val_ad_id_seqs,
                                                   val_advertiser_id_seqs, val_product_id_seqs],
                                                  batch_size)
    
    oof_pred_age[val_index] = pred_age
    oof_pred_gender[val_index] = pred_gender

    # 对测试集预测
    pred_age, pred_gender = model.model_predict([test_creative_id_seqs, test_ad_id_seqs,
                                                   test_advertiser_id_seqs, test_product_id_seqs],
                                                  batch_size)

    test_pred_age += pred_age / 5
    test_pred_gender += pred_gender / 5


df_oof = target.loc[all_index][['user_id', 'age', 'gender']]
df_oof['predicted_age'] = np.argmax(oof_pred_age, axis=1)
df_oof['predicted_gender'] = np.argmax(oof_pred_gender, axis=1)

acc_age = accuracy_score(df_oof['age'], df_oof['predicted_age'])
acc_gender = accuracy_score(df_oof['gender'], df_oof['predicted_gender'])
acc = acc_age + acc_gender
print(acc, acc_age, acc_gender)


np.save('../../probs/sub_age_m13_torch', test_pred_age)
np.save('../../probs/oof_age_m13_torch', oof_pred_age)


np.save('../../probs/sub_gender_m3_torch', test_pred_gender)
np.save('../../probs/oof_gender_m3_torch', oof_pred_gender)
