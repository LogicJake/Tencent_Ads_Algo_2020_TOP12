#!/usr/bin/env python
# coding: utf-8

#################################################################################
# AGE model 9: Torch Transformer+LSTM 3 inputs
# score: 
# 五折: 0.50078 (线下)
# 五折: 0.51433 (线上)
# 训练时长: ~ 4 days
#################################################################################


import pandas as pd
import warnings
import gc
from tqdm import tqdm
import numpy as np
from collections import defaultdict, OrderedDict
import math
import pickle
import random
import torch
import torch.nn as nn
from sklearn import preprocessing
from pytorchtools import EarlyStopping
import os
from sklearn.model_selection import KFold
import torch_optimizer as optim
from sklearn.metrics import accuracy_score
from m9_transformer_3inputs_age import *

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

df_log = pd.read_pickle('../../data/torch/log.pkl')

seq_embedding_features = OrderedDict({
    'creative_id': {
        'embedding_file': '../../w2v_models/w2v_creative_id_128.pkl',
        'embedding_dim': 128,
        'pretrained_embedding': None,
    },
    'advertiser_id': {
        'embedding_file': '../../w2v_models/w2v_advertiser_id_128.pkl',
        'embedding_dim': 128,
        'pretrained_embedding': None,
    },
    'product_id': {
        'embedding_file': '../../w2v_models/w2v_product_id_128.pkl',
        'embedding_dim': 128,
        'pretrained_embedding': None,
    },
})


for f in tqdm(seq_embedding_features.keys()):
    le = preprocessing.LabelEncoder()
    le.fit(df_log[f].values.tolist())

    df_emb = pd.read_pickle(seq_embedding_features[f]['embedding_file'])
    df_emb = df_emb[df_emb[f].isin(df_log[f].values.tolist())]
    assert df_emb.shape[1] == seq_embedding_features[f]['embedding_dim'] + 1
    df_emb[f] = le.transform(df_emb[f].values.tolist()) + 1

    # 补上作为序列填补的 0 向量
    df_default = pd.DataFrame()
    df_default[f] = [0]
    df_emb = df_emb.append(df_default)
    df_emb.fillna(0, inplace=True)

    # 按 id 排序
    df_emb.sort_values([f], inplace=True)
    embedding_columns = [c for c in df_emb.columns if c != f]
    seq_embedding_features[f]['pretrained_embedding'] = [
        v for v in df_emb[embedding_columns].values
    ]

    del df_default, df_emb
    gc.collect()

    df_log[f] = le.transform(df_log[f].values.tolist()) + 1
    seq_embedding_features[f]['nunique'] = df_log[f].nunique() + 1


# # 序列特征
# ## 序列 id 特征
seq_len = 128

def gen_seq_data(data, features, seq_len, prefix=''):
    data.sort_values('time', inplace=True)
    data_set = OrderedDict()

    user_ids = []
    for user_id, hist in tqdm(data.groupby('user_id')):
        user_ids.append(user_id)

        # 取最近的记录
        for f in features:
            hist_f = hist[f].values
            hist_f = hist_f[-seq_len:]

            if f not in data_set:
                data_set[f] = []

            data_set[f].append(hist_f)

    for f in features:
        df_context = pd.DataFrame()
        df_context['user_id'] = user_ids
        df_context['{}_seq'.format(f)] = data_set[f]

        df_context.to_pickle('../../data/torch/{}{}_seqs_{}.pkl'.format(prefix, f, seq_len)) 



# 是否从本地加载seq数据
load_seq = True
prefix = str(df_log.shape[0]) + '_' if df_log.shape[0] < 10000 else ''
# 不加载本地seq，强制重新生成所有seq
seq_features = []
if not load_seq:
    seq_features = list(seq_embedding_features.keys())
else:
    for f in seq_embedding_features.keys():
        if not os.path.exists('../../data/torch/{}{}_seqs_{}.pkl'.format(prefix, f, seq_len)):
            seq_features += [f]
print(seq_features)

if len(seq_features) != 0:
    gen_seq_data(df_log, seq_features, seq_len, prefix)

# 合并序列
all_users = list(df_log['user_id'].unique())
all_users.sort()
df_context = pd.DataFrame(all_users)
df_context.columns = ['user_id']
for f in seq_embedding_features.keys():
    df_seq = pd.read_pickle('../../data/torch/{}{}_seqs_{}.pkl'.format(prefix, f, seq_len))
    df_context = df_context.merge(df_seq, how='left')
    del df_seq
    gc.collect()


# ## 序列统计特征

seq_statistics_features = []
df_statistics_context = None


# # 合并其他特征

# ## 标签

df_feature = pd.read_pickle('../../data/torch/feature.pkl')
df_feature['age'] = df_feature['age'].astype('float')
df_feature['age'] = df_feature['age'] - 1
del df_feature['gender']

user_ids = list(set(df_log['user_id'].values))
df_feature = df_feature[df_feature['user_id'].isin(user_ids)]
df_feature.sort_values(['user_id'], inplace=True)
df_feature.reset_index(drop=True, inplace=True)

df_feature = df_feature.merge(df_context, how='left')

if df_statistics_context:
    df_feature = df_feature.merge(df_statistics_context, how='left')
    del df_statistics_context

del df_context
gc.collect()

del df_log
gc.collect()


# ## target encoding 特征

statistics_features = []


# # 模型训练

train_model_inputs = df_feature[df_feature['age'].notnull()].reset_index(
    drop=True)
test_model_inputs = df_feature[df_feature['age'].isnull()].reset_index(
    drop=True)


kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(train_model_inputs)):
    print('\nFold_{} Training ============================================\n'.
          format(fold_id + 1))

    train_data = train_model_inputs.iloc[trn_idx]
    val_data = train_model_inputs.iloc[val_idx]

    # 模型定义
    model = LSTMCLF(seq_embedding_features=seq_embedding_features,
                    statistics_features=statistics_features,
                    seq_statistics_features=seq_statistics_features,
                    seq_len=seq_len,
                    device=device).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Ranger(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           factor=0.1,
                                                           patience=2,
                                                           min_lr=1e-6,
                                                           verbose=True)
    early_stopping = EarlyStopping(
        file_name='../../models/age_m9_checkpoint{}.pt'.format(fold_id),
        patience=10,
        verbose=True,
        delta=0.00000001)

    model.set(criterion, optimizer, scheduler, early_stopping)

    batch_size = 512
    # 6000
    epoches = 600
    best_age_acc = model.model_train(train_data, val_data, epoches, batch_size)
    print('age_acc: {}'.format(best_age_acc))
    

test_data = test_model_inputs

oof_pred_age = np.zeros((train_model_inputs.shape[0], 10))
test_pred_age = np.zeros((test_data.shape[0], 10))

kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(train_model_inputs)):
    print('\nFold_{} Training ============================================\n'.
          format(fold_id + 1))

    model = LSTMCLF(seq_embedding_features=seq_embedding_features,
                    statistics_features=statistics_features,
                    seq_statistics_features=seq_statistics_features,
                    seq_len=seq_len,
                    device=device).to(device)
    model.load_state_dict(torch.load('../../models/age_m9_checkpoint{}.pt'.format(fold_id)), strict=False)
    
    model.eval()
    
    with torch.no_grad():
        val_data = train_model_inputs.iloc[val_idx]

        # 对训练集预测
        model_pred_age, _, _ = model.model_predict(val_data, batch_size, False)
        oof_pred_age[val_idx] += model_pred_age

        # 对测试集预测
        model_pred_age, _, _ = model.model_predict(test_data, batch_size, False)
        test_pred_age += model_pred_age / 5


df_oof = train_model_inputs[['user_id', 'age']]
df_oof['predicted_age'] = np.argmax(oof_pred_age, axis=1)
acc_age = accuracy_score(df_oof['age'], df_oof['predicted_age'])
print(acc_age)


np.save('../../probs/sub_age_m9_torch', test_pred_age)
np.save('../../probs/oof_age_m9_torch', oof_pred_age)




