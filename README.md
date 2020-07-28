队伍: 最后一次打比赛  
队员：@jackhuntcn @PandasCute @LogicJake
## 运行环境
### 硬件
- P40 显存 24G
- 内存 114G 或以上
- 磁盘 300G 或以上

### 软件
run.sh 有安装指令

- gensim
- torcch
- transformers
- keras == 2.3.1
- keras_self_attention 
- keras_multi_head 
- keras_position_wise_feed_forward 
- keras_layer_normalization
- lightgbm

## 代码目录

```
.
├── data
│   ├── keras
│   ├── lgb
│   └── torch
├── models
├── probs
├── raw_data
│   ├── test
│   ├── train_preliminary
│   └── train_semi_final
├── run.sh
├── src
│   ├── blending
│   │   └── f1_blend_and_submit.py
│   ├── keras
│   │   ├── f10_AGE_m11_transformer_lstm_5inputs_train_fold.py
│   │   ├── f11_GENDER_m1_transformer_3inputs_train_fold.py
│   │   ├── f12_GENDER_m2_transformer_lstm_3inputs_train_fold.py
│   │   ├── f13_merge_fold_results.py
│   │   ├── f1_save_data.py
│   │   ├── f2_save_sequence.py
│   │   ├── f3_save_embeddings.py
│   │   ├── f4_AGE_m3_lstm_4inputs_train_fold.py
│   │   ├── f5_AGE_m4_transformer_4inputs_train_fold.py
│   │   ├── f6_AGE_m5_transformer_3inputs_train_fold.py
│   │   ├── f7_AGE_m6_transformer_lstm_2inputs_train_fold.py
│   │   ├── f8_AGE_m8_transformer_lstm_3inputs_2r_train_fold.py
│   │   └── f9_AGE_m10_transformer_lstm_5inputs_train_fold.py
│   ├── lgb
│   │   ├── f1_save_tfidf_countvec.py
│   │   ├── f2_save_target_encoding.py
│   │   ├── f3_save_AGE_tf_idf_stacking_feats.py
│   │   ├── f4_save_GENDER_tf_idf_stacking_feats.py
│   │   └── f5_run_fold_training.py
│   ├── stacking
│   │   ├── f1_merge_stacking_feats.py
│   │   ├── f2_save_embeddings.py
│   │   ├── f3_stacking_DNN_120probs_train_fold.py
│   │   ├── f4_stacking_transformer_2inputs_90probs_train_fold.py
│   │   └── f5_merge_fold_results.py
│   └── torch
│       ├── f1_save_data.py
│       ├── f2_save_embedding_w2v.py
│       ├── f3_AGE_m7_lstm_3inputs_train_5fold.py
│       ├── f4_AGE_m1_lstm_6inputs_train_5fold.py
│       ├── f5_AGE_m9_transformer_3inputs_train_5fold.py
│       ├── f6_AGE_m2_transformer_6inputs_train_5fold.py
│       ├── f7_save_data.py
│       ├── f8_AGE_GENDER_m13_transformer_4inputs_train_5fold.py
│       ├── lookahead.py
│       ├── m13_transformer_4inputs.py
│       ├── m1_lstm_6inputs_age.py
│       ├── m2_transformer_6inputs_age.py
│       ├── m7_lstm_3inputs_age.py
│       ├── m9_transformer_3inputs_age.py
│       └── pytorchtools.py
└── w2v_models

17 directories, 40 files
```

- src 运行代码目录, 分为 torch/keras/lgb 三种框架
- data 预处理完成数据目录
- models 模型生成目录
- probs 模型生成概率存放目录
- raw_data 比赛的原始数据, 包含初赛和复赛数据
- run.sh 一键执行脚本
- w2v_models 为存放 w2v embedding 模型的目录

## 模型说明

## keras

对 age 和 gender 两个目标分别建模

每种模型分为不同的输入 id 个数，具体模型如下: (分数均指 A 榜分数，下同)

#### AGE

- LSTM + Attention 四输入五折, 线上大概 0.512
- transformer 四输入五折, 线上 0.516
- transformer 三输入五折, 线上 0.515
- transformer + LSTM 二输入五折，线上大概 0.515
- transformer + LSTM 三输入五折，线上大概 0.515
- transformer + LSTM 四输入五折，线上 0.517
- transformer + LSTM 五输入五折，线上大概 0.517

#### GENDER
- transformer 三输入五折, 线上 0.9500
- transformer + LSTM 三输入五折, 线上 0.9501


## torch

原生 transformers 对 age 和 gender 两个目标分别建模

huggingface transformers 同时对两个目录建模，两路输出

#### AGE

- LSTM + Attention 六输入五折, 线上 0.513 
- transformer + LSTM 六输入五折, 线上 0.516
- transformer + LSTM 三输入五折, 线上 0.514

#### AGE & GENDER

- AGE: transformer + LSTM 四输入五折, 线上 0.519
- GENDER: transformer + LSTM 四输入五折, 线下 0.9468, 线上未测

## LGB

LGB 使用的特征为 TF-IDF 和 COUNTVEC 以及目标编码特征

后期加入了线性模型产生的概率特征, AGE 线下分数大概 0.48 

本次比赛中表现不如 NN 强势, 只采用了 AGE 概率用于 stacking

## stacking

本次比赛使用了两种 stacking 方式:

- 纯概率特征 stacking: 将上面模型跑出来的概率分层进入 DNN (如 keras 产生的概率与 keras 产生的概率合并，torch 概率与 torch 概率合并, keras 概率先进入, torch 在 keras 概率经过了几层全连接之后再 concat, 实验证明这种做法可以避免相关性较高的概率带来的融合不利影响), 五折线上分数大概为 0.525
- 混合特征 stacking: 一方面采用了不同的两 id 序列输入的 transformers 模型，在最后经过全连接层之前跟上面的九个相关性较低的模型进行 concat, 起到一种类似于残差的作用，避免过拟合, 五折线上分数为 0.523

## blending

### AGE

比赛结束前一周我们使用了 huggingface transformers 重新实现了 transformer + LSTM 模型, 五折分数为 0.519, 而且相关度与之前实现的 keras 和 torch 都较低, 只有 0.93,0.94 左右 (相比 keras 之间的相似度高达 0.98, torch 之间相似度 0.96), 所有我们单独将这个模型与上面所生成的两个 stacking 模型进行融合，取得 0.52780 线上分数： 


```
0.50 * DNN_stacking + 0.15 * transformer_stacking + 0.35 * age_m13
```

### GENDER

Gender 主要是三个 transformer 模型进行基本均等的融合：线上分数 0.95048

```
0.35 * gender_m1 + 0.35 * gender_m2 + 0.30 * gender_m3
```