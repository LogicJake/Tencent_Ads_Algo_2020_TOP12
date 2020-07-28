#!/bin/bash

# 安装必要的 packages

pip install gensim
pip install keras==2.3.1
pip install keras_self_attention keras_multi_head keras_position_wise_feed_forward keras_layer_normalization
pip install pytorch
pip install transformers
pip install lightgbm

# keras

cd src/keras

python f1_save_data.py
python f2_save_sequence.py
python f3_save_embeddings.py
for i in `seq 0 4`; do python f4_AGE_m3_lstm_4inputs_train_fold.py "fold${i}"; done
for i in `seq 0 4`; do python f5_AGE_m4_transformer_4inputs_train_fold.py "fold${i}"; done
for i in `seq 0 4`; do python f6_AGE_m5_transformer_3inputs_train_fold.py "fold${i}"; done
for i in `seq 0 4`; do python f7_AGE_m6_transformer_lstm_2inputs_train_fold.py "fold${i}"; done
for i in `seq 0 4`; do python f8_AGE_m8_transformer_lstm_3inputs_2r_train_fold.py "fold${i}"; done
for i in `seq 0 4`; do python f9_AGE_m10_transformer_lstm_5inputs_train_fold.py "fold${i}"; done
for i in `seq 0 4`; do python f10_AGE_m11_transformer_lstm_5inputs_train_fold.py "fold${i}"; done
for i in `seq 0 4`; do python f11_GENDER_m1_transformer_3inputs_train_fold.py "fold${i}"; done
for i in `seq 0 4`; do python f12_GENDER_m2_transformer_lstm_3inputs_train_fold.py "fold${i}"; done
python f13_merge_fold_results.py 'age_m3_keras'
python f13_merge_fold_results.py 'age_m4_keras'
python f13_merge_fold_results.py 'age_m5_keras'
python f13_merge_fold_results.py 'age_m6_keras'
python f13_merge_fold_results.py 'age_m8_keras'
python f13_merge_fold_results.py 'age_m10_keras'
python f13_merge_fold_results.py 'age_m11_keras'
python f13_merge_fold_results.py 'gender_m1_keras'
python f13_merge_fold_results.py 'gender_m2_keras'

cd ../..

# torch

cd src/torch

python f1_save_data.py
python f2_save_embedding_w2v.py
python f3_AGE_m7_lstm_3inputs_train_5fold.py
python f4_AGE_m1_lstm_6inputs_train_5fold.py
python f5_AGE_m9_transformer_3inputs_train_5fold.py
python f6_AGE_m2_transformer_6inputs_train_5fold.py
python f7_save_data.py
python f8_AGE_GENDER_m13_transformer_4inputs_train_5fold.py

cd ../..

# lgb

cd src/lgb

python f1_save_tfidf_countvec.py
python f2_save_target_encoding.py
python f3_save_AGE_tf_idf_stacking_feats.py
python f4_save_GENDER_tf_idf_stacking_feats.py
python f5_run_fold_training.py

cd ../..

# stacking

cd src/stacking

python f1_merge_stacking_feats.py
python f2_save_embeddings.py
for i in `seq 0 4`; do python f3_stacking_DNN_120probs_train_fold.py "fold${i}"; done
for i in `seq 0 4`; do python f4_stacking_transformer_2inputs_90probs_train_fold.py "fold${i}"; done
python f5_merge_fold_results.py "dnn_stacking"
python f5_merge_fold_results.py "transformer_stacking"

cd ../..

# blend and submit

cd src/blending

python f1_blend_and_submit.py
