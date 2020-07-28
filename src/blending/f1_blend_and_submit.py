#!/usr/bin/env python
# coding: utf-8

##########################################################################################
# AGE: 0.50 * DNN_stacking + 0.15 * transformer_stacking + 0.35 * age_m13  score: 0.52780
# GENDER: 0.35 * gender_m1 + 0.35 * gender_m2 + 0.30 * gender_m3           score: 0.95048
##########################################################################################

import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np

from scipy.special import softmax


prob1_age_0 = np.load('../../probs/sub_age_dnn_stacking_0.npy')
prob1_age_1 = np.load('../../probs/sub_age_dnn_stacking_1.npy')
prob1_age_2 = np.load('../../probs/sub_age_dnn_stacking_2.npy')
prob1_age_3 = np.load('../../probs/sub_age_dnn_stacking_3.npy')
prob1_age_4 = np.load('../../probs/sub_age_dnn_stacking_4.npy')
prob1_age = (prob1_age_0 + prob1_age_1 + prob1_age_2 + prob1_age_3 + prob1_age_4) / 5

prob2_age_0 = np.load('../../probs/sub_age_transformer_stacking_0.npy')
prob2_age_1 = np.load('../../probs/sub_age_transformer_stacking_1.npy')
prob2_age_2 = np.load('../../probs/sub_age_transformer_stacking_2.npy')
prob2_age_3 = np.load('../../probs/sub_age_transformer_stacking_3.npy')
prob2_age_4 = np.load('../../probs/sub_age_transformer_stacking_4.npy')
prob2_age = (prob2_age_0 + prob2_age_1 + prob2_age_2 + prob2_age_3 + prob2_age_4) / 5

prob3_age = softmax(np.load('../../probs/sub_age_m13_torch.npy'), axis=1)

prob_age = 0.5*prob1_age + 0.15*prob2_age + 0.35*prob3_age


prob1_gender_0 = np.load('../../probs/sub_gender_m1_keras_0.npy')[:,0]
prob1_gender_1 = np.load('../../probs/sub_gender_m1_keras_1.npy')[:,0]
prob1_gender_2 = np.load('../../probs/sub_gender_m1_keras_2.npy')[:,0]
prob1_gender_3 = np.load('../../probs/sub_gender_m1_keras_3.npy')[:,0]
prob1_gender_4 = np.load('../../probs/sub_gender_m1_keras_4.npy')[:,0]
prob1_gender = (prob1_gender_0 + prob1_gender_1 + prob1_gender_2 + prob1_gender_3 + prob1_gender_4) / 5

prob2_gender_0 = np.load('../../probs/sub_gender_m2_keras_0.npy')[:,0]
prob2_gender_1 = np.load('../../probs/sub_gender_m2_keras_1.npy')[:,0]
prob2_gender_2 = np.load('../../probs/sub_gender_m2_keras_2.npy')[:,0]
prob2_gender_3 = np.load('../../probs/sub_gender_m2_keras_3.npy')[:,0]
prob2_gender_4 = np.load('../../probs/sub_gender_m2_keras_4.npy')[:,0]
prob2_gender = (prob2_gender_0 + prob2_gender_1 + prob2_gender_2 + prob2_gender_3 + prob2_gender_4) / 5

prob3_gender = softmax(np.load('../../probs/sub_gender_m3_torch.npy'), axis=1)[:,1]

prob_gender = 0.35*prob1_gender + 0.35*prob2_gender + 0.3*prob3_gender


sub = pd.DataFrame({'user_id': range(3000001,4000001), 'predicted_age': [-1]*1000000, 'predicted_gender': [-1]*1000000})
sub['predicted_age'] = np.argmax(prob_age, axis=1) + 1


sub['prob_gender'] = prob_gender
sub['predicted_gender'] = sub['prob'].apply(lambda x: 2 if x>0.5 else 1)
sub.drop(['prob_gender'], axis=1, inplace=True)



sub.to_csv('../../submissions.csv')

