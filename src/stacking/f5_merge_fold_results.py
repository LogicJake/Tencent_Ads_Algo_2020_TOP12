#!/usr/bin/env python
# coding: utf-8

#############################################################################
# 将五折跑完的结果合并成 oof 和 sub 概率
#############################################################################

import sys

import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

# 参数类似 age_m3_keras or gender_m1_keras
name = sys.argv[1]

probs_pth = '../../probs'

train_f0 = np.load(f'{probs_pth}/oof_{name}_4.npy')
train_f1 = np.load(f'{probs_pth}/oof_{name}_3.npy')
train_f2 = np.load(f'{probs_pth}/oof_{name}_2.npy')
train_f3 = np.load(f'{probs_pth}/oof_{name}_1.npy')
train_f4 = np.load(f'{probs_pth}/oof_{name}_0.npy')

test_f0 = np.load(f'{probs_pth}/sub_{name}_4.npy')
test_f1 = np.load(f'{probs_pth}/sub_{name}_3.npy')
test_f2 = np.load(f'{probs_pth}/sub_{name}_2.npy')
test_f3 = np.load(f'{probs_pth}/sub_{name}_1.npy')
test_f4 = np.load(f'{probs_pth}/sub_{name}_0.npy')


oof_probs = np.concatenate((train_f0, train_f1, train_f2, train_f3, train_f4), axis=0)
test_probs = (test_f0 + test_f1 + test_f2 + test_f3 + test_f4) / 5


np.save(f'{probs_pth}/oof_{name}', oof_probs)
np.save(f'{probs_pth}/sub_{name}', test_probs)

