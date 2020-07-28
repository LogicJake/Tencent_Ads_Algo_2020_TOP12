import numpy as np
import pandas as pd

from scipy.special import softmax

probs_pth = '../../probs'

x1_oof = softmax(np.load(f'{probs_pth}/oof_age_m1_torch'), axis=1)
x1_preds = softmax(np.load(f'{probs_pth}/sub_age_m1_torch'), axis=1)
x1 = np.concatenate((x1_oof, x1_preds), axis=0)

x2_oof = softmax(np.load(f'{probs_pth}/oof_age_m2_torch'), axis=1)
x2_preds = softmax(np.load(f'{probs_pth}/sub_age_m2_torch'), axis=1)
x2 = np.concatenate((x2_oof, x2_preds), axis=0)

x3_oof = np.load(f'{probs_pth}/oof_age_m3_keras')
x3_preds = np.load(f'{probs_pth}/sub_age_m3_keras')
x3 = np.concatenate((x3_oof, x3_preds), axis=0)

x4_oof = np.load(f'{probs_pth}/oof_age_m4_keras')
x4_preds = np.load(f'{probs_pth}/sub_age_m4_keras')
x4 = np.concatenate((x4_oof, x4_preds), axis=0)

x5_oof = np.load(f'{probs_pth}/oof_age_m5_keras')
x5_preds = np.load(f'{probs_pth}/sub_age_m5_keras')
x5 = np.concatenate((x5_oof, x5_preds), axis=0)

x6_oof = np.load(f'{probs_pth}/oof_age_m6_keras')
x6_preds = np.load(f'{probs_pth}/sub_age_m6_keras')
x6 = np.concatenate((x6_oof, x6_preds), axis=0)

x7_oof = softmax(np.load(f'{probs_pth}/oof_age_m7_torch'), axis=1)
x7_preds = softmax(np.load(f'{probs_pth}/sub_age_m7_torch'), axis=1)
x7 = np.concatenate((x7_oof, x7_preds), axis=0)

x8_oof = np.load(f'{probs_pth}/oof_age_m8_keras')
x8_preds = np.load(f'{probs_pth}/sub_age_m8_keras')
x8 = np.concatenate((x8_oof, x8_preds), axis=0)

x9_oof = softmax(np.load(f'{probs_pth}/oof_age_m9_torch'), axis=1)
x9_preds = softmax(np.load(f'{probs_pth}/sub_age_m9_torch'), axis=1)
x9 = np.concatenate((x9_oof, x9_preds), axis=0)

x10_oof = np.load(f'{probs_pth}/oof_age_m10_keras')
x10_preds = np.load(f'{probs_pth}/sub_age_m10_keras')
x10 = np.concatenate((x10_oof, x10_preds), axis=0)

x11_oof = np.load(f'{probs_pth}/oof_age_m11_keras')
x11_preds = np.load(f'{probs_pth}/sub_age_m11_keras')
x11 = np.concatenate((x11_oof, x11_preds), axis=0)


lgb_oof = pd.read_csv(f'{probs_pth}/oof_age_lgb.csv')[[f'age_prob_{i}' for i in range(1,11)]].values
lgb_preds = pd.read_csv('{probs_pth}/sub_age_lgb.csv')[[f'age_prob_{i}' for i in range(1,11)]].values
lgb_probs = np.concatenate((lgb_oof, lgb_preds), axis=0)

x_stacking = np.concatenate((lgb_probs, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11), axis=1)

np.save(f"{probs_pth}/x_stacking_120probs", x_stacking)
