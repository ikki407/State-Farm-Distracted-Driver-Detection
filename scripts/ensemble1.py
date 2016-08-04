import pandas as pd
import numpy as np

def averaging(pred_list, submission_name=''):

    cols = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

    pred_ens = pd.DataFrame(np.zeros(79726*10).reshape(79726,10), columns=cols)
    for i in pred_list:
        a = pd.read_csv('submission/' + i)
        pred_ens[cols] += a[cols]

    pred_ens = pred_ens / len(pred_list)
    pred_ens['img'] = a['img'].values
    pred_ens.to_csv('submission/' + submission_name + '.csv', index=False)

def weighted_averaging(pred_list, weights=[], submission_name=''):

    cols = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

    pred_ens = pd.DataFrame(np.zeros(79726*10).reshape(79726,10), columns=cols)
    for w, i in enumerate(pred_list):
        a = pd.read_csv('submission/' + i)
        pred_ens[cols] += a[cols] * weights[w]

    pred_ens['img'] = a['img'].values
    pred_ens.to_csv('submission/' + submission_name + '.csv', index=False)


pred_lists = [
    'ensemble_Model1_VGG19_all.csv',
    'ensemble_Model2_VGG19_all.csv',
    'ensemble_Model3_VGG19_all.csv',
    'ensemble_Model4_VGG19_all.csv',
    'ensemble_Model5_VGG19_all.csv',
]

averaging(pred_lists, 'ensemble1_5')
