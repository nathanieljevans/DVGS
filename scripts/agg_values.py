'''

'''

import argparse 
import uuid 
import torch 
from sklearn.utils.class_weight import compute_class_weight
import time
import numpy as np
import pandas as pd
import hashlib
import os
import copy 
import pickle as pkl

import sys 
sys.path.append('../src/')
from utils import load_data, load_config, Logger, get_filtered_scores
from DVGS import DVGS
from DVRL import DVRL
from DShap import DShap
from LOO import LOO
from DVRL2 import DVRL2


def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", type=str,
                        help="data values directory")
    
    parser.add_argument("--method", type=str,
                        help="method to accumulate")

    parser.add_argument("--out", type=str,
                        help="output file")

    return parser.parse_args()


if __name__ == '__main__': 

    args = get_args()

    osig = pd.read_csv('../data/processed/ordered_siginfo.tsv', sep='\t', low_memory=False)
    res = osig[['sig_id']]
    sids = osig.sig_id.values

    dv_res = pd.read_csv(args.dir + '/results.csv', sep='\t')
    method_uids = dv_res[lambda x: x.method == args.method].uid.values

    root = args.dir + '/data_values/'
    for uid in os.listdir(root): 

        if uid in method_uids: 
            vals = np.load(f'{root}/{uid}/data_values.npy')
            kwargs = np.load(f'{root}/{uid}/kwarg_dict.pkl', allow_pickle=True)
            train_idx = kwargs['idx_train']
            _sids = sids[train_idx]
            res = res.merge(pd.DataFrame({'sig_id':_sids, uid:vals}), on='sig_id', how='left')
    
    v = res.values[:, 1:].astype(float)
    v[np.all(np.isnan(v), axis=1)] = -1.01
    vals = np.nanmean(v, axis=1)

    res = pd.DataFrame({'sig_id':sids, 'value':vals})

    res.to_csv(args.out, index=False)


        