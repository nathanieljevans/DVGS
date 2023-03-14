'''

$ python run_supervised_lincs_embedding_analysis.py --vals_path ../results/exp10/dvgs_vals.csv --out ../results/exp10/dvgs_dti_results.pkl

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
import h5py
from sklearn.metrics import r2_score

import sys 
sys.path.append('../src/')
from utils import load_data, load_config, Logger, get_filtered_scores
from DVGS import DVGS
from DVRL import DVRL
from DShap import DShap
from LOO import LOO
from DVRL2 import DVRL2
from LincsEmbNN import LincsEmbNN


### CONFIG PARAMS ### 

MODEL_ARGS =   {'cell_channels'    : 256, 
                'pert_channels'    : 256, 
                'out_channels'     : 978, 
                'num_layers'       : 2, 
                'hidden_channels'  : 200, 
                'norm'             : True, 
                'dropout'          : 0.05, 
                'act'              : torch.nn.Mish}


TRAIN_NUM = 100000

BATCH_SIZE  = 1024
EPOCHS      = 100
LR          = 1e-3

CRIT = torch.nn.MSELoss() 
OPTIMIZER = torch.optim.Adam
WD = 0

QS = np.linspace(0, 0.9, 9)

NUM_NEG_DTI_PAIRS = int(1e6)

COS_SIM_BATCH_SIZE = 1024 
NORMALIZE_EMBEDDING = True

DEVICE = 'cpu'

NUM_REPL = 5

FILL_NA_VAL = -666

######################


def train_model(model, pert_idx, cell_idx, z_time, log_conc, y):

    device = DEVICE
    model.to(device).train()

    optim = OPTIMIZER(model.parameters(), lr=LR, weight_decay=WD)

    for epoch in range(EPOCHS): 
        _losses = []
        ii=0
        _r2s = []
        for batch_idx in torch.split(torch.randperm(y.size(0)), BATCH_SIZE): 

            pert_batch = pert_idx[batch_idx].to(device)
            cell_batch = cell_idx[batch_idx].to(device)
            time_batch = z_time[batch_idx].to(device)
            conc_batch = log_conc[batch_idx].to(device)

            y_batch = y[batch_idx, :].to(device)
            yhat = model(pert_idx=pert_batch, cell_idx=cell_batch, z_time=time_batch, log_conc=conc_batch)

            optim.zero_grad()
            loss = CRIT(yhat, y_batch)
            loss.backward() 
            optim.step()
            
            _losses.append(loss.item())
            _r2s.append(r2_score(y_batch.detach().cpu().numpy(), yhat.detach().cpu().numpy(), multioutput='uniform_average'))
            print(f'[{ii}/{1+int(y.size(0)/BATCH_SIZE)}]', end='\r')
            ii+=1

        print(f'epoch: {epoch} || avg. loss: {np.mean(_losses):.4f} || avg. R2: {np.mean(_r2s):.4f}', end='\r')
    print()
    return model

def _get_pert_idx_pairs_with_shared_targets(pert2targ): 

    pos_class_idx1 = []
    pos_class_idx2 = []

    targspace = pert2targ.columns[3:]

    for i,t in enumerate(targspace): 
        print(f'progress: {i}/{len(targspace)}', end='\r')
        perts_sharing_targ = pert2targ[t].values.nonzero()[0]
        for j,p1 in enumerate(perts_sharing_targ): 
            for p2 in perts_sharing_targ[(j+1):]: 
                pos_class_idx1.append(p1)
                pos_class_idx2.append(p2)

    pos_class_idx1 = torch.tensor(pos_class_idx1, dtype=torch.long)
    pos_class_idx2 = torch.tensor(pos_class_idx2, dtype=torch.long)

    return pos_class_idx1, pos_class_idx2

def get_dti_shared_target_pairs(): 

    pert2targ = pd.read_csv('../data/processed/pert2targets.tsv', sep='\t')

    pos_class_idx1, pos_class_idx2 = _get_pert_idx_pairs_with_shared_targets(pert2targ)

    neg_class_idx1 = torch.randint(osig_idx.pert_idx.max() + 1, size=(NUM_NEG_DTI_PAIRS,))  # grand majority of combinations are negative class ; random pairings should estimate neg
    neg_class_idx2 = torch.randint(osig_idx.pert_idx.max() + 1, size=(NUM_NEG_DTI_PAIRS,))

    return (pos_class_idx1, pos_class_idx2), (neg_class_idx1, neg_class_idx2)


def batched_cosine_similarity(embedding, idx1, idx2, batch_size=1024, normalize=True): 
    ''''''

    if normalize: 
        embedding = copy.deepcopy(embedding)
        embedding.weight.data = embedding.weight.data - embedding.weight.data.mean(dim=0)
        embedding.weight.data = embedding.weight.data / embedding.weight.data.std(dim=0)

    cos_sim = torch.nn.CosineSimilarity(dim=1)
    s = []
    for batch_idx in torch.split(torch.arange(len(idx1)), batch_size):
        with torch.no_grad(): 
            z1 = embedding(idx1[batch_idx])
            z2 = embedding(idx2[batch_idx])
            s.append( cos_sim(z1, z2) )

    return torch.cat(s, dim=-1).mean()


def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--vals_path", type=str,
                        help="path to data values (should be .csv file where the first column is sig_id and the second column is the data value")

    parser.add_argument("--out", type=str,
                        help="output path; should be .pkl file")

    return parser.parse_args()

if __name__ == '__main__': 

    args = get_args()

    # LOAD DATA  
    if TRAIN_NUM is not None: 
        osig_idx = pd.read_csv('../data/processed/osig_indexed.tsv', sep='\t').dropna().sample(TRAIN_NUM, replace=False, axis=0)
    else: 
        osig_idx = pd.read_csv('../data/processed/osig_indexed.tsv', sep='\t').dropna()

    _idxs = np.sort(osig_idx.index.values)

    f = h5py.File('../data/processed/lincs.h5')
    y = torch.tensor(f['data'][_idxs, ...], dtype=torch.float32)

    pert_idx = torch.tensor(osig_idx.pert_idx.values, dtype=torch.long)
    cell_idx = torch.tensor(osig_idx.cell_idx.values, dtype=torch.long)
    log_conc = torch.tensor(osig_idx.log_conc.values, dtype=torch.float32)
    z_time = torch.tensor(osig_idx.z_time.values, dtype=torch.float32)

    sig_ids = osig_idx.sig_id.values

    model = LincsEmbNN(num_lines=osig_idx.cell_idx.max() + 1, 
                       num_perts=osig_idx.pert_idx.max() + 1,
                       **MODEL_ARGS)
    
    vals_csv = pd.read_csv(args.vals_path)
    vals_csv = vals_csv.rename({vals_csv.columns[0]:'sig_id', vals_csv.columns[1]:'value'}, axis=1)
    vals_csv = pd.DataFrame({'sig_id':sig_ids}).merge(vals_csv, on='sig_id', how='left') # vals_csv should now match the order of `sig_ids` 
    vals_csv = vals_csv.fillna(FILL_NA_VAL)
    vals = vals_csv.value.values

    print(vals_csv.head())

    pos_pairs, neg_pairs = get_dti_shared_target_pairs()

    res = {'q':[], 'low':{'pos_sim':[], 'neg_sim':[]}, 'high':{'pos_sim':[], 'neg_sim':[]}}

    for i,q in enumerate(QS): 
        print(f'training filtered models... progress: {i}/{len(QS)}')
        for j in range(NUM_REPL):
            res['q'].append(q)

            for remove in ['low', 'high']: 

                if remove == 'low':
                    t = np.quantile(vals, q)
                    include_idxs = (vals >= t).nonzero()[0] 
                else: 
                    t = np.quantile(vals, (1-q))
                    include_idxs = (vals <= t).nonzero()[0]

                trained_model = train_model(copy.deepcopy(model), 
                                            pert_idx[include_idxs], 
                                            cell_idx[include_idxs], 
                                            z_time[include_idxs], 
                                            log_conc[include_idxs], 
                                            y[include_idxs])
                
                z = trained_model.pert_embedding.cpu()

                with torch.no_grad(): 
                    pos_sim = batched_cosine_similarity(copy.deepcopy(z), *pos_pairs, batch_size=COS_SIM_BATCH_SIZE, normalize=NORMALIZE_EMBEDDING)
                    neg_sim = batched_cosine_similarity(copy.deepcopy(z), *neg_pairs, batch_size=COS_SIM_BATCH_SIZE, normalize=NORMALIZE_EMBEDDING)

                res[remove]['pos_sim'].append(pos_sim)
                res[remove]['neg_sim'].append(neg_sim)

                print(f'pos sim: {pos_sim:.4f} || neg sim: {neg_sim:.4f}')

    with open(args.out, 'wb') as ff: 
        pkl.dump(res, ff)

    


    
    