'''

usage
```
$ python calc_APC.py --data ../data/ --out ../data/APC/
```

This script computes the Average Pearson Correlation (APC) value for each level bio-replicate in the level 4 compound dataset. 

     x2
   / | 
x1 - x3
   \ | 
     x4

Given 4 bio-replicates (x1-4), we can compute the APC value by the average of replicate pairwise correlations.
'''

import numpy as np
import pandas as pd 
import argparse
import os 
import h5py
import torch
import time 

#################
# configs 
#################
# the maximum number of bio-replicates to include in the APC calculation, if there are more than this # then some will be omitted from calculating APC
_MAX_NUM_REPL = 250
#################

def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str,
                        help="path to the data dir")

    parser.add_argument("--out", type=str,
                        help="output dir; will save as rAPC.csv")

    return parser.parse_args()


if __name__ == '__main__': 

    args = get_args()

    # siginfo for level 5 for distil ids 
    siginfo = pd.read_csv(f'{args.data}/siginfo_beta.txt', sep='\t', low_memory=False)[lambda x: (x.pert_type=='trt_cp') & (x.qc_pass == 1.)]
    siginfo = siginfo.reset_index() 

    # load L1000 data 
    hdf = h5py.File(f'{args.data}/level4_beta_trt_cp_n1805898x12328.gctx')
    sample_ids = hdf['0']['META']['COL']['id'][...].astype(str)
    gene_ids = hdf['0']['META']['ROW']['id'][...].astype(int)
    
    # select landmark genes only 
    geneinfo = pd.read_csv(f'{args.data}/geneinfo_beta.txt', sep='\t')
    landmark_gene_ids = geneinfo[lambda x: x.feature_space == 'landmark'].gene_id.values.astype(int)
    landmark_hdf_ixs = np.isin(gene_ids, landmark_gene_ids).nonzero()[0]
    assert (np.sort(gene_ids[landmark_hdf_ixs]) == np.sort(landmark_gene_ids)).all(), 'landmark gene selection failed.'
    
    ## sid2idx 
    ## speeds up indexing by 1e6x 
    sid2idx = {sid:ii for ii,sid in enumerate(sample_ids)}

    ## Compute APC 
    res = {'sig_id':[], 'APC':[]}
    fails = []
    tic = time.time()
    for i, row in siginfo.iterrows(): 
        
        if i % 100 == 0:
            time_per_sample = (time.time()-tic)/(i+ 1)
            samples_to_go = len(siginfo) - (i + 1)
            est_time_remain = samples_to_go*time_per_sample/60 # min
            print(f'calculating APC values... progress: {i}/{len(siginfo)} [est. time remaining: {est_time_remain:.2f} min]', end='\r')
            
        # lincs level 4 sample_ids 
        sids = np.array(row.distil_ids.split('|')).astype(str)

        if len(sids) > 1: 
            
            try: 
                # get sids index 
                #sidx = np.isin(sample_ids, sids).nonzero()[0]
                sidx = [sid2idx[sid] for sid in sids]

                # load level 4 data into memory 
                xx = hdf['0']['DATA']['0']['matrix'][sidx, ...][:, landmark_hdf_ixs].astype(float)

                # compute correlation
                Rmat = np.corrcoef(xx)
                assert Rmat.shape[0] == len(sids), f'expected Rmat dim to be same as sids but got: {Rmat.shape[0]} =\= {len(sids)}'
                APC = np.mean(Rmat[np.triu_indices(Rmat.shape[0], k=1)])

                res['sig_id'].append(row.sig_id)
                res['APC'].append(APC)
            
            except:
                fails.append(row.sig_id)

    print()
    print('# failed:', len(fails))

    res = pd.DataFrame(res)
    res.to_csv(f'{args.out}/APC.csv', index=False)