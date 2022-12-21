'''

usage
```
$ python get_lvl4_APC.py --data ../data/ --out ../data/APC/
```

This script computes the Average Pearson Correlation (APC) value for each level 4 sample.

     x2
   /
x1 - x3
   \ 
     x4

Given 4 bio-replicates (x1-4), we can compute the APC value for x1 by the average of it's pearson correlation with x2,x3,x4. 
'''

import numpy as np
import pandas as pd 
import argparse
import os 
import h5py
import torch

def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str,
                        help="path to the data dir")

    parser.add_argument("--out", type=str,
                        help="output dir; will save as rAPC.csv")

    return parser.parse_args()


if __name__ == '__main__': 

    args = get_args()

    # get sample space
    instinfo = pd.read_csv(f'{args.data}/instinfo_beta.txt', sep='\t', low_memory=False)[lambda x: (x.pert_type == 'trt_cp') & (x.qc_pass == 1.)]
    sampspace = instinfo.sample_id.unique().astype(str)

    # load L1000 data 
    hdf = h5py.File(f'{args.data}/level4_beta_trt_cp_n1805898x12328.gctx')
    sample_ids = hdf['0']['META']['COL']['id'][...].astype(str)
    gene_ids = hdf['0']['META']['ROW']['id'][...].astype(int)
    
    # select landmark genes only 
    geneinfo = pd.read_csv(f'{args.data}/geneinfo_beta.txt', sep='\t')
    landmark_gene_ids = geneinfo[lambda x: x.feature_space == 'landmark'].gene_id.values.astype(int)
    landmark_hdf_ixs = np.isin(gene_ids, landmark_gene_ids).nonzero()[0]
    assert (np.sort(gene_ids[landmark_hdf_ixs]) == np.sort(landmark_gene_ids)).all(), 'landmark gene selection failed.'
    sample_hdf_ixs = np.isin(sample_ids, sampspace).nonzero()[0]
    assert (np.sort(sample_ids[sample_hdf_ixs]) == np.sort(sampspace)).all(), 'hdf sample id selection failed'   

    # load all data into memory 
    sid2mat = {}
    for i,(six, sid) in enumerate(zip(sample_hdf_ixs, sample_ids[sample_hdf_ixs])): 
        if i%100==0: print(f'making sid2mat: {i}/{len(sample_hdf_ixs)}', end='\r')
        sid2mat[sid] = hdf['0']['DATA']['0']['matrix'][six, ...][landmark_hdf_ixs].astype(float)

    ## Compute APC 
    biorepl = instinfo.groupby(['pert_id', 'cell_iname', 'pert_dose', 'pert_time'])[['sample_id']].agg(lambda x: list(x))
    biorepl = biorepl.assign(nrepl = lambda x: [len(xx) for xx in x.sample_id])

    res = {'sample_id':[], 'rAPC':[], 'nrepl':[], 'L1':[], 'L2':[]}

    for i,row in biorepl.reset_index().iterrows(): 
        print(f'computing level 4 rAPC, progress: {i}/{len(biorepl)}', end='\r')

        if row.nrepl > 1: 
            
            for sid in row.sample_id: 
                other_ids = list(row.sample_id)
                other_ids.remove(sid)
                APC = np.mean([np.corrcoef(sid2mat[sid], sid2mat[oid])[0,1] for oid in other_ids])
                res['sample_id'].append(sid)
                res['rAPC'].append(APC)
                res['nrepl'].append(len(other_ids) + 1)
                res['L1'].append(np.mean(np.abs(sid2mat[sid])))
                res['L2'].append(np.mean(sid2mat[sid]**2))

        else:
            assert len(row.sample_id) == 1, f'expected # of replicates to be 1, got: {row.nrepl} | {row.sample_id}'
            res['sample_id'].append(row.sample_id[0])
            res['rAPC'].append(None)
            res['nrepl'].append(1)
            res['L1'].append(None)
            res['L2'].append(None)

    res = pd.DataFrame(res)

    res.to_csv(f'{args.out}/rAPC.csv', index=False)