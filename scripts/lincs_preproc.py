'''


use: 

```
(dvgs) $ python lincs_preproc.py --data ../data/ --out ../data/lincs/ --prop_test 0.25 --thresh_rAPC 0.7 --seed 0
```
'''

import pandas as pd 
import argparse
import h5py 
import numpy as np
import torch 
import os 

def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str,
                        help="path to the data dir")

    parser.add_argument("--apc_dir", type=str,
                        help="path to the APC dir")

    parser.add_argument("--out", type=str,
                        help="output dir; will save as rAPC.csv")

    parser.add_argument("--zscore", action='store_true', default=False,
                        help='zscore across all data')

    return parser.parse_args()


if __name__ == '__main__': 

    args = get_args()

    print('loading metadata...')
    ## Load Data 
    siginfo = pd.read_csv(f'{args.data}/siginfo_beta.txt', sep='\t', low_memory=False)[lambda x: (x.pert_type=='trt_cp') & (x.qc_pass == 1.)]
    APC = pd.read_csv(f'{args.apc_dir}/APC.csv')
    hdf = h5py.File(f'{args.data}/level5_beta_trt_cp_n720216x12328.gctx')
    geneinfo = pd.read_csv(f'{args.data}/geneinfo_beta.txt', sep='\t')

    # get row,col values 
    sample_ids = hdf['0']['META']['COL']['id'][...].astype(str)
    gene_ids = hdf['0']['META']['ROW']['id'][...].astype(int)

    # select landmark genes only 
    landmark_gene_ids = geneinfo[lambda x: x.feature_space == 'landmark'].gene_id.values.astype(int)
    landmark_hdf_ixs = np.isin(gene_ids, landmark_gene_ids).nonzero()[0]
    assert (np.sort(gene_ids[landmark_hdf_ixs]) == np.sort(landmark_gene_ids)).all(), 'landmark gene selection failed.'
    landmark_gene_ids = gene_ids[landmark_hdf_ixs]

    # load all samples into mem 
    print('loading data...')
    data = hdf['0']['DATA']['0']['matrix'][..., landmark_hdf_ixs].astype(np.float32)

    assert not np.isnan(data).any(), 'LINCS expression data contains NAN'
    assert not np.isinf(data).any(), 'LINCS expression data contains INF'

    ## z-score data 
    if args.zscore: 
        print('!!NOTE!!: z-scoring expression data')
        _mean = data.mean(axis=0)
        _std = data.std(axis=0) 
        data = (data - _mean)/_std

    # save data to disk as hdf file (for ease of reading by idx)
    with h5py.File(f'{args.out}/lincs.h5', 'w') as fout: 
        dset = fout.create_dataset("data", data=data, dtype='float')

    # just in case we're short on mem
    del data

    # save subset of instance info to disk (sample_id order identical to data)
    siginfo = pd.DataFrame({'sig_id':sample_ids}).merge(siginfo, on='sig_id', how='left').merge(APC, on='sig_id', how='left')
    siginfo.to_csv(f'{args.out}/ordered_siginfo.tsv', sep='\t', index=False)

    print(f'lincs preprocessing complete...saved to: {args.out}')