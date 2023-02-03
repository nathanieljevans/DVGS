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

    parser.add_argument("--out", type=str,
                        help="output dir; will save as rAPC.csv")

    parser.add_argument("--prop_test", type=float,
                        help="valid:test split proportion (recommended: 0.25)")

    parser.add_argument("--thresh_rAPC", type=float,
                        help="rAPC threshold to separate source:target splits (recommended: 0.7)")

    parser.add_argument("--seed", type=int,
                        help="rng seed")

    return parser.parse_args()


if __name__ == '__main__': 

    args = get_args()

    print('loading metadata...')
    instinfo = pd.read_csv(f'{args.data}/instinfo_beta.txt', sep='\t', low_memory=False)[lambda x: (x.pert_type=='trt_cp') & (x.qc_pass == 1.)]
    rAPC = pd.read_csv(f'{args.data}/rAPC.csv')
    hdf = h5py.File(f'{args.data}/level4_beta_trt_cp_n1805898x12328.gctx')
    geneinfo = pd.read_csv(f'{args.data}/geneinfo_beta.txt', sep='\t')

    sample_ids = hdf['0']['META']['COL']['id'][...].astype(str)
    gene_ids = hdf['0']['META']['ROW']['id'][...].astype(int)

    landmark_gene_ids = geneinfo[lambda x: x.feature_space == 'landmark'].gene_id.values.astype(int)
    landmark_hdf_ixs = np.isin(gene_ids, landmark_gene_ids).nonzero()[0]
    assert (np.sort(gene_ids[landmark_hdf_ixs]) == np.sort(landmark_gene_ids)).all(), 'landmark gene selection failed.'
    landmark_gene_ids = gene_ids[landmark_hdf_ixs]

    # about 7 GB
    print('loading data...')
    data = hdf['0']['DATA']['0']['matrix'][..., landmark_hdf_ixs].astype(np.float32)

    assert not np.isnan(data).any(), 'LINCS expression data contains NAN'
    assert not np.isinf(data).any(), 'LINCS expression data contains INF'

    data = torch.tensor(data, dtype=torch.float32)

    print('splitting data...')
    apc = rAPC.fillna(-666)
    apc = pd.DataFrame({'sample_id':sample_ids}).merge(apc, on='sample_id', how='left') # order identically to hdf data matrix 
    apc = apc.assign(train=lambda x: x.rAPC < args.thresh_rAPC)
    apc = apc.assign(valid=lambda x: x.rAPC >= args.thresh_rAPC)

    train_idx = (apc.train.values * 1.).nonzero()[0]
    valid_idx = (apc.valid.values * 1.).nonzero()[0]
    np.random.seed(args.seed)
    np.random.shuffle(valid_idx)
    test_idx = valid_idx[:int(len(valid_idx)*args.prop_test)]
    valid_idx = valid_idx[int(len(valid_idx)*args.prop_test):]

    print('train (source) size:', len(train_idx))
    print('valid (target) size:', len(valid_idx))
    print('test size:', len(test_idx))

    x_train = data[train_idx, :]
    x_valid = data[valid_idx, :]
    x_test = data[test_idx, :]

    train_ids = apc.sample_id.values[train_idx]
    valid_ids = apc.sample_id.values[valid_idx]
    test_ids = apc.sample_id.values[test_idx]

    assert len(train_ids) == x_train.shape[0], f'train id length does not match train dataset size: expected {x_train.shape[0]}, got: {len(train_ids)}'
    assert len(valid_ids) == x_valid.shape[0], f'valid id length does not match train dataset size: expected {x_valid.shape[0]}, got: {len(valid_ids)}'
    assert len(test_ids) == x_test.shape[0], f'test id length does not match train dataset size: expected {x_test.shape[0]}, got: {len(test_ids)}'

    print('saving to disk...')
    if not os.path.exists(args.out): os.mkdir(args.out)

    torch.save(data, f'{args.out}/l1000_data_ALL.pt')
    torch.save(x_train, f'{args.out}/l1000_data_TRAIN.pt')
    torch.save(x_valid, f'{args.out}/l1000_data_VALID.pt')
    torch.save(x_test, f'{args.out}/l1000_data_TEST.pt')

    np.save(f'{args.out}/l1000_ids_ALL.npy', apc.sample_id.values)
    np.save(f'{args.out}/l1000_ids_TRAIN.npy', train_ids)
    np.save(f'{args.out}/l1000_ids_VALID.npy', valid_ids)
    np.save(f'{args.out}/l1000_ids_TEST.npy', test_ids)

    np.save(f'{args.out}/l1000_GENE_IDs.npy', landmark_gene_ids)

    print(f'lincs preprocessing complete...saved to: {args.out}')