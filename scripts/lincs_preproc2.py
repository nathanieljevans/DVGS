'''


use: 

```
(dvgs) $
```
'''

import pandas as pd 
import argparse
import h5py 
import numpy as np
import torch 
import os 
import copy 

_NUM_TARGETS_TO_INCLUDE = 1000


def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str,
                        help="path to the data dir")

    parser.add_argument("--out", type=str,
                        help="output dir")


    return parser.parse_args()


if __name__ == '__main__': 

    args = get_args()

    osig_orig = pd.read_csv(f'{args.data}/processed/ordered_siginfo.tsv', sep='\t', low_memory=False)

    osig = copy.deepcopy(osig_orig)
    osig = osig[['sig_id', 'pert_id', 'cell_iname', 'pert_time', 'pert_dose']].dropna()
    osig = osig[lambda x: x.pert_time > 0] # drop pert_time = -666

    drugspace = np.sort(osig.pert_id.unique())
    cellspace = np.sort(osig.cell_iname.unique())

    drug2idx = {d:i for i,d in enumerate(drugspace)}
    cell2idx = {c:i for i,c in enumerate(cellspace)}

    #print('# drugs:', len(drug2idx))
    #print('# cells:', len(cell2idx))

    pert_idx = osig.pert_id.apply(lambda x: drug2idx[x])
    cell_idx = osig.cell_iname.apply(lambda x: cell2idx[x])
    log_conc = osig.pert_dose.apply(lambda x: np.log10(x))

    _time = np.clip(osig.pert_time.values, 0, 72) 
    z_time = (_time - _time.mean()) / _time.std()

    out = pd.DataFrame({'sig_id':osig.sig_id.values, 'pert_idx':pert_idx, 'cell_idx':cell_idx,'log_conc':log_conc,'z_time':z_time})

    out.to_csv(f'{args.out}/osig_indexed.tsv', sep='\t', index=False)

    # now DTI data 
    druginfo = pd.read_csv(f'{args.data}/compoundinfo_beta.txt', sep='\t', low_memory=False)
    druginfo = druginfo[lambda x: x.pert_id.isin(drugspace)]

    targspace = druginfo.groupby('target').count()[['pert_id']].sort_values('pert_id', ascending=False)
    targspace = targspace[lambda x: x.pert_id >= 2]
    targspace = targspace.head(_NUM_TARGETS_TO_INCLUDE).index.values.astype(str)

    dinfo = druginfo.groupby('pert_id')[['target']].agg(lambda x: list(x)).reset_index()
    dinfo = dinfo.assign(pert_idx=lambda x: x.pert_id.apply(lambda x: drug2idx[x]))

    for t in targspace: 
        dinfo = dinfo.assign(**{t:lambda x: [t in xx for xx in x.target]})

    dinfo.to_csv(f'{args.out}/pert2targets.tsv', sep='\t', index=False)