'''
Supervised LINCS dataset
'''
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np

class LINCSDataset(Dataset):
    def __init__(self, y, sample_ids, pertspace, cellspace, data_dir, verbose=False):
        '''
        '''
        if torch.is_tensor(y): 
            self.y = y
        else: 
            self.y = torch.tensor(y, dtype=torch.float32)

        self.pertspace = pertspace
        self.cellspace = cellspace

        self.sample_ids = sample_ids 
        
        instinfo = pd.read_csv(data_dir + '/instinfo_beta.txt', low_memory=False, sep='\t')
        instinfo = instinfo[['sample_id', 'pert_id', 'cell_iname', 'pert_dose', 'pert_time']]
        instinfo = pd.DataFrame({'sample_id':sample_ids, '_':True}).merge(instinfo, on='sample_id', how='left') # sort instinfo by sample_id to match `sample_ids`

        pert2idx = {x:i for i,x in enumerate(self.pertspace)}
        cell2idx = {x:i for i,x in enumerate(self.cellspace)}

        if verbose: print('converting pert to idx...')
        pert_idx = torch.tensor([pert2idx[x] for x in instinfo.pert_id.values], dtype=torch.long)
        if verbose: print('converting pert to idx...')
        cell_idx = torch.tensor([cell2idx[x] for x in instinfo.cell_iname.values], dtype=torch.long)
        if verbose: print('log conc transformation...')
        log_conc = torch.tensor(np.log10(instinfo.pert_dose.fillna(0) + 1), dtype=torch.float32)
        if verbose: print('time transformation...')
        z_time = torch.tensor(np.clip(instinfo.pert_time, 0, 1000), dtype=torch.float32)
        z_time = (z_time - z_time.mean())/z_time.std()

        assert ~torch.isnan(pert_idx).any(), '`pert_idx` contains nan'
        assert ~torch.isnan(cell_idx).any(), '`cell_idx` contains nan'
        assert ~torch.isnan(log_conc).any(), '`log_conc` contains nan'
        assert ~torch.isnan(z_time).any(), '`z_time` contains nan'

        assert ~torch.isinf(pert_idx).any(), '`pert_idx` contains inf'
        assert ~torch.isinf(cell_idx).any(), '`cell_idx` contains inf'
        assert ~torch.isinf(log_conc).any(), '`log_conc` contains inf'
        assert ~torch.isinf(z_time).any(), '`z_time` contains inf'

        self.pert_idx = pert_idx
        self.cell_idx = cell_idx
        self.log_conc = log_conc 
        self.z_time = z_time
        self.instinfo = instinfo

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, idx):
        y = self.y[idx, :]
        pert_idx = self.pert_idx[idx]
        cell_idx = self.cell_idx[idx]
        z_time = self.z_time[idx]
        log_conc = self.log_conc[idx]

        return (pert_idx, cell_idx, z_time, log_conc), y