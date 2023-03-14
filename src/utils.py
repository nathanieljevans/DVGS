import h5py
import numpy as np
import copy
import torch
from data_loading import load_tabular_data, preprocess_data
import shutil 
import torchvision 
from torchvision import transforms
from data_loading import corrupt_label
import importlib.util
import sys
from pathlib import Path
import pandas as pd 
import os

class Logger(object):
    '''(Amith Koujalgi) https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting'''
    def __init__(self, path):
        self.terminal = sys.stdout
        Path(path).mkdir(parents=True, exist_ok=True)
        self.log = open(path + '/console.log', "a")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass   

def load_config(path): 
    '''(Sebastian Rittau) https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path'''

    spec = importlib.util.spec_from_file_location("my.config", path)
    config = importlib.util.module_from_spec(spec)
    sys.modules["my.config"] = config
    spec.loader.exec_module(config)
    return config

def load_data(dataset, train_num, valid_num, exog_noise=0., endog_noise=0., save_dir='./temp-data/', clean_up=True, lincs_dir='../data/processed/', transforms=None): 
    
    # spot to stick extra stuff
    kwargs = None

    if dataset in ['adult', 'blog']: 
        noise_idx = load_tabular_data(dataset, {'train':train_num, 'valid':valid_num}, noise_rate=endog_noise, out=save_dir) 

        x_train, y_train, x_valid, y_valid, x_test, y_test, col_names = preprocess_data(normalization='minmax', train_file_name='train.csv', valid_file_name='valid.csv', test_file_name='test.csv', data=save_dir)

        if clean_up: 
            shutil.rmtree(save_dir)
        
        x_train       = torch.tensor(x_train, dtype=torch.float32)
        y_train       = torch.tensor(y_train, dtype=torch.long).view(-1,1)

        x_valid       = torch.tensor(x_valid, dtype=torch.float32)
        y_valid       = torch.tensor(y_valid, dtype=torch.long).view(-1,1)

        x_test       = torch.tensor(x_test, dtype=torch.float32)
        y_test       = torch.tensor(y_test, dtype=torch.long).view(-1,1)

        endog_noise = np.zeros(x_train.size(0))
        endog_noise[noise_idx] = 1


    elif dataset in ['cifar10', 'cifar10-unsupervised']: 
        transform = transforms

        train_dataset = torchvision.datasets.CIFAR10(root=save_dir, train=True,
                                                download=True, transform=transform)

        test_dataset = torchvision.datasets.CIFAR10(root=save_dir, train=False,
                                            download=True, transform=transform)

        # split train -> train/valid 
        assert train_num + valid_num <= 50000, 'only 50k obs'
        _idxs = torch.randperm(50000)
        train_idx = _idxs[:train_num]
        valid_idx = _idxs[train_num:(train_num+valid_num)]

        test_idx = torch.randperm(len(test_dataset))[:valid_num]

        def _read_from_dataset(dataset, idx): 
            x = []; y = []
            for j,i in enumerate(idx): 
                print(f'reading cifar10 into memory... {j/len(idx)*100:.0f}%', end='\r')
                (xx,yy) = dataset.__getitem__(i) 
                x.append(xx); y.append(yy)
            print()
            x = torch.stack(x, dim=0)
            y = torch.tensor(y, dtype=torch.long).view(-1,1)
            return x,y

        x_train, y_train = _read_from_dataset(train_dataset, train_idx)
        x_valid, y_valid = _read_from_dataset(train_dataset, valid_idx)
        x_test, y_test = _read_from_dataset(test_dataset, test_idx)

        # clean-up disk 
        if clean_up: 
            shutil.rmtree(save_dir)

        # corrupt endog 
        y_train, noise_idx = corrupt_label(y_train.detach().numpy().ravel(), noise_rate=endog_noise) 
        y_train = torch.tensor(y_train, dtype=torch.long).view(-1,1)
        endog_noise = np.zeros(x_train.size(0))
        endog_noise[noise_idx] = 1. 

        if dataset == 'cifar10-unsupervised': 
            y_train = x_train 
            y_valid = x_valid 
            y_test = x_test 

    elif dataset in ['lincs', 'lincs-hi-apc-target', 'lincs-hi-apc']: 
        
        assert os.path.exists(lincs_dir), 'have you run `./src/lincs_setup.sh`?'

        # hi-apc-threshold 
        _APC_THRESHOLD = 0.5
        
        # load ordered instinfo into memory 
        osig = pd.read_csv(f'{lincs_dir}/ordered_siginfo.tsv', sep='\t', low_memory=False)

        # get high apc indices 
        hi_apc_idx = (osig.APC.values > _APC_THRESHOLD).nonzero()[0]
        print('total # samples in high APC subset:', len(hi_apc_idx))

        if dataset == 'lincs-hi-apc': 
            assert train_num + valid_num <= hi_apc_idx.shape[0], f'only {hi_apc_idx.shape[0]} obs'

            # shuffle 
            hi_apc_idx = np.random.permutation(hi_apc_idx)

            train_idx = hi_apc_idx[:train_num]
            valid_idx = hi_apc_idx[train_num:int(train_num+valid_num)]
            test_idx = hi_apc_idx[int(train_num+valid_num):] 

            target_indices = hi_apc_idx

        elif dataset in ['lincs-hi-apc-target', 'lincs']: 

            if dataset == 'lincs-hi-apc-target':
                # NOTE: target + test will be balanced by pert, and selected from high apc samples
                # NOTE: test set will be of same size as target; intended to best match target-test
                # train will be selected from all other samples 
                _perts = osig.pert_id.values[hi_apc_idx].astype(str)

                target_indices = hi_apc_idx

            else: 
                # dataset == lincs 
                # NOTE: target + test will be balanced by pert, but not selected from high apc samples
                # NOTE: test will be same size as valid 
                _perts = osig.pert_id.values.astype(str)

                target_indices = np.arange(len(osig))

            assert len(_perts) > valid_num*2, f'maximum appropriate `valid_num` is {int(len(_perts)/2)}'

            unq_pids, pid_cnts = np.unique(_perts, return_counts=True)
            pid_weight = {pid: len(_perts)/(len(unq_pids)*cnts) for pid, cnts in zip(unq_pids, pid_cnts)}

            sel_probs = np.array([pid_weight[pid] for pid in _perts])
            sel_probs /= sel_probs.sum()
            
            _idxs = np.random.choice(target_indices, p=sel_probs, size=(int(2*valid_num),), replace=False)

            valid_idx = _idxs[:valid_num]
            test_idx = _idxs[valid_num:]

            if train_num == -1: 
                train_idx = np.random.permutation(np.delete(np.arange(len(osig)), _idxs))[:]
            else:
                train_idx = np.random.permutation(np.delete(np.arange(len(osig)), _idxs))[:train_num]

        train_idx = np.sort(train_idx)
        valid_idx = np.sort(valid_idx)
        test_idx = np.sort(test_idx)

        hdf = h5py.File(f'{lincs_dir}/lincs.h5')
        
        x_train = torch.tensor(hdf['data'][train_idx, :], dtype=torch.float)
        x_valid = torch.tensor(hdf['data'][valid_idx, :], dtype=torch.float)
        x_test = torch.tensor(hdf['data'][test_idx, :], dtype=torch.float)

        # because unsupervised
        y_train = x_train
        y_valid = x_valid
        y_test = x_test

        kwargs= {'idx_train':train_idx, 'idx_valid':valid_idx, 'idx_test':test_idx}

        # for compatibility later ; kinda hacky
        endog_noise = np.zeros(x_train.size(0))

    else:
        raise Exception('unrecognized dataset name.')

    # add gaussian noise 
    if exog_noise > 0: 
        exog_noise = exog_noise*torch.rand(size=(x_train.size(0),)).view(-1, *([1]*len(x_train.size()[1:])))   # exog_noise rates, sampled from uniform dist (max: exog_noise)
        x_noise = exog_noise*torch.randn_like(x_train)
        x_train += x_noise
        exog_noise = exog_noise.detach().numpy().ravel()
    else: 
        exog_noise = None

    if endog_noise.sum() == 0: 
        endog_noise = None 


    return x_train.detach(), y_train.detach(), x_valid.detach(), y_valid.detach(), x_test.detach(), y_test.detach(), exog_noise, endog_noise, kwargs


def get_corruption_scores(vals, noise_idx, train_size, noise_prop, n_scores=500):
    '''
    '''
    ks = np.linspace(1, train_size-1, n_scores)
    p_corr = []
    p_perfect = [] 
    p_random = [] 

    for k in ks: 
        idx_dvgs = np.argpartition(vals, int(k))[:int(k)]
        p_corr.append( len(set(noise_idx).intersection(set(idx_dvgs)))/(noise_prop*train_size) ) 
        p_perfect.append(min(int(k)/(train_size*noise_prop), 1.))
        p_random.append(k/train_size)

    pk = ks/train_size
    return pk, p_corr, p_perfect, p_random


def get_filtered_scores(vals, model, crit, metric, x_train, y_train, x_test, y_test, qs=np.linspace(0., 0.5, 5), batch_size=256, lr=1e-3, epochs=200, repl=1, reset_params=True):
    ''''''

    remove_low_values = []
    remove_high_values= []
    for i,q in enumerate(qs): 
        print(f'training filtered models... progress: {i}/{len(qs)}', end='\r')
        low_t = np.quantile(vals, q)
        high_t = np.quantile(vals, (1-q))

        remove_low_mask = (vals >= low_t).nonzero()[0] 
        remove_high_mask = (vals <= high_t).nonzero()[0]
        
        _low_res = []; _high_res = []
        for j in range(repl): 

            if reset_params: 
                model.reset_parameters()
                 
            _, remove_low_res = train_model(model           = copy.deepcopy(model),
                                            crit            = crit, 
                                            metric          = metric,
                                            x_train         = x_train[remove_low_mask, :], 
                                            y_train         = y_train[remove_low_mask, :], 
                                            x_test          = x_test, 
                                            y_test          = y_test, 
                                            batch_size      = batch_size,
                                            lr              = lr,
                                            epochs          = epochs)

            _, remove_high_res = train_model(model           = copy.deepcopy(model),
                                            crit            = crit, 
                                            metric          = metric,
                                            x_train         = x_train[remove_high_mask, :], 
                                            y_train         = y_train[remove_high_mask, :], 
                                            x_test          = x_test, 
                                            y_test          = y_test, 
                                            batch_size      = batch_size,
                                            lr              = lr,
                                            epochs          = epochs)

            _low_res.append(remove_low_res)
            _high_res.append(remove_high_res)

        remove_low_values.append(np.mean(_low_res))
        remove_high_values.append(np.mean(_high_res))
    print()

    return remove_low_values, remove_high_values
        

def train_model(model, crit, metric, x_train, y_train, x_test, y_test, batch_size=256, lr=1e-3, epochs=200, verbose=False, use_cuda=True): 

    if torch.cuda.is_available() & use_cuda: 
        device = 'cuda'
    else: 
        device = 'cpu'
    if verbose: print('using device:', device)

    model = copy.deepcopy(model).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs): 
        _losses = []
        for batch_idx in torch.split(torch.randperm(x_train.size(0)), batch_size): 
            x = x_train[batch_idx, :]
            y = y_train[batch_idx, :]
            x,y = x.to(device), y.to(device)
            optim.zero_grad()
            yhat_train = model(x)
            loss = crit(yhat_train, y)
            loss.backward()
            optim.step()
            _losses.append(loss.item())
        if verbose: print(f'epoch: {epoch} | loss: {np.mean(_losses)}', end='\r')

    model = model.eval() 

    _ys=[]; _yhats=[] 
    for batch_idx in torch.split(torch.arange(x_test.size(0)), batch_size):  
        x = x_test[batch_idx, :]
        y = y_test[batch_idx, :]
        x,y = x.to(device), y.to(device)
        _ys.append(y.detach().cpu())
        _yhats.append(model(x).detach().cpu())

    y = torch.cat(_ys, dim=0).detach().cpu().numpy()
    yhat = torch.cat(_yhats, dim=0).detach().cpu().numpy()

    res = metric(y, yhat)
    return model, res




## code that tests our probability adjustment for lincs sampling 
'''
a = torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,2,2,2,2])
classes,cnts = torch.unique(a,return_counts=True)

# n_samples / (n_classes*n_samples_in_class)
w = len(a)/(len(classes)*cnts)
print(w)
ap = w[a] / w[a].sum()
print(ap)

z = np.random.choice(a.detach().numpy(), p=ap.detach().numpy(), size=(1000000,)) 

_, scnts = np.unique(z, return_counts=True)
scnts/1000000
'''