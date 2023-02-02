
import numpy as np
import copy
import torch
from GenDataset import GenDataset
from data_loading import load_tabular_data, preprocess_data
import shutil 
import torchvision 
from torchvision import transforms
from data_loading import corrupt_label
import importlib.util
import sys
from pathlib import Path

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

def load_data(dataset, train_num, valid_num, exog_noise=0., endog_noise=0., save_dir='./temp-data/', clean_up=True): 

    if dataset in ['adult', 'blog']: 
        noise_idx = load_tabular_data(dataset, {'train':train_num, 'valid':valid_num}, noise_rate=endog_noise, out=save_dir) 
        x_train, y_train, x_valid, y_valid, x_test, y_test, col_names = preprocess_data('minmax', 'train.csv', 'valid.csv', 'test.csv', data=save_dir)

        if clean_up: 
            shutil.rmtree(save_dir)
        
        x_train       = torch.tensor(x_train, dtype=torch.float)
        y_train       = torch.tensor(y_train, dtype=torch.float).view(-1,1)

        x_valid       = torch.tensor(x_valid, dtype=torch.float)
        y_valid       = torch.tensor(y_valid, dtype=torch.float).view(-1,1)

        x_test       = torch.tensor(x_test, dtype=torch.float)
        y_test       = torch.tensor(y_test, dtype=torch.float).view(-1,1)

        # add gaussian noise 
        exog_noise = exog_noise*torch.randn_like(x_train)
        x_train += exog_noise

        endog_noise = np.zeros(x_train.shape[0])
        endog_noise[noise_idx] = 1.

        return x_train, y_train, x_valid, y_valid, x_test, y_test, exog_noise, endog_noise

    elif dataset == 'cifar10': 
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = torchvision.datasets.CIFAR10(root=save_dir, train=True,
                                                download=True, transform=transform)

        test_dataset = torchvision.datasets.CIFAR10(root=save_dir, train=False,
                                            download=True, transform=transform)

        # get in tensor form 
        x = []; y = []
        for i in range(len(train_dataset)): 
            (xx,yy) = train_dataset.__getitem__(i) 
            x.append(xx); y.append(yy)

        x_train = torch.stack(x, dim=0)
        y_train = torch.tensor(y, dtype=torch.long).view(-1,1)

        # split train -> train/valid 
        assert train_num + valid_num <= 50000, 'only 50k obs'
        _idxs = torch.randperm(50000)
        train_idx = _idxs[:train_num]
        valid_idx = _idxs[train_num:(train_num+valid_num)]

        x_valid = x_train[valid_idx, :]
        y_valid = y_train[valid_idx, :]

        x_train = x_train[train_idx, :]
        y_train = y_train[train_idx, :] 

        x = []; y = []
        for i in range(len(test_dataset)): 
            (xx,yy) = test_dataset.__getitem__(i) 
            x.append(xx); y.append(yy)

        x_test = torch.stack(x, dim=0)
        y_test = torch.tensor(y, dtype=torch.long).view(-1,1)

        # clean-up disk 
        if clean_up: 
            shutil.rmtree(save_dir)

        # corrupt endog 
        y_train, noise_idx = corrupt_label(y_train.detach().numpy().ravel(), noise_rate=endog_noise) 
        y_train = torch.tensor(y_train, dtype=torch.long).view(-1,1)
        endog_noise = np.zeros(x_train.size(0))
        endog_noise[noise_idx] = 1. 

        # add gaussian noise 
        exog_noise = exog_noise*torch.randn_like(x_train)
        x_train += exog_noise

        return x_train, y_train, x_valid, y_valid, x_test, y_test, exog_noise, endog_noise

    elif dataset == 'lincs': 
        pass 

    else:
        raise Exception('unrecognized dataset name.')



def get_corruption_scores(vals, noise_idx, train_size, noise_prop, n_scores=500):
    '''
    '''
    ks = np.linspace(1, train_size-1, n_scores)
    p_corr = []
    p_perfect = [] 
    p_random = [] 

    print(vals.shape)

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