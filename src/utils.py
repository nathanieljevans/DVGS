
import numpy as np
import copy
import torch
from GenDataset import GenDataset

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


def get_filtered_scores(vals, model, crit, metric, x_train, y_train, x_test, y_test, qs=np.linspace(0., 0.5, 5), batch_size=256, num_workers=1, lr=1e-3, epochs=200, repl=1):
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
            _, remove_low_res = train_model(model           = copy.deepcopy(model),
                                            crit            = crit, 
                                            metric          = metric,
                                            train_dataset   = GenDataset(x = x_train[remove_low_mask, :], y=y_train[remove_low_mask, :]), 
                                            test_dataset    = GenDataset(x = x_test, y = y_test), 
                                            batch_size      = batch_size,
                                            num_workers     = num_workers, 
                                            lr              = lr,
                                            epochs          = epochs)

            _, remove_high_res = train_model(model           = copy.deepcopy(model),
                                            crit            = crit, 
                                            metric          = metric,
                                            train_dataset   = GenDataset(x = x_train[remove_high_mask, :], y=y_train[remove_high_mask, :]), 
                                            test_dataset    = GenDataset(x = x_test, y = y_test), 
                                            batch_size      = batch_size,
                                            num_workers     = num_workers, 
                                            lr              = lr,
                                            epochs          = epochs)

            _low_res.append(remove_low_res)
            _high_res.append(remove_high_res)

        remove_low_values.append(np.mean(_low_res))
        remove_high_values.append(np.mean(_high_res))
    print()

    return remove_low_values, remove_high_values
        


def get_filtered_scores_TORCHVISION(vals, model, crit, metric, train_dataset, test_dataset, qs=np.linspace(0., 0.5, 5), batch_size=256, num_workers=1, lr=1e-3, epochs=200, repl=1):
    '''for use with torchvision datasets'''

    remove_low_values = []
    for i,q in enumerate(qs): 
        print(f'training filtered models... progress: {i}/{len(qs)}', end='\r')
        low_t = np.quantile(vals, q)

        remove_low_mask = (vals >= low_t).nonzero()[0] 

        _train_dataset = copy.deepcopy(train_dataset)
        _train_dataset.data = _train_dataset.data[remove_low_mask,:,:]
        _train_dataset.targets = _train_dataset.targets[remove_low_mask]
        
        _low_res = []
        for j in range(repl): 

            _, remove_low_res = train_model(model           = copy.deepcopy(model),
                                            crit            = crit, 
                                            metric          = metric,
                                            train_dataset   = _train_dataset, 
                                            test_dataset    = test_dataset, 
                                            batch_size      = batch_size,
                                            num_workers     = num_workers, 
                                            lr              = lr,
                                            epochs          = epochs)

            _low_res.append(remove_low_res)

        remove_low_values.append(np.mean(_low_res))
    print()

    return remove_low_values

def train_model(model, crit, metric, train_dataset, test_dataset, batch_size=256, num_workers=1, lr=1e-3, epochs=200, verbose=False, use_cuda=True): 

    if torch.cuda.is_available() & use_cuda: 
        device = 'cuda'
    else: 
        device = 'cpu'
    #if verbose: print('using device:', device)

    model = copy.deepcopy(model).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    for epoch in range(epochs): 
        _losses = []
        for x,y in train_loader: 
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
    for x,y in test_loader: 
        x,y = x.to(device), y.to(device)
        _ys.append(y.detach().cpu())
        _yhats.append(model(x).detach().cpu())
    y = torch.cat(_ys, dim=0).detach().cpu().numpy()
    yhat = torch.cat(_yhats, dim=0).detach().cpu().numpy()
    res = metric(y, yhat)
    return model, res