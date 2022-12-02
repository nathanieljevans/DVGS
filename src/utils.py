
import numpy as np
import copy
import torch

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


def train_model(model, crit, train_dataset, test_dataset, batch_size=256, num_workers=1, lr=1e-3, epochs=200): 

    model = copy.deepcopy(model)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    for epoch in range(epochs): 

        optim.zero_grad()
        yhat_train = model(x_train).squeeze()
        loss = crit(yhat_train, y_train)
        loss.backward()
        optim.step()

    yhat_test = 1. * (model(x_test).detach().numpy() > 0.5).ravel()
    y_test = y_test.detach().numpy().ravel()

    acc = (yhat_test == y_test).sum() / y_test.shape[0]

    return acc