'''
Implementation of `TMC-Data Shapley` by Nathaniel Evans (evansna@ohsu.edu). Citation: 

@inproceedings{ghorbani2019data,
  title={Data Shapley: Equitable Valuation of Data for Machine Learning},
  author={Ghorbani, Amirata and Zou, James},
  booktitle={International Conference on Machine Learning},
  pages={2242--2251},
  year={2019}
}
'''
import torch 
import copy
import numpy as np
from scipy.stats import spearmanr
from matplotlib import pyplot as plt 

class V():
    '''performance score'''
    def __init__(self, val_dataset, perf_metric):
        self.val_dataset =  val_dataset
        self.perf_metric = perf_metric

    def get_score(self, model):
        model.eval()
        with torch.no_grad(): 
            y = self.val_dataset.y.detach().cpu().numpy()
            yhat = model(self.val_dataset.x).argmax(dim=-1).detach().cpu().numpy()

        return self.perf_metric(y, yhat)

    def to(self, device): 
        self.val_dataset.x = self.val_dataset.x.to(device)
        self.val_dataset.y = self.val_dataset.y.to(device)

    
class DShap(): 
    ''''''
    def __init__(self, model, crit, train_dataset, V, epochs=50, lr=1e-3, optim=torch.optim.Adam, tol=0.01): 
        self.model = model 
        self.crit = crit
        self.train_dataset = train_dataset
        self.V = V
        self.epochs = epochs
        self.lr = lr
        self.optim = optim
        self.tol = tol

        _ii = 25
        # get v0 
        self.v0 = np.mean([V.get_score(model=self.train_(model     = copy.deepcopy(self.model), 
                                                         x         = self.train_dataset.x,
                                                         y         = self.train_dataset.y[torch.randperm(self.train_dataset.y.size(0)), :])) for _ in range(_ii)])

        # get VD 
        self.vD = np.mean([V.get_score(model=self.train_(model     = copy.deepcopy(self.model), 
                                                         x         = self.train_dataset.x,
                                                         y         = self.train_dataset.y)) for _ in range(_ii)])

        print(f'v0: {self.v0:.4f}')
        print(f'vD: {self.vD:.4f}')
        print()

    def train_(self, model, x,y):
        '''faster training for small batches'''
        model = model.train()
        optim = self.optim(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs): 
            optim.zero_grad()
            loss = self.crit(model(x), y)
            loss.backward()
            optim.step() 

        return model

    def to(self, device): 
        self.model = self.model.to(device)
        self.train_dataset.x = self.train_dataset.x.to(device)
        self.train_dataset.y = self.train_dataset.y.to(device)

    def TMC(self, max_iterations=1000, min_iterations=100, use_cuda=True, T=5, stopping_criteria=0.999): 
        ''''''
        print('starting Data Shapley TMC...')
        phi = np.zeros((len(self.train_dataset), max_iterations))

        if torch.cuda.is_available() & use_cuda: 
            device='cuda'
        else: 
            device='cpu'

        # move everything over to model device first
        self.to(device)
        self.V.to(device)
        for t in range(max_iterations): 

            pi = torch.randperm(len(self.train_dataset))
            vj = self.v0

            trunc_counter = 0
            for j in range(len(self.train_dataset)): 
                if trunc_counter > 5:
                    break 
                
                model = copy.deepcopy(self.model)
                idx = pi[0:(j+1)]
                x,y = self.train_dataset.x[idx, :], self.train_dataset.y[idx, :]
                model = self.train_(model, x, y)
                vj_new = self.V.get_score(model)

                phi[pi[j], t] = (vj_new - vj)

                vj = vj_new

                if np.abs(self.vD - vj) < self.tol*self.vD: 
                    trunc_counter += 1
                else: 
                    trunc_counter = 0 

            #max_err = error(phi[:, :t].T, min_iter=min_iterations)
            if t > (T+1): 
                running_rank_corr = np.mean(compute_rank_convergence(phi[:, :t])[-T:])
            else: 
                running_rank_corr = -1.

            if (running_rank_corr > stopping_criteria) & (t >= min_iterations): 
                print()
                print(f'MC stopping criteria met. running avg rank correlation: {running_rank_corr:.4f}')
                break

            print(f'iter: {t} || max j: {j} || max vj: {vj:.4f} || rank_corr: {running_rank_corr:.4f}', end='\r')

        return phi[:, :t].mean(axis=1)


def compute_rank_convergence(phi): 

    vals_i = [phi[:, :i].mean(axis=1) for i in range(1, phi.shape[1])]  # compute cumulative data value at iteration i 
    
    # compute spearman correlation of each consecutive iteration cumulative data value 
    # expect to see asymptotic change approaching 1 (e.g., perfect corr; no rank change)
    rhos = []
    for i in range(1, len(vals_i)):
        last = vals_i[i-1]
        curr = vals_i[i]
        r,p = spearmanr(last, curr)
        rhos.append(r)

    return rhos
