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
    def __init__(self, x_valid, y_valid, perf_metric):
        self.x = x_valid 
        self.y = y_valid 
        self.perf_metric = perf_metric

    def get_score(self, model):
        model.eval()
        with torch.no_grad(): 
            y = self.y.detach().cpu().numpy()
            yhat = model(self.x).detach().cpu().numpy()
        return self.perf_metric(y, yhat)

    def to(self, device): 
        self.x = self.x.to(device)
        self.y = self.y.to(device)

    
class DShap(): 
    ''''''
    def __init__(self, model, crit, x_train, y_train, x_valid, y_valid, perf_metric, epochs=50, lr=1e-3, optim=torch.optim.Adam, tol=0.01, baseline_repl=10, verbose=True): 

        self.x_train = x_train 
        self.y_train = y_train
        self.V = V(x_valid, y_valid, perf_metric)
        self.model = model 
        self.crit = crit
        self.epochs = epochs
        self.lr = lr
        self.optim = optim
        self.tol = tol
        self.verbose = verbose

        # get v0 
        self.v0 = np.mean([self.V.get_score(model=self.fit(model     = copy.deepcopy(self.model), 
                                                      x         = self.x_train,
                                                      y         = self.y_train[torch.randperm(self.y_train.size(0)), :])) for _ in range(baseline_repl)])

        # get VD 
        self.vD = np.mean([self.V.get_score(model=self.fit(model     = copy.deepcopy(self.model), 
                                                      x         = self.x_train,
                                                      y         = self.y_train)) for _ in range(baseline_repl)])

        if self.verbose: print(f'v0 (null model): {self.v0:.4f}')
        if self.verbose: print(f'vD (all data): {self.vD:.4f}')
        if self.verbose: print()

    def fit(self, model, x,y):
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
        '''move all data to device; speed up for small datasets'''
        self.model = self.model.to(device)
        self.x_train = self.x_train.to(device)
        self.y_train = self.y_train.to(device)

    def TMC(self, max_iterations=1000, min_iterations=100, use_cuda=True, T=5, stopping_criteria=0.999): 
        ''''''
        if self.verbose: print('starting Data Shapley TMC...')
        phi = np.zeros((self.x_train.size(0), max_iterations))

        if torch.cuda.is_available() & use_cuda: 
            device='cuda'
        else: 
            device='cpu'

        # move everything over to model device first
        self.to(device)
        self.V.to(device)

        for t in range(max_iterations): 
            pi = torch.randperm(self.x_train.size(0))
            vj = self.v0

            trunc_counter = 0
            for j in range(self.x_train.shape[0]): 
                if trunc_counter > 5:
                    break 
                
                model = copy.deepcopy(self.model)
                idx = pi[0:(j+1)]
                x,y = self.x_train[idx, :], self.y_train[idx, :]
                model = self.fit(model, x, y)
                vj_new = self.V.get_score(model)

                phi[pi[j], t] = (vj_new - vj)

                vj = vj_new

                if np.abs(self.vD - vj) < self.tol*self.vD: 
                    trunc_counter += 1
                else: 
                    trunc_counter = 0 

            # NOTE: 
            # divergence from Data Shapley paper
            # here we use a convergence critieria based on rank change 
            # since our primary outcome (label corruption) is rank based
            if t > (T+1): 
                running_rank_corr = np.mean(compute_rank_convergence(phi[:, :t])[-T:])
            else: 
                running_rank_corr = -1.

            if (running_rank_corr > stopping_criteria) & (t >= min_iterations): 
                if self.verbose: print()
                if self.verbose: print(f'MC stopping criteria met. running avg rank correlation: {running_rank_corr:.4f}')
                break

            if self.verbose: print(f'iter: {t} || max j: {j} || max vj: {vj:.4f} || rank_corr: {running_rank_corr:.4f}', end='\r')

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
