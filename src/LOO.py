'''leave one out data values'''

import torch 
import copy
import numpy as np


class LOO: 
    def __init__(self, x_train, y_train, x_valid, y_valid, model, metric, crit, epochs=100, lr=1e-3, optim=torch.optim.Adam, batch_size=256, use_cuda=True, verbose=True, baseline_repl=1, n_repl=1): 

        self.x_train = x_train 
        self.y_train = y_train 
        self.x_valid = x_valid 
        self.y_valid = y_valid 

        self.model = model
        self.metric = metric 
        self.crit = crit 

        self.epochs = epochs 
        self.lr = lr 
        self.optim = optim 
        self.batch_size = batch_size
        self.baseline_repl = baseline_repl
        self.n_repl = n_repl
        self.verbose = verbose

        if torch.cuda.is_available() and use_cuda: 
            self.device = 'cuda'
        else: 
            self.device = 'cpu'

        self.baseline, self.baseline_std = self._get_baseline()
        if self.verbose: print(f'baseline performance: {self.baseline:.3f} +/- {self.baseline_std:.3f}')

    def _fit(self, x, y): 

        model = copy.deepcopy(self.model).train().to(self.device)
        optim = self.optim(model.parameters(), lr=self.lr)
        crit = self.crit

        for i in range(self.epochs): 
            _loss = []
            for batch_idx in torch.split(torch.randperm(x.size(0)), self.batch_size):
                x_batch = x[batch_idx, :].to(self.device)
                y_batch = y[batch_idx, :].to(self.device)

                optim.zero_grad() 
                yhat_batch = model(x_batch)
                loss = crit(yhat_batch, y_batch)
                loss.backward() 
                optim.step() 
                _loss.append(loss.item())

        return model.cpu().eval()

    def _get_perf(self, model, x, y): 

        yy = [] 
        yyhat = []
        for batch_idx in torch.split(torch.arange(x.size(0)), self.batch_size):
            x_batch = x[batch_idx, :]
            y_batch = y[batch_idx, :]

            yy.append(y_batch.detach().cpu().numpy())
            yyhat.append(model(x_batch).detach().cpu().numpy())

        yy = np.concatenate(yy, axis=0)
        yyhat = np.concatenate(yyhat, axis=0)

        perf = self.metric(yy, yyhat)

        return perf 

    def _get_baseline(self): 
        
        _perfs = []
        for i in range(self.baseline_repl): 
            model = self._fit(self.x_train, self.y_train)
            _perfs.append(self._get_perf(model, self.x_valid, self.y_valid))
        return np.mean(_perfs), np.std(_perfs)

    def run(self): 
        
        data_values = []
        for exclude_idx in range(self.x_train.size(0)): 
            if self.verbose: 
                print(f'[progress: {exclude_idx}/{self.x_train.size(0)}]', end='\r')
            include_idx = np.delete(np.arange(self.x_train.size(0)), [exclude_idx])
            print(len(include_idx))

            x = self.x_train[include_idx, :]
            y = self.y_train[include_idx, :]

            perfs = []
            for n in range(self.n_repl): 
                model = self._fit(x, y)
                perfs.append(self._get_perf(model, self.x_valid, self.y_valid))
            perf = np.mean(perfs)
            data_values.append(perf - self.baseline)

        return data_values


            

