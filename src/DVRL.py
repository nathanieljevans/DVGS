'''

'''

import torch
import numpy as np
import time 
import copy

class DVRL(): 

    def __init__(self, train_dataset, valid_dataset, test_dataset, predictor, estimator): 
        ''''''
        self.train = train_dataset 
        self.valid = valid_dataset 
        self.test = test_dataset 
        self.predictor = predictor
        self.estimator = estimator 

    def _get_perf(model, dataset, metric): 
        ''''''
        yhat, y = self._predict(model, dataset)

        if metric == 'mse': 
            crit = torch.nn.MSELoss()
            return crit(yhat, y)
        elif metric == 'bce': 
            crit = torch.nn.BCELoss()
            return crit(yhat, y)
        else: 
            raise Exception('not implemented')

    def _predict(self, model, dataset): 
        loader = torch.utils.data.DataLoader(dataset, num_workers=1, batch_size=256)
        yhat = []
        y = []
        with torch.no_grad(): 
            for x,y in loader: 
                yhat.append(model(x))
                y.append(y)
        return torch.cat(yhat, dim=0), torch.cat(y, dim=0)

    def run(self, perf_metric, crit_pred, outer_iter=1000, inner_iter=100, outer_batch=2000, outer_workers=1, inner_batch=256, estim_lr=1e-2, pred_lr=1e-2, moving_average_window=20): 
        '''

        args: 
            perf_metric                     metric to optimize, current options: ["mse", "bce"]
            crit_pred                       predictor loss criteria, NOTE: must not have an reduction, e.g., out.size(0) == x.size(0); sample loss must be multiplied by bernoulli sampled value ([0,1])
            outer_iter
            inner_iter  
            outer_batch
            outer_workers 
            inner_batch 
            estim_lr
            pred_lr 
            moving_average_window
        
        '''

        # baseline 
        d=self._get_perf(self.predictor, self.valid, metric=perf_metric)

        est_optim = torch.optim.Adam(self.estimator.parameters(), lr=estim_lr) 


        train_loader = torch.utils.data.DataLoader(self.train, num_workers=outer_workers, batch_size=outer_batch)


        for i in range(outer_iter): 
            tic = time.time()
            for x,y in train_loader:
                est_optim.zero_grad()
                logits = self.estimator(torch.cat((x,y), dim=1))
                dist = torch.distributions.Bernoulli(logits=logits+1e-8)
                s = dist.sample()

                predictor = copy.deepcopy(self.predictor)
                pred_optim = torch.optim.Adam(predictor, lr=pred_lr) 
                for j in range(inner_iter):
                    pred_optim.zero_grad()
                    batch_idx = torch.randint(0,x.size(0),size=inner_batch)
                    xx = x[batch_idx, :]
                    yy = y[batch_idx, :]
                    ss = s[batch_idx]
                    yyhat = predictor(xx)
                    loss = ss*crit_pred(yyhat, yy)
                    loss.backward()
                    pred_optim.step()

                dvrl_perf = self._get_perf(predictor, self.valid, metric=perf_metric)
                
                if perf_metric in ['mse', 'bce']: 
                    reward = d - dvrl_perf
                else: 
                    reward = dvrl_perf - d 
                    raise Exception('not implemented')

                loss = -dist.log_prob(s) * reward
                loss.backward() 
                est_optim.step() 

            # update baseline
            d = (moving_average_window - 1)*d/moving_average_window + dvrl_perf/moving_average_window

            print(f'outer iteration: {i} || reward: {reward:.4f} || dvrl perf: {dvrl_perf:.4f} || epoch elapsed: {(time.time() - tic):.1} s')

                

                    
