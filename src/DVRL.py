'''

'''

import torch
import numpy as np
import time 
import copy
from sklearn.metrics import roc_auc_score

class DVRL(): 

    def __init__(self, train_dataset, valid_dataset, test_dataset, predictor, estimator): 
        ''''''
        self.train = train_dataset 
        self.valid = valid_dataset 
        self.test = test_dataset 
        self.predictor = predictor
        self.estimator = estimator 

    def pretrain(self, model, dataset, crit, num_workers=1, batch_size=256, lr=1e-3, epochs=100, use_cuda=True): 
        '''pretrain predictor'''

        if torch.cuda.is_available() & use_cuda: 
            device = 'cuda'
        else: 
            device = 'cpu'

        model = model.to(device)
        loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        
        for i in range(epochs):
            for x,y in loader: 
                x,y = x.to(device), y.to(device)
                optim.zero_grad()
                yhat = model(x)
                loss = crit(yhat, y)
                loss.backward() 
                optim.step()
                print(f'epoch: {i} | loss: {loss.item()}', end='\r')

        return model.cpu()

    def _get_perf(self, model, dataset, metric, device='cpu'): 
        ''''''
        yhat, y = self._predict(model, dataset, device=device)

        if metric == 'mse': 
            crit = torch.nn.MSELoss()
            return crit(yhat, y)
        elif metric == 'bce': 
            crit = torch.nn.BCELoss()
            return crit(yhat, y)
        elif metric == 'acc': 
            return ((1.*yhat > 0.5) == y.squeeze(1)).sum()/y.size(0)
        elif metric == 'auroc': 
            # only applies to binary classification
            return roc_auc_score(y.detach().cpu().numpy().ravel(), yhat[:,1].detach().cpu().numpy())
        else: 
            raise Exception('not implemented')

    def _predict(self, model, dataset, device='cpu'): 
        loader = torch.utils.data.DataLoader(dataset, num_workers=1, batch_size=256, shuffle=False)
        model.eval()
        _yhat = []
        _y = []
        with torch.no_grad(): 
            for x,y in loader: 
                x,y = x.to(device), y.to(device)
                _yhat.append(model(x))
                _y.append(y)
        return torch.cat(_yhat, dim=0), torch.cat(_y, dim=0)

    def _predict_values(self, dataset, use_cuda=True): 
        '''estimator data values - returns logits'''
        loader = torch.utils.data.DataLoader(dataset, num_workers=1, batch_size=256, shuffle=False)

        if torch.cuda.is_available() & use_cuda: 
            device = 'cuda'
        else: 
            device = 'cpu'

        estimator = self.estimator.eval().to(device)
        val_model = self.val_model.eval().to(device)
        ori_model = self.ori_model.eval().to(device)

        data_vals = []
        with torch.no_grad(): 
            for x,y in loader: 
                x,y = x.to(device), y.to(device)
                with torch.no_grad(): 
                    yhat_valid = val_model(x)
                    yhat_train = ori_model(x)
                    m = torch.abs(yhat_train - yhat_valid).detach() # marginal 
                inp = torch.cat((x,y,m), dim=1)
                data_vals.append(estimator(inp).detach().cpu())
        return torch.cat(data_vals, dim=0)

    def run(self, perf_metric, crit_pred, outer_iter=1000, inner_iter=100, outer_batch=2000, outer_workers=1, inner_batch=256, estim_lr=1e-2, pred_lr=1e-2, moving_average_window=20, exploration_weight=1e-3, exploration_threshold=0.9, fix_baseline=False, use_cuda=True): 
        '''
        train the estimator model

        args: 
            perf_metric                     metric to optimize, current options: ["mse", "bce", "acc", "auroc"]
            crit_pred                       predictor loss criteria, NOTE: must not have an reduction, e.g., out.size(0) == x.size(0); sample loss must be multiplied by bernoulli sampled value ([0,1])
            outer_iter                      number of estimator epochs to train
            inner_iter                      number of iterations to train predictor model at each estimator step
            outer_batch                     outer loop batch size; Bs 
            outer_workers                   number of workers to use for the outer loop dataloader
            inner_batch                     inner loop batch size; Bp
            estim_lr                        estimator learning rate
            pred_lr                         predictor learning rate
            moving_average_window           moving average window of the baseline
        
        output 
            data values (logits) 
        '''
        print('pretraining `ori_model` and `val_model` ')
        ori_model = self.pretrain(copy.deepcopy(self.predictor), self.train, crit_pred, epochs=50)
        val_model = self.pretrain(copy.deepcopy(self.predictor), self.valid, crit_pred, epochs=50)

        if torch.cuda.is_available() & use_cuda: 
            device = 'cuda'
        else: 
            device = 'cpu'

        estimator = self.estimator.to(device).train()
        ori_model = ori_model.to(device).eval()
        val_model = val_model.to(device).eval()

        # baseline 
        d=self._get_perf(ori_model.eval(), self.valid, metric=perf_metric, device=device)

        est_optim = torch.optim.Adam(estimator.parameters(), lr=estim_lr) 
        train_loader = torch.utils.data.DataLoader(self.train, num_workers=outer_workers, batch_size=outer_batch, shuffle=True)

        #predictor = copy.deepcopy(self.predictor).train().to(device)
        #pred_optim = torch.optim.Adam(predictor.parameters(), lr=pred_lr)

        for i in range(outer_iter): 
            tic = time.time()
            for x,y in train_loader:
                x,y = x.to(device), y.to(device)
                est_optim.zero_grad()
                with torch.no_grad(): 
                    yhat_valid = val_model(x)
                    yhat_train = ori_model(x)
                    m = torch.abs(yhat_train - yhat_valid).detach() # marginal 

                logits = estimator(torch.cat((x,y,m), dim=1))
                #dist = torch.distributions.Bernoulli(logits=logits)
                dist = torch.distributions.Binomial(total_count=1, logits=logits)
                s = dist.sample()

                predictor = copy.deepcopy(self.predictor).train().to(device)
                pred_optim = torch.optim.Adam(predictor.parameters(), lr=pred_lr)

                #  select only s=1 train data  
                s_idx = s.nonzero(as_tuple=True)[0]
                xs = x[s_idx, :]
                ys = y[s_idx, :]

                for j in range(inner_iter):
                    pred_optim.zero_grad()
                    batch_idx = torch.randperm(xs.size(0))[:inner_batch]
                    xx = xs[batch_idx, :]
                    yy = ys[batch_idx, :]
                    yyhat = predictor(xx)
                    loss = crit_pred(yyhat, yy)
                    loss.backward()
                    pred_optim.step()

                dvrl_perf = self._get_perf(predictor, self.valid, metric=perf_metric, device=device)

                if perf_metric in ['mse', 'bce']: 
                    reward = d - dvrl_perf
                elif perf_metric in ['acc', 'auroc']: 
                    reward = dvrl_perf - d 
                else: 
                    raise Exception('not implemented')

                p = torch.sigmoid(logits)
                loss = -dist.log_prob(s).sum() * reward + exploration_weight*(max(p.mean() - exploration_threshold, 0) + max(1-exploration_threshold-p.mean(), 0))
                loss.backward() 
                est_optim.step() 

            # update baseline
            if not fix_baseline: 
                d = (moving_average_window - 1)*d/moving_average_window + dvrl_perf/moving_average_window

            print(f'outer iteration: {i} || reward: {reward:.4f} || dvrl perf: {dvrl_perf:.4f} || baseline: {d:.4f} || epoch elapsed: {(time.time() - tic):.1} s', end='\r')

        self.estimator = estimator 
        self.ori_model = ori_model 
        self.val_model = val_model

        return self._predict_values(dataset=self.train, use_cuda=use_cuda)

                

                    
