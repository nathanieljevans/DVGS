'''
Implementation `Data Valuation with Reinforcement learning` by Nathaniel Evans (evansna@ohsu.edu). Citation: 

@misc{https://doi.org/10.48550/arxiv.1909.11671,
  doi = {10.48550/ARXIV.1909.11671},
  url = {https://arxiv.org/abs/1909.11671},
  author = {Yoon, Jinsung and Arik, Sercan O. and Pfister, Tomas},
  keywords = {Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Data Valuation using Reinforcement Learning},
  publisher = {arXiv},
  year = {2019},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

'''

import torch
import numpy as np
import time 
import copy
from sklearn.metrics import roc_auc_score

class DVRL(): 

    def __init__(self, x_train, y_train, x_valid, y_valid, predictor, estimator, problem, include_marginal=True): 
        ''''''
        self.x_train = x_train 
        self.y_train = y_train 
        self.x_valid = x_valid 
        self.y_valid = y_valid 

        self.predictor  = predictor
        self.estimator  = estimator 
        self.problem    = problem

        self.include_marginal = include_marginal

        if self.problem == 'classification':
            self.num_classes = len(np.unique(self.y_train.detach().cpu().numpy()))

    def fit(self, model, x, y, crit, batch_size=256, lr=1e-3, epochs=100, use_cuda=True, verbose=False): 
        '''pretrain predictor'''

        if torch.cuda.is_available() & use_cuda: 
            device = 'cuda'
        else: 
            device = 'cpu'

        model = model.to(device).train()
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        
        for i in range(epochs):
            for batch_idx in torch.split(torch.randperm(x.size(0)), batch_size): 

                x_batch = x[batch_idx, :].to(device)
                y_batch = y[batch_idx, :].to(device)

                optim.zero_grad()
                yhat_batch = model(x_batch)
                loss = crit(yhat_batch, y_batch)
                loss.backward() 
                optim.step()
                if verbose: print(f'epoch: {i} | loss: {loss.item()}', end='\r')

        return model.cpu().eval()

    def _get_perf(self, model, x, y, metric, device='cpu'): 
        '''return performance of `model` on `dataset` using `metric`'''
        yhat, y = self._predict(model, x, y, device=device)

        if metric == 'mse': 
            crit = torch.nn.MSELoss()
            return crit(yhat, y)
        elif metric == 'bce': 
            crit = torch.nn.BCELoss()
            return crit(yhat, y)
        elif metric == 'acc': 
            return ((yhat.argmax(dim=1)).view(-1) == y.view(-1)).sum()/y.size(0)
        elif metric == 'auroc': 
            # only applies to binary classification
            return roc_auc_score(y.detach().cpu().numpy().ravel(), yhat[:,1].detach().cpu().numpy())
        else: 
            raise Exception('not implemented')

    def _predict(self, model, x, y, device='cpu', batch_size=256): 
        '''return y,yhat for given model,dataset'''
        model = model.eval().to(device)
        _yhat = []
        _y = []
        with torch.no_grad(): 
            for batch_idx in torch.split(torch.arange(x.size(0)), batch_size): 

                x_batch = x[batch_idx, :].to(device)
                y_batch = y[batch_idx, :].to(device)

                _yhat.append(model(x_batch))
                _y.append(y_batch)
        return torch.cat(_yhat, dim=0), torch.cat(_y, dim=0)

    def _get_endog_and_marginal(self, x, y, val_model): 
        '''predict the `val_model` yhat and concat with y; input into estimator'''

        if self.problem == 'classification': 
            y_onehot = torch.nn.functional.one_hot(y.type(dtype=torch.long),num_classes=self.num_classes).type(torch.float).view(x.size(0), -1)
        else: 
            y_onehot = y 

        if self.include_marginal:
            with torch.no_grad(): 
                yhat_valid = val_model(x)
            
            # marginal calculation as done in DVRL code line 419-421 (https://github.com/google-research/google-research/blob/master/dvrl/dvrl.py)
            if self.problem == 'classification': 
                marginal = torch.abs(y_onehot - yhat_valid)
            else: 
                marginal = torch.abs(y_onehot - yhat_valid) / y_onehot
        
        # alternative option: don't include marginal; for sensitivity analysis 
        else: 
            marginal = torch.zeros_like(y_onehot)

        return torch.cat((y_onehot, marginal), dim=1)


    def _predict_values(self, x, y, use_cuda=True, batch_size=256): 
        '''estimator data values - returns probs'''

        if torch.cuda.is_available() & use_cuda: 
            device = 'cuda'
        else: 
            device = 'cpu'

        estimator = self.estimator.eval().to(device)
        val_model = self.val_model.eval().to(device)

        data_vals = []
        with torch.no_grad(): 
            for batch_idx in torch.split(torch.arange(x.size(0)), batch_size): 

                x_batch = x[batch_idx, :].to(device)
                y_batch = y[batch_idx, :].to(device)
                
                inp = self._get_endog_and_marginal(x=x_batch, y=y_batch, val_model=val_model)

                data_vals.append(estimator(x=x_batch,y=inp,idx=batch_idx.to(device)).detach().cpu())
                
        return torch.cat(data_vals, dim=0)

    def run(self, perf_metric, crit_pred, outer_iter=1000, inner_iter=100, outer_batch=2000,  inner_batch=256, estim_lr=1e-2, pred_lr=1e-2, moving_average_window=20, entropy_beta=0.1, entropy_decay=0.99, fix_baseline=False, use_cuda=True, noise_labels=None): 
        '''
        train the estimator model

        args: 
            perf_metric                     metric to optimize, current options: ["mse", "bce", "acc", "auroc"]
            crit_pred                       predictor loss criteria, NOTE: must not have an reduction, e.g., out.size(0) == x.size(0); sample loss must be multiplied by bernoulli sampled value ([0,1])
            outer_iter                      number of estimator epochs to train
            inner_iter                      number of iterations to train predictor model at each estimator step
            outer_batch                     outer loop batch size; Bs 
            inner_batch                     inner loop batch size; Bp
            estim_lr                        estimator learning rate
            pred_lr                         predictor learning rate
            moving_average_window           moving average window of the baseline
            entropy_beta                    starting entropy weight 
            entropy_decay                   exponential decay rate of entropy weight
            fix_baseline                    whether to use a fixed baseline or moving average window 
            use_cuda                        whether to use GPU if available 
        
        output 
            data values                     probability of inclusion [0,1]; shape: (nsamples,)
        '''
        print('pretraining `ori_model` and `val_model` ')
        ori_model = self.fit(copy.deepcopy(self.predictor), self.x_train, self.y_train, crit_pred, epochs=inner_iter, use_cuda=use_cuda)
        val_model = self.fit(copy.deepcopy(self.predictor), self.x_valid, self.y_valid, crit_pred, epochs=inner_iter, use_cuda=use_cuda)
        self.ori_model = ori_model 
        self.val_model = val_model

        if torch.cuda.is_available() & use_cuda: 
            device = 'cuda'
        else: 
            device = 'cpu'

        estimator = self.estimator.to(device).train()
        ori_model = ori_model.to(device).eval()
        val_model = val_model.to(device).eval()

        # baseline 
        d = self._get_perf(ori_model.eval(), self.x_valid, self.y_valid, metric=perf_metric, device=device)

        est_optim = torch.optim.Adam(estimator.parameters(), lr=estim_lr) 

        for i in range(outer_iter): 
            tic = time.time()
        
            outer_idxs = torch.randperm(self.x_train.size(0))[:outer_batch]
            x = self.x_train[outer_idxs, :]
            y = self.y_train[outer_idxs, :]
            x,y = x.to(device), y.to(device).view(-1,1)

            inp = self._get_endog_and_marginal(x=x, y=y, val_model=val_model)
            p = estimator(x=x,y=inp,idx=outer_idxs.to(device)).view(-1,1)
            dist = torch.distributions.Bernoulli(probs=p)
            s = dist.sample()

            #  select only s=1 train data ; speed up inner loop training 
            s_idx = s.nonzero(as_tuple=True)[0]

            if len(s_idx) == 0: 
                # could implement regularization term here (toward prob 0.5); but for ease we assume this is an uncommon occurence
                continue

            xs = x[s_idx, :].detach()
            ys = y[s_idx, :].detach()

             # NOTE:
            # in Algorithm 1 (https://arxiv.org/pdf/1909.11671.pdf) - theta is initialized only once, outside of outer loop; 
            # however, in the dvrl code (https://github.com/google-research/google-research/blob/master/dvrl/dvrl.py; lines 319-320) theta is reinitialized every inner loop. 
            # we follow the DVRL code in this respect. 
            predictor = copy.deepcopy(self.predictor).train().to(device)
            predictor = self.fit(predictor, xs, ys, crit_pred, inner_batch, pred_lr, inner_iter, use_cuda, verbose=False)
            dvrl_perf = self._get_perf(predictor, self.x_valid, self.y_valid, metric=perf_metric, device=device)

            # compute reward 
            if perf_metric in ['mse', 'bce']: 
                reward = d - dvrl_perf
            elif perf_metric in ['acc', 'auroc']: 
                reward = dvrl_perf - d
            else: 
                raise Exception('not implemented')

            # NOTE: 
            # divergence from the DVRL algorithm and code
            # including entropy regularizes the data values and stabilizes training
            # This [paper](https://arxiv.org/pdf/1811.11214.pdf) recommends that entropy and large learning rates work better. Also, suggests that entropy term decay during training improves convergence to optimal policies. 
            # This [paper](https://arxiv.org/pdf/1602.01783.pdf) originally showed improvements in modern RL approaches with entropy regularization
            # original dvrl: exploration_weight*(max(p.mean() - exploration_threshold, 0) + max(1-exploration_threshold-p.mean(), 0))
            
            exploration_weight = 1e3
            exploration_threshold = 0.9
            explore_loss = -entropy_beta*dist.entropy().mean() + exploration_weight*(max(p.mean() - exploration_threshold, 0) + max((1-exploration_threshold)-p.mean(), 0))

            # NOTE: this is equivalent to torch.sum(s*torch.log(p) + (1-s)*torch.log(1-p))   
            log_prob = dist.log_prob(s).sum()

            # update estimator params 
            est_optim.zero_grad()
            loss = -reward*log_prob + explore_loss
            loss.backward() 
            est_optim.step() 

            # update baseline
            # NOTE: 
            # in Algorithm 1 (https://arxiv.org/pdf/1909.11671.pdf) - a moving average baseline is used; 
            # however, in the code the authors of DVRL do not update the baseline during estimator training. 
            if not fix_baseline: 
                d = (moving_average_window - 1)*d/moving_average_window + dvrl_perf/moving_average_window

            # entropy term decay; set `entropy_decay=1` to use fixed entropy term. 
            entropy_beta *= entropy_decay

            if noise_labels is not None: 
                iii = outer_idxs.detach().cpu().numpy()
                ppp = 1 - p.detach().cpu().numpy()
                auc = roc_auc_score(noise_labels[iii], ppp)
            else: 
                auc = -1

            print(f'outer iteration: {i} || reward: {reward:.4f} || dvrl perf: {dvrl_perf:.4f} || baseline: {d:.4f} || log entropy beta: {np.log10(entropy_beta + 1e-10):.4f} || noise auc: {auc:.2f}|| epoch elapsed: {(time.time() - tic):.1} s', end='\r')

        # predict final data values 
        return self._predict_values(x=self.x_train, y=self.y_train, use_cuda=use_cuda)

                

                    
