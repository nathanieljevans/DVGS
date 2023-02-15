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
from sklearn.metrics import roc_auc_score, r2_score
from matplotlib import pyplot as plt

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

    def fit(self, model, x, y, crit, batch_size, lr, iters, use_cuda=True, verbose=False): 
        '''Fits a model with a given number of iterations
        
        NOTE: There is a critical distinction between training with iterations rather than epochs; with epochs, more obs. results in 
              more iterations/gradient-updates, whereas with iterations the amount of data is indenpendant of the number of iterations. 
              This is critical for our DVRL implementation b/c the RL algorithm may be confounded by performance differences due to number 
              of weight updates, e.g., hyper parameter of `inner_iter`. By training with iterations (rather than epochs), we ensure independance 
              of `inner_iter` with sample importance. 
        '''

        if torch.cuda.is_available() & use_cuda: 
            device = 'cuda'
        else: 
            device = 'cpu'

        model = model.to(device).train()
        optim = torch.optim.Adam(model.parameters(), lr=lr)

        for i in range(iters):
            
            batch_idx = torch.randint(0, x.size(0), size=(batch_size,))

            x_batch = x[batch_idx].to(device)
            y_batch = y[batch_idx].to(device)

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
            return crit(yhat, y).item()
        elif metric == 'r2': 
            return r2_score(y.detach().cpu().numpy(), yhat.detach().cpu().numpy(), multioutput='uniform_average')
        elif metric == 'bce': 
            crit = torch.nn.BCELoss()
            return crit(yhat, y).item()
        elif metric == 'acc': 
            return ((1.*((yhat.argmax(dim=1)).view(-1) == y.view(-1))).mean()).item()
        elif metric == 'auroc': 
            y = y.view(-1).detach().cpu().numpy()
            yhat = torch.softmax(yhat, dim=-1).detach().cpu().numpy()

            if yhat.shape[1] == 2: 
                # binary classification
                return roc_auc_score(y, yhat[:,1])
            elif yhat.shape[1] > 2: 
                # multiclass 
                return roc_auc_score(y, yhat, multi_class='ovr')
            else: 
                # yhat size == 1; only makes sense for regression - and auroc is for classification 
                # assumes we're not using BCE
                raise Exception('is this a regression problem? choose different perf. metric.')
        else: 
            raise Exception('not implemented')

    def _predict(self, model, x, y, device='cpu', batch_size=256): 
        '''return y,yhat for given model, dataset'''
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
            y_onehot = torch.nn.functional.one_hot(y.type(dtype=torch.long), num_classes=self.num_classes).type(torch.float).view(x.size(0), -1)
        else: 
            y_onehot = y 

        if self.include_marginal:
            with torch.no_grad(): 
                yhat_logits = val_model(x)
                yhat_valid = torch.softmax(yhat_logits, dim=-1)
            
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

                data_vals.append(estimator(x=x_batch,y=inp).detach().cpu())
                
        return torch.cat(data_vals, dim=0)

    def run(self, perf_metric, crit_pred, outer_iter=1000, inner_iter=100, outer_batch=2000,  inner_batch=256, estim_lr=1e-2, pred_lr=1e-2, moving_average_window=20, fix_baseline=False, use_cuda=True, noise_labels=None): 
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

        if torch.cuda.is_available() & use_cuda: 
            device = 'cuda'
        else: 
            device = 'cpu'

        print('pretraining `ori_model` and `val_model` ')
        # init. baseline calc
        # do this a few times to get good estimate.
        _d = []
        for i in range(3): 
            print(f'{i+1}/4', end='\r')
            ori_model = self.fit(model=copy.deepcopy(self.predictor), x=self.x_train, y=self.y_train, crit=crit_pred, iters=inner_iter, use_cuda=use_cuda, lr=pred_lr, batch_size=inner_batch)
            _d.append(self._get_perf(model=ori_model.eval(), x=self.x_valid, y=self.y_valid, metric=perf_metric, device=device))
        print(_d)
        d = np.mean(_d)

        # for marginal calc
        val_model = self.fit(model=copy.deepcopy(self.predictor), x=self.x_valid, y=self.y_valid, crit=crit_pred, iters=inner_iter, use_cuda=use_cuda, lr=pred_lr, batch_size=inner_batch)
        print('4/4')
        self.val_model = val_model

        estimator = self.estimator.to(device).train()
        val_model = val_model.to(device).eval()

        est_optim = torch.optim.Adam(estimator.parameters(), lr=estim_lr, weight_decay=0) 

        for i in range(outer_iter): 
            tic = time.time()

            with torch.no_grad(): 
                #outer_idxs = torch.randint(0, self.x_train.size(0), size=(outer_batch,))
                outer_idxs = torch.randperm(self.x_train.size(0))[:outer_batch]
                
                x = self.x_train[outer_idxs]
                y = self.y_train[outer_idxs]
                x = x.to(device)
                y = y.to(device).view(y.size(0),-1)
                inp = self._get_endog_and_marginal(x=x, y=y, val_model=val_model)
            
            p = estimator(x=x,y=inp).view(-1,1)
            dist = torch.distributions.Bernoulli(probs=p)
            s = dist.sample()

            #  select only s=1 train data ; speed up inner loop training 
            s_idx = s.nonzero(as_tuple=True)[0]

            xs = x[s_idx].detach()
            ys = y[s_idx].detach()

            # NOTE:
            # in Algorithm 1 (https://arxiv.org/pdf/1909.11671.pdf) - theta is initialized only once, outside of outer loop; 
            # however, in the dvrl code (https://github.com/google-research/google-research/blob/master/dvrl/dvrl.py; lines 319-320) theta is reinitialized every inner loop. 
            # we follow the DVRL code in this respect. 
            predictor = copy.deepcopy(self.predictor).train().to(device)
            #predictor.reset_parameters()
            predictor = self.fit(model=predictor, x=xs, y=ys, crit=crit_pred, batch_size=inner_batch, lr=pred_lr, iters=inner_iter, use_cuda=use_cuda, verbose=False)
            
            with torch.no_grad(): 
                dvrl_perf = self._get_perf(model=predictor, x=self.x_valid, y=self.y_valid, metric=perf_metric, device=device)

                # compute reward 
                if perf_metric in ['mse', 'bce']: 
                    reward = d - dvrl_perf
                elif perf_metric in ['acc', 'auroc', 'r2']: 
                    reward = dvrl_perf - d
                else: 
                    raise Exception('not implemented')

            # NOTE: adding entropy regularization may help stabilize training 
            # This [paper](https://arxiv.org/pdf/1811.11214.pdf) recommends that entropy and large learning rates work better. Also, suggests that entropy term decay during training improves convergence to optimal policies. 
            # This [paper](https://arxiv.org/pdf/1602.01783.pdf) originally showed improvements in modern RL approaches with entropy regularization
            
            exploration_weight = 1e3
            exploration_threshold = 0.9
            explore_loss = exploration_weight*(max(p.mean() - exploration_threshold, 0) + max((1-exploration_threshold)-p.mean(), 0))

            # NOTE: this is equivalent to torch.sum(s*torch.log(p) + (1-s)*torch.log(1-p))   
            log_prob = dist.log_prob(s).sum()

            # update estimator params 
            est_optim.zero_grad()
            loss = -reward*log_prob + explore_loss
            loss.backward() 
            est_optim.step() 

            with torch.no_grad(): 
                # update baseline
                # NOTE: 
                # in Algorithm 1 (https://arxiv.org/pdf/1909.11671.pdf) - a moving average baseline is used; 
                # however, in the code the authors of DVRL do not update the baseline during estimator training. 
                if not fix_baseline: 
                    d = (moving_average_window - 1)*d/moving_average_window + dvrl_perf/moving_average_window

                # logging 
                if noise_labels is not None: 
                    iii = outer_idxs.detach().cpu().numpy()
                    ppp = 1 - p.detach().cpu().numpy()
                    auc = roc_auc_score(noise_labels[iii], ppp)
                    n_corr_sel = int(noise_labels[iii][s_idx.detach().cpu().numpy()].sum())
                else: 
                    auc = -1
                    n_corr_sel = -1

                print(f'outer iteration: {i} || reward: {reward:.4f} || dvrl perf: {dvrl_perf:.4f} || baseline: {d:.4f} || noise auc: {auc:.2f}|| crpt/tot: {n_corr_sel}/{s_idx.size(0)} [{n_corr_sel/s_idx.size(0):.2f}] || epoch elapsed: {(time.time() - tic):.1} s', end='\r')

        # predict final data values 
        print()
        return self._predict_values(x=self.x_train, y=self.y_train, use_cuda=use_cuda)

                

                    
