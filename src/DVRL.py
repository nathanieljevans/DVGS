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

    def __init__(self, train_dataset, valid_dataset, test_dataset, predictor, estimator, problem): 
        ''''''
        self.train      = train_dataset 
        self.valid      = valid_dataset 
        self.test       = test_dataset 
        self.predictor  = predictor
        self.estimator  = estimator 
        self.problem    = problem

        if self.problem == 'classification':
            self.num_classes = len(np.unique([self.train.__getitem__(i)[1] for i in range(len(self.train))]))

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
        '''return performance of `model` on `dataset` using `metric`'''
        yhat, y = self._predict(model, dataset, device=device)

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

    def _predict(self, model, dataset, device='cpu'): 
        '''return y,yhat for given model,dataset'''
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
        '''estimator data values - returns probs'''
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
                if self.problem == 'classification':
                    y_onehot = torch.nn.functional.one_hot(y.type(dtype=torch.long),num_classes=self.num_classes).type(torch.float).view(x.size(0), -1)
                    inp = torch.cat((y_onehot,m), dim=1)
                else: 
                    inp = torch.cat((y,m), dim=1)
                data_vals.append(estimator(x,inp).detach().cpu())
        return torch.cat(data_vals, dim=0)

    def run(self, perf_metric, crit_pred, outer_iter=1000, inner_iter=100, outer_batch=2000, outer_workers=1, inner_batch=256, estim_lr=1e-2, pred_lr=1e-2, moving_average_window=20, entropy_beta=0.1, entropy_decay=0.99, fix_baseline=False, use_cuda=True): 
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
            entropy_beta                    starting entropy weight 
            entropy_decay                   exponential decay rate of entropy weight
            fix_baseline                    whether to use a fixed baseline or moving average window 
            use_cuda                        whether to use GPU if available 
        
        output 
            data values                     probability of inclusion [0,1]; shape: (nsamples,)
        '''
        print('pretraining `ori_model` and `val_model` ')
        ori_model = self.pretrain(copy.deepcopy(self.predictor), self.train, crit_pred, epochs=inner_iter)
        val_model = self.pretrain(copy.deepcopy(self.predictor), self.valid, crit_pred, epochs=inner_iter)

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

        for i in range(outer_iter): 
            tic = time.time()
            for x,y in train_loader:
                x,y = x.to(device), y.to(device).view(-1,1)
                est_optim.zero_grad()
                with torch.no_grad(): 
                    yhat_valid = val_model(x)
                    yhat_train = ori_model(x)
                    m = torch.abs(yhat_train - yhat_valid).detach() # 'marginal' 

                if self.problem == 'classification':
                    y_onehot = torch.nn.functional.one_hot(y.type(dtype=torch.long),num_classes=self.num_classes).type(torch.float).view(x.size(0), -1)
                    inp = torch.cat((y_onehot,m), dim=1)
                else: 
                    inp = torch.cat((y,m), dim=1)
                
                p = estimator(x,inp).view(-1,1)
                dist = torch.distributions.Bernoulli(probs=p)
                s = dist.sample()

                # NOTE:
                # in Algorithm 1 (https://arxiv.org/pdf/1909.11671.pdf) - theta is initialized only once, outside of outer loop; 
                # however, in the dvrl code (https://github.com/google-research/google-research/blob/master/dvrl/dvrl.py; lines 319-320) theta is reinitialized every inner loop. 
                # we follow the DVRL code in this respect. 
                predictor = copy.deepcopy(self.predictor).train().to(device)
                pred_optim = torch.optim.Adam(predictor.parameters(), lr=pred_lr)

                #  select only s=1 train data ; speed up inner loop training 
                s_idx = s.nonzero(as_tuple=True)[0]
                if len(s_idx) == 0: 
                    # if no samples were selected.
                    _loss = (-entropy_beta*dist.entropy().mean())
                    _loss.backward()
                    est_optim.step()
                    continue
                xs = x[s_idx, :]
                ys = y[s_idx, :]

                for j in range(inner_iter):
                    # NOTE: 
                    # in Algorithm 1 (https://arxiv.org/pdf/1909.11671.pdf) - mini-batches are sampled for Ns iterations 
                    # in the code (https://github.com/google-research/google-research/blob/master/dvrl/dvrl.py; line 323) the inner loop model is trained for Ns epochs. 
                    # this behavior is likely to affect performance as batch sizes change. We follow DVRL code and train for Ns epochs. 
                    # we follow the DVRL code in this respect. 
                    batch_idxs = torch.randperm(xs.size(0)).split(inner_batch)
                    for batch_idx in batch_idxs: 
                        pred_optim.zero_grad()
                        xx = xs[batch_idx, :]
                        yy = ys[batch_idx, :]
                        yyhat = predictor(xx)
                        loss = crit_pred(yyhat, yy)
                        loss.backward()
                        pred_optim.step()

                dvrl_perf = self._get_perf(predictor, self.valid, metric=perf_metric, device=device)

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
                explore_loss = -entropy_beta*dist.entropy().sum()   

                # NOTE: this is equivalent to torch.sum(s*torch.log(p) + (1-s)*torch.log(1-p))   
                log_prob = dist.log_prob(s).sum()

                # update estimator params 
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
            print(f'outer iteration: {i} || reward: {reward:.4f} || dvrl perf: {dvrl_perf:.4f} || baseline: {d:.4f} || log entropy beta: {np.log10(entropy_beta):.4f} || epoch elapsed: {(time.time() - tic):.1} s', end='\r')

        self.estimator = estimator 
        self.ori_model = ori_model 
        self.val_model = val_model

        # predict final data values 
        return self._predict_values(dataset=self.train, use_cuda=use_cuda)

                

                    
