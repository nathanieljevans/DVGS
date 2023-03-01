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

class DVRL2(): 

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
            return r2_score(y.detach().cpu().numpy().ravel(), yhat.detach().cpu().numpy().ravel(), multioutput='uniform_average')
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

        return y_onehot, marginal


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
    
    def _run_policy(self, state, policy):
        '''
        inputs 
            state       the x,y values 
            policy      the estimator 
        return 
            action      x,y selected values 
        '''
        pi = policy(*state).view(-1,1)
        dist = torch.distributions.Bernoulli(probs=pi)
        action = dist.sample()
        #print(action.size());3/0
        #sum_log_pi = dist.log_prob(action).sum()
        
        return action
    
    def _run_env(self, action, state, kwargs): 
        '''
        the environment in this case is the performance of the trained model, when given a set of selected data (x,y), e.g., the action. 

        inputs 
            state       the x,y values 
            action      the selected x,y values 

        return 
            advantage   reward - baseline_reward 
        '''

        ## Get Device for training 
        if torch.cuda.is_available() & kwargs['use_cuda']: 
            device = 'cuda'
        else: 
            device = 'cpu'

        x,y,y_val = state 
        
        # compute action reward 

        #  select only s=1 train data ; speed up inner loop training 
        s_idx = action.nonzero(as_tuple=True)[0]
        xs = x[s_idx].detach()
        ys = y[s_idx].detach()
        predictor = copy.deepcopy(self.predictor).train().to(device)
        predictor = self.fit(model      = predictor, 
                             x          = xs, 
                             y          = ys, 
                             crit       = kwargs['crit_pred'], 
                             batch_size = kwargs['batch_pred'], 
                             lr         = kwargs['pred_lr'], 
                             iters      = kwargs['iters_pred'], 
                             use_cuda   = kwargs['use_cuda'], 
                             verbose    = False)

        with torch.no_grad(): 
            # Reward of given action; R(a)
            Ra = self._get_perf(model        = predictor, 
                                x            = kwargs['x_valid'], 
                                y            = kwargs['y_valid'], 
                                metric       = kwargs['perf_metric'], 
                                device       = device)
        
        # compute advantages 
        # A(s,a)
        if kwargs['perf_metric'] in ['mse', 'bce']: 
            Asa = - Ra
        elif kwargs['perf_metric'] in ['acc', 'auroc', 'r2']: 
            Asa = Ra
        else: 
            raise Exception('not implemented')
        
        return Asa 

    def _get_state(self, outer_idxs, device, val_model): 
        x = self.x_train[outer_idxs]
        y = self.y_train[outer_idxs]
        x = x.to(device)
        y = y.to(device)
        y_onehot, y_val = self._get_endog_and_marginal(x=x, y=y, val_model=val_model)
        state = (x,y_onehot,y_val)
        return state

    def run(self, perf_metric, crit_pred, outer_iter=1000, inner_iter=10, iters_rl=100, iters_pred=100, 
                    outer_batch=2000,  batch_pred=256, estim_lr=1e-3, pred_lr=1e-3, use_cuda=True, 
                        noise_labels=None, entropy_beta=0, eps=0.2, rl_batch=10): 
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
            use_cuda                        whether to use GPU if available 
        
        output 
            data values                     probability of inclusion [0,1]; shape: (nsamples,)
        '''

        if torch.cuda.is_available() & use_cuda: 
            device = 'cuda'
        else: 
            device = 'cpu'

        kwargs = {'x_valid'     : self.x_valid, 
                  'y_valid'     : self.y_valid, 
                  'perf_metric' : perf_metric, 
                  'iters_pred'  : iters_pred,
                  'pred_lr'     : pred_lr, 
                  'use_cuda'    : use_cuda, 
                  'crit_pred'   : crit_pred, 
                  'batch_pred'  : batch_pred
                  } 
        
        # for marginal calc
        val_model = self.fit(model=copy.deepcopy(self.predictor), x=self.x_valid, y=self.y_valid, crit=crit_pred, iters=inner_iter, use_cuda=use_cuda, lr=pred_lr, batch_size=iters_pred)
        self.val_model = val_model

        policy = self.estimator.to(device).train()
        old_policy = copy.deepcopy(policy).eval()

        val_model = val_model.to(device).eval()

        est_optim = torch.optim.Adam(policy.parameters(), lr=estim_lr) 

        actions = []; advantages = []; state_idxs = []; old_policys = []
        for i in range(outer_iter): 
            tic = time.time()

            # use same state for each iter
            with torch.no_grad(): 
                outer_idxs = torch.randperm(self.x_train.size(0))[:outer_batch]
                state = self._get_state(outer_idxs, device, val_model)

            # update old policy 
            old_policy = copy.deepcopy(policy).eval()

            for j in range(inner_iter): 
                
                action = self._run_policy(state=state, policy=policy)
                advantage = self._run_env(action=action, state=state, kwargs=kwargs)
                actions.append(action); advantages.append(advantage); state_idxs.append(outer_idxs); old_policys.append(old_policy)

            # clip samples to the NNN most recently collected 
            # really old policies won't have much impact and are essentially obsolete 
            MAX_RL_SAMPLES = 100
            if len(actions) > MAX_RL_SAMPLES: 
                actions = actions[-MAX_RL_SAMPLES:]
                advantages = advantages[-MAX_RL_SAMPLES:]
                state_idxs = state_idxs[-MAX_RL_SAMPLES:]
                old_policys = old_policys[-MAX_RL_SAMPLES:]

            # normalize advantages 
            adv_norm = (np.array(advantages) - np.mean(advantages))/(np.std(advantages) + 1e-8)

            # train policy parameters for `iters_rl`
            est_optim.zero_grad()
            for k in range(iters_rl): 
                for _idx in torch.randperm(inner_iter)[:rl_batch]:
                    _action = actions[_idx]
                    _advantage = adv_norm[_idx]
                    _state_idx = state_idxs[_idx]
                    _state = self._get_state(_state_idx, device, val_model)
                    _old_policy = old_policys[_idx]

                    policy_dist = torch.distributions.Bernoulli(probs=policy(*_state).view(-1,1))
                    log_pi = policy_dist.log_prob(_action)

                    log_pi_old = torch.distributions.Bernoulli(probs=_old_policy(*_state).view(-1,1)).log_prob(_action).detach()

                    ratio = torch.exp(log_pi - log_pi_old)
                    
                    L_clip = -torch.minimum(ratio*_advantage, torch.clip(ratio, 1-eps, 1+eps)*_advantage).mean()
                    loss = L_clip - entropy_beta*policy_dist.entropy().mean()
                    loss = loss/rl_batch # average instead of sum
                    loss.backward() 
                #torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.)
                est_optim.step()

            # logging 
            with torch.no_grad(): 

                p = policy(*state).view(-1,1)

                # logging 
                if noise_labels is not None: 
                    iii = outer_idxs.detach().cpu().numpy()
                    ppp = 1 - p.detach().cpu().numpy()
                    auc = roc_auc_score(noise_labels[iii], ppp)

                    plt.figure()
                    bins = np.linspace(0,1,50)
                    plt.hist(p.detach().cpu().numpy().ravel()[(noise_labels[iii] == 1).nonzero()[0]], bins=bins, color='r', alpha=0.2)
                    plt.hist(p.detach().cpu().numpy().ravel()[(noise_labels[iii] == 0).nonzero()[0]], bins=bins, color='b', alpha=0.2)
                    plt.show() 
                else: 
                    auc = -1
                    n_corr_sel = -1

                print(f'outer iteration: {i} || avg. advantage: {np.mean(advantages[-inner_iter:]):.4f} || noise auc: {auc:.2f} || epoch elapsed: {(time.time() - tic):.1} s', end='\r')

        # predict final data values 
        print()
        self.estimator = policy.eval()
        return self._predict_values(x=self.x_train, y=self.y_train, use_cuda=use_cuda)

                

                    
