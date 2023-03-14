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

    def __init__(self, x_train, y_train, x_valid, y_valid, predictor, policy, critic, problem, include_marginal=True): 
        ''''''
        self.x_train = x_train 
        self.y_train = y_train 
        self.x_valid = x_valid 
        self.y_valid = y_valid 

        self.predictor  = predictor
        self.policy     = policy
        self.critic     = critic
        self.problem    = problem

        self.include_marginal = include_marginal

        if self.problem == 'classification':
            self.num_classes = len(np.unique(self.y_train.detach().cpu().numpy()))

    def fit(self, model, x, y, crit, batch_size, lr, iters, metric, use_cuda=True, verbose=False): 
        '''Fits a model with a given number of iterations
        '''

        if torch.cuda.is_available() & use_cuda: 
            device = 'cuda'
        else: 
            device = 'cpu'

        model = model.to(device)
        optim = torch.optim.Adam(model.parameters(), lr=lr)

        advantages = []
        for i in range(iters):

            model.train()

            #x_batch = x[batch_idx].to(device)
            #y_batch = y[batch_idx].to(device)

            optim.zero_grad()
            yhat_batch = model(x)
            loss = crit(yhat_batch, y)
            loss.backward() 
            optim.step()

            advantages.append( self._get_perf(model, self.x_valid, self.y_valid, metric, device) ) 

            if verbose: print(f'epoch: {i} | loss: {loss.item()}', end='\r')

        return model.cpu().eval(), advantages

    def _get_perf(self, model, x, y, metric, device='cpu'): 
        '''return performance of `model` on `dataset` using `metric`'''
        model = model.eval()
        with torch.no_grad(): 
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
                
                y_onehot, y_val = self._get_endog_and_marginal(x=x_batch, y=y_batch, val_model=val_model)

                state = (x_batch, y_onehot, y_val)

                data_vals.append(estimator(*state).detach().cpu())
                
        return torch.cat(data_vals, dim=0)
    
    def _run_policy(self, state, policy):
        '''
        inputs 
            state       the x,y values 
            policy      the estimator 
        return 
            action      x,y selected values 
        '''
        pi = policy(*state)
        dist = torch.distributions.Bernoulli(probs=pi.view(-1,1))
        action = dist.sample()
        return action
    
    def _run_env(self, xs, ys, kwargs): 
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

        predictor = copy.deepcopy(self.predictor).train().to(device)
        
        predictor.reset_parameters()

        # compute advantages 
        predictor, advantages = self.fit(model      = predictor, 
                                            x          = xs, 
                                            y          = ys, 
                                            crit       = kwargs['crit_pred'], 
                                            batch_size = kwargs['batch_pred'], 
                                            lr         = kwargs['pred_lr'], 
                                            iters      = kwargs['iters_pred'], 
                                            use_cuda   = kwargs['use_cuda'],
                                            metric     = kwargs['perf_metric'], 
                                            verbose    = False)
        
        # flip sign depending on the metric; minimize vs maximize  
        # A(s,a)
        if kwargs['perf_metric'] in ['mse', 'bce']: 
            advantages = -np.array(advantages)
        elif kwargs['perf_metric'] in ['acc', 'auroc', 'r2']: 
            Asa = np.array(advantages)
        else: 
            raise Exception('not implemented')
        
        return advantages[-1]

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
                        noise_labels=None, entropy_beta=0, entropy_decay=1, eps=0.2, rl_batch=10, kappa=1.): 
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
        val_model, _ = self.fit(model=copy.deepcopy(self.predictor), x=self.x_valid, y=self.y_valid, crit=crit_pred, iters=inner_iter, use_cuda=use_cuda, lr=pred_lr, batch_size=iters_pred, metric=perf_metric)
        self.val_model = val_model

        policy = self.policy.to(device).train()
        critic = self.critic.to(device).train()

        val_model = val_model.to(device).eval()

        policy_optim = torch.optim.Adam(policy.parameters(), lr=estim_lr) 
        critic_optim = torch.optim.Adam(critic.parameters(), lr=estim_lr) 

        MSE = torch.nn.MSELoss()
        
        selected_idxs = torch.zeros((0,), dtype=torch.long)
        selected_returns = torch.zeros((0,))
        for i in range(outer_iter): 
            tic = time.time()

            # use same state for each iter
            with torch.no_grad(): 
                outer_idxs = torch.randperm(self.x_train.size(0))[:outer_batch]
                x,y,y_val = self._get_state(outer_idxs, device, val_model)

            # update old policy 
            old_policy = copy.deepcopy(policy).eval()

            actions = torch.zeros((0,)); advantages = torch.zeros((0,)); returns = torch.zeros((0,))
            perf_last = 0 
            for j in range(outer_batch): 
                with torch.no_grad(): 
                    print(f'collecting trajectory: {j}/{outer_batch}', end='\r')
                    state = (x[[j]], y[[j]], y_val[[j]])
                    action = self._run_policy(state=state, policy=old_policy)
                    vi = critic(*state).detach()
                    actions = torch.cat((actions, action.view(1)), dim=-1)

                if action == 0:
                    ri = 0
                    ai = 0
                    perf = None
                else: 
                    xs,ys = x[actions.nonzero(as_tuple=True)[0]], y[actions.nonzero(as_tuple=True)[0]]
                    perf = self._run_env(xs=xs, ys=ys, kwargs=kwargs)
                    ri = perf - perf_last 
                    ai = ri - vi
                    perf_last = perf
                    selected_idxs = torch.cat((selected_idxs, outer_idxs[[j]].view(-1)), dim=-1)
                    selected_returns = torch.cat((selected_returns, torch.tensor([ri], dtype=torch.float32)), dim=-1)

                returns = torch.cat((returns, torch.tensor([ri], dtype=torch.float32)), dim=-1)
                advantages = torch.cat((advantages, torch.tensor([ai], dtype=torch.float32)), dim=-1)

            # compute discounted rewards 
            # rt + γrt+1 + · · · + γn−1rt+n−1 + 
            # https://arxiv.org/pdf/1602.01783.pdf 
            for i in range(advantages.size(0)): 
                gamma = 1.01
                discounts = gamma**torch.arange(advantages.size(0) - i)
                advantages[i] = torch.sum(advantages[i:]*discounts)

            ####################
            ### critic updates 
            ####################
            _lcrit = []
            for k in range(iters_rl): 
                _losses = []
                for batch_idxs in torch.split(torch.randperm(selected_returns.size(0)), 1024): 
                    critic_optim.zero_grad()
                    data_idxs = selected_idxs[batch_idxs]
                    xx, yy, yy_val = self._get_state(data_idxs, device, val_model)
                    vi = critic(xx,yy,yy_val)
                    loss = MSE(vi.view(-1), selected_returns[batch_idxs].detach().view(-1))
                    loss.backward()
                    critic_optim.step()
                    _losses.append(loss.item())
                
                _lcrit.append(np.mean(_losses))

            plt.figure()
            plt.plot(_lcrit, 'b-', label='crit')
            plt.legend()
            plt.show()

            ####################
            ### policy updates 
            ####################

            # this shouldn't change during policy updates 
            old_pi = old_policy(x,y,y_val)
            old_policy_dist = torch.distributions.Bernoulli(probs=old_pi.view(-1,1))
            old_logpi = old_policy_dist.log_prob(actions).detach()

            _lclip = []; _lentr = []#; _lcrit = []
            for k in range(iters_rl): 
                print(f'updating policy/critic: {k}/{iters_rl}', end='\r')
                policy_optim.zero_grad() 

                pi = policy(x,y,y_val)
                policy_dist = torch.distributions.Bernoulli(probs=pi.view(-1,1))
                logpi = policy_dist.log_prob(actions)
                ratio = torch.exp(logpi - old_logpi)

                L_clip = - torch.minimum(ratio*advantages, torch.clip(ratio, 1-eps, 1+eps)*advantages).mean() 
                L_entr = - entropy_beta*policy_dist.entropy().mean() 

                loss = L_clip + L_entr
                loss.backward() 
                policy_optim.step()

                _lclip.append(L_clip.item())
                _lentr.append(L_entr.item())

            plt.figure()
            plt.plot(_lclip, 'r-', label='clip')
            plt.legend()
            plt.show()

            # logging 
            with torch.no_grad(): 

                p = policy(x,y,y_val)
                v = critic(x,y,y_val)
                p = p.view(-1,1)
                v = v.view(-1,1)

                print('p', p.view(-1)[:5])
                print('v', v.view(-1)[:5])

                # logging 
                if noise_labels is not None: 
                    iii = outer_idxs.detach().cpu().numpy()
                    #ppp = 1 - v.detach().cpu().numpy()
                    vvv = -v.detach().cpu().numpy()
                    ppp = -p.detach().cpu().numpy()
                    auc_vi = roc_auc_score(noise_labels[iii], vvv)
                    auc_pi = roc_auc_score(noise_labels[iii], ppp)

                    plt.figure()
                    bins = np.linspace(v.min(),v.max(),50)
                    plt.hist(v.detach().cpu().numpy().ravel()[(noise_labels[iii] == 1).nonzero()[0]], bins=bins, color='r', alpha=0.2)
                    plt.hist(v.detach().cpu().numpy().ravel()[(noise_labels[iii] == 0).nonzero()[0]], bins=bins, color='b', alpha=0.2)
                    plt.show() 

                    plt.figure()
                    bins = np.linspace(p.min(),p.max(),50)
                    plt.hist(p.detach().cpu().numpy().ravel()[(noise_labels[iii] == 1).nonzero()[0]], bins=bins, color='r', alpha=0.2)
                    plt.hist(p.detach().cpu().numpy().ravel()[(noise_labels[iii] == 0).nonzero()[0]], bins=bins, color='b', alpha=0.2)
                    plt.show() 
                else: 
                    auc = -1

                print(f'outer iteration: {i} || last perf: {perf_last:.4f} || noise auc (vi): {auc_vi:.2f} || noise auc (pi): {auc_pi:.2f}|| log entropy beta: {np.log10(entropy_beta + 1e-12):.2f} epoch elapsed: {(time.time() - tic):.2} s')
                entropy_beta*=entropy_decay
        # predict final data values 
        print()
        self.estimator = policy.eval()

        return self._predict_values(x=self.x_train, y=self.y_train, use_cuda=use_cuda)

                

                    
