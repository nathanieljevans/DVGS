'''

'''

import torch
import numpy as np
import time 
from functorch import make_functional_with_buffers, vmap, grad
from uuid import uuid4
from os import listdir, mkdir
from os.path import exists
from shutil import rmtree

class DVGS(): 

    def __init__(self, x_source, y_source, x_target, y_target, model): 
        ''''''
        self.x_source = x_source 
        self.y_source = y_source 
        self.x_target = x_target 
        self.y_target = y_target
        self.model = model

    def pretrain_(self, crit, batch_size=256, lr=1e-3, epochs=100, use_cuda=True, verbose=True, report_metric=None): 
        '''
        in-place model pre-training on source dataset 
        '''
        if torch.cuda.is_available() & use_cuda: 
            device = 'cuda'
        else: 
            device = 'cpu'
        if verbose: print('using device:', device)

        self.model.train().to(device)
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)

        for i in range(epochs): 
            losses = []
            for idx_batch in torch.split(torch.randperm(self.x_source.size(0)), batch_size): 
                x = self.x_source[idx_batch, :]
                y = self.y_source[idx_batch, :]
                x,y = myto(x,y,device)
                optim.zero_grad()
                yhat = self.model(x)
                loss = crit(yhat, y)
                loss.backward()
                optim.step()
                losses.append(loss.item())
                if report_metric is not None: 
                    metric = report_metric(y.detach().cpu().numpy(), yhat.detach().cpu().numpy())
                else: 
                    metric = -666
            if verbose: print(f'epoch: {i} | loss: {np.mean(losses):.4f} | metric: {metric:0.4f}', end='\r')

    
    def _get_grad(self, loss, params): 
        ''''''
        dl = torch.autograd.grad(loss, params, create_graph=True)
        dl= torch.cat([x.view(-1) for x in dl])
        return dl

    def run(self, target_crit, source_crit, num_restarts=1, save_dir='./dvgs_results/', similarity=torch.nn.CosineSimilarity(dim=1), optim=torch.optim.Adam, lr=1e-2, num_epochs=100, compute_every=1, target_batch_size=512, source_batch_size=512, num_workers=1, grad_params=None, verbose=True, use_cuda=True): 
        '''
        trains the model and returns data values 

        args: 
            crit                            loss criteria, e.g., torch.nn.BCELoss()
            save_dir                        directory to save each iteration of gradient similarities to disk
            similarity                      similarity metric
            optim                           optimization method 
            lr                              learning rate 
            iterations                      number of iterations to train the model 
            compute_every                   period to compute gradient similarities, value of 1 will compute every step. 
            target_batch_size               batch_size to use for model "training" on the target dataset; if batch_size > len(self.valid) then batches won't be used. 
            source_batch_size               batch size to use for gradient calculation on the source dataset; reducing this can improve memory foot print. 
            num_workers                     number of workers to use with the validation dataloader 
            grad_params                     list of parameter names to be used to compute gradient similarity; If None, then all parameters are used. 
            verbose                         if True, will print epoch loss and accuracy 
            use_cuda                        use cuda-enabled GPU if available
        
        output: 
            data values     (N_t,N_p)       N_t is number of training samples, N_p is the number of sampled data values (int(num_epochs/compute_every)) 
        '''
        if not exists(save_dir): mkdir(save_dir)
        self.run_id = uuid4()
        mkdir(f'{save_dir}/{self.run_id}')

        model = self.model
        #optim = optim(model.parameters(), lr=lr)
        
        # if no grad params are provided
        if grad_params is None: 
            grad_params = [n for n,_ in model.named_parameters()]

        # place holder for vals 
        data_vals = torch.zeros((self.x_source.size(0),))

        if torch.cuda.is_available() & use_cuda: 
            device = 'cuda'
        else: 
            device = 'cpu'
            print('source gradient calculations will use randomness="same" instead of "different" as it with GPU.')
        if verbose: print('using device:', device)

        model.to(device)

        nn = 0
        iter = 0
        elapsed = []
        for ii in range(num_restarts):
            model.reset_parameters() 
            opt = optim(model.parameters(), lr=lr)

            for epoch in range(num_epochs): 
                losses = []

                for idx_target in torch.split(torch.randperm(self.x_target.size(0)), target_batch_size):
            
                    x_target = self.x_target[idx_target, :]
                    y_target = self.y_target[idx_target, :]
                    model.train()
                    x_target, y_target = myto(x_target, y_target, device)

                    # step 1: compute target/validation gradient
                    yhat_target = model(x_target)
                    loss = target_crit(yhat_target, y_target)
                    grad_target = self._get_grad(loss, [p for n,p in model.named_parameters() if n in grad_params])

                    if iter%compute_every==0: 
                        tic = time.time()
                        # step 2: for each source/train sample 
                        j = 0
                        # to speed up per-sample gradient calculations 
                        fmodel, params, buffers = make_functional_with_buffers(model)
                        ft_compute_sample_grad = get_per_sample_grad_func(source_crit, fmodel, device)
                        for idx_source in torch.split(torch.arange(self.x_source.size(0)), source_batch_size): 
                            _tic = time.time()
                            x_source = self.x_source[idx_source, :]
                            y_source = self.y_source[idx_source, :]
                            x_source, y_source = myto(x_source, y_source, device)
                            ft_per_sample_grads = ft_compute_sample_grad(params, buffers, x_source, y_source)
                            batch_grads = torch.cat([_g.view(y_source.size(0), -1) for _g, (n,p) in zip(ft_per_sample_grads, model.named_parameters()) if n in grad_params], dim=1)
                            batch_sim = similarity(grad_target.unsqueeze(0).expand(y_source.size(0), -1), batch_grads).detach().cpu()
                            data_vals[j:int(j + y_source.size(0))] = batch_sim 
                            #print(f'epoch {epoch} || computing sample gradients... [{j}/{self.x_source.size(0)}] ({((time.time()-_tic)/source_batch_size)*self.x_source.size(0)/60:.4f} min/source-epoch)', end='\r')
                            j += y_source.size(0)
                        nn += 1
                        np.save(f'{save_dir}/{self.run_id}/data_value_iter={nn}', data_vals)
                        elapsed.append(1e6 * (time.time()-tic)/idx_source.size(0)) # micro-seconds 

                    # step 3: optimize on target/validation gradient 
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    losses.append(loss.item())
                    iter += 1

                print(' '*100, end='\r')
                print(f'[restart: {ii}] iteration {epoch} || avg target loss: {np.mean(losses):.2f} || gradient sim. calc. elapsed / sample: {np.mean(elapsed):.1f} us', end='\r')

            print()
        return self.run_id

    def agg(self, path, reduction='mean'):
        '''aggregate all data values in path directory'''
        files = [x for x in listdir(path) if 'data_value_' in x]

        if reduction == 'none': 
            return np.stack([np.load(f'{path}/{f}').ravel() for f in files], axis=1)
        
        elif reduction == 'mean': 
            # NOTE: more memory efficient for large datasets
            x = np.load(f'{path}/{files[0]}')
            for f in files[1:]: 
                x += np.load(f'{path}/{f}')
            return x / len(files)

    def clean(self, path): 
        '''removes path'''
        rmtree(path)

def myto(x_valid, y_valid, device): 
    ''''''
    if torch.is_tensor(x_valid): 
        x_valid, y_valid = x_valid.to(device), y_valid.to(device)
    else: 
        y_valid = y_valid.to(device)
        x_valid = [el.to(device) for el in x_valid]

    return x_valid, y_valid

def get_per_sample_grad_func(crit, fmodel, device): 

    def compute_loss_stateless_model (params, buffers, sample, target):
        if torch.is_tensor(sample):
            batch = sample.unsqueeze(0)
        else:
            # list of tensors, from LINCSDataset
            batch = [el.unsqueeze(0) for el in sample] 

        targets = target.unsqueeze(0)
        predictions = fmodel(params, buffers, batch) 
        loss = crit(predictions, targets)
        return loss

    ft_compute_grad = grad(compute_loss_stateless_model)
    
    if device == 'cpu': 
        return vmap(ft_compute_grad, in_dims=(None, None, 0, 0), randomness='same')
    else: 
        return vmap(ft_compute_grad, in_dims=(None, None, 0, 0), randomness='different')

