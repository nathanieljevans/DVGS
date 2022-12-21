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

    def __init__(self, train_dataset, valid_dataset, test_dataset, model): 
        ''''''
        self.train = train_dataset 
        self.valid = valid_dataset 
        self.test = test_dataset 
        self.model = model

    def pretrain_(self, crit, num_workers=1, batch_size=256, lr=1e-3, epochs=100, use_cuda=True, verbose=True, report_metric=None): 
        '''
        '''

        if torch.cuda.is_available() & use_cuda: 
            device = 'cuda'
        else: 
            device = 'cpu'
        if verbose: print('using device:', device)

        self.model.train().to(device)
        loader = torch.utils.data.DataLoader(self.train, num_workers=num_workers, batch_size=batch_size, shuffle=True)
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)

        for i in range(epochs): 
            losses = []
            for x,y in loader: 
                x,y = myto(x,y, device)
                optim.zero_grad()
                yhat = self.model(x)
                loss = crit(yhat, y)
                loss.backward()
                optim.step()
                losses.append(loss.item())
                if report_metric is not None: 
                    metric = report_metric(y.detach().cpu().numpy(), yhat.detach().cpu().numpy())
                    #print(f'batch loss: {loss.item():.2f} || batch metric: {metric:.2f}', end='\r')
                else: 
                    metric = -666
            if verbose: print(f'epoch: {i} | loss: {np.mean(losses):.4f} | metric: {metric:0.4f}', end='\r')

    
    def _get_grad(self, loss, params): 
        ''''''
        dl = torch.autograd.grad(loss, params, create_graph=True)
        dl= torch.cat([x.view(-1) for x in dl])
        return dl

    def run(self, crit, save_dir='./dvgs_results/', similarity=torch.nn.CosineSimilarity(dim=1), optim=torch.optim.Adam, lr=1e-2, num_epochs=100, compute_every=1, target_batch_size=512, source_batch_size=512, num_workers=1, grad_params=None, verbose=True, use_cuda=True): 
        '''
        trains the model and returns data values 

        args: 
            crit                            loss criteria, e.g., torch.nn.BCELoss()
            save_dir                        directory to save each iteration of gradient similarities to disk
            similarity                      similarity metric
            optim                           optimization method 
            lr                              learning rate 
            num_epochs                      number of epochs to train the model 
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
        optim = optim(model.parameters(), lr=lr)
        
        # if no grad params are provided
        if grad_params is None: 
            grad_params = [n for n,_ in model.named_parameters()]

        valid_loader = torch.utils.data.DataLoader(self.valid, batch_size=target_batch_size, num_workers=num_workers, shuffle=True)
        train_loader = torch.utils.data.DataLoader(self.train, batch_size=source_batch_size, num_workers=num_workers, shuffle=False)

        # place holder for vals 
        data_vals = torch.zeros((len(self.train),))

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
        for epoch in range(num_epochs): 
            losses = []
            for x_valid, y_valid in valid_loader:
                model.train()
                x_valid, y_valid = myto(x_valid, y_valid, device)

                # step 1: compute validation gradient
                optim.zero_grad()
                yhat_valid = model(x_valid)
                loss = crit(yhat_valid, y_valid)
                grad_valid = self._get_grad(loss, [p for n,p in model.named_parameters() if n in grad_params])

                if iter%compute_every==0: 
                    tic = time.time()
                    # step 2: for each train sample 
                    j = 0
                    # to speed up per-sample gradient calculations 
                    fmodel, params, buffers = make_functional_with_buffers(model)
                    ft_compute_sample_grad = get_per_sample_grad_func(crit, fmodel, device)
                    for x_batch, y_batch in train_loader: 
                        _tic = time.time()
                        x_batch, y_batch = myto(x_batch, y_batch, device)
                        ft_per_sample_grads = ft_compute_sample_grad(params, buffers, x_batch, y_batch)
                        batch_grads = torch.cat([_g.view(y_batch.size(0), -1) for _g, (n,p) in zip(ft_per_sample_grads, model.named_parameters()) if n in grad_params], dim=1)
                        batch_sim = similarity(grad_valid.unsqueeze(0).expand(y_batch.size(0), -1), batch_grads).detach().cpu()
                        data_vals[j:int(j + y_batch.size(0))] = batch_sim 
                        print(f'epoch {epoch} || computing sample gradients... [{j}/{len(self.train)}] ({((time.time()-_tic)/source_batch_size)*len(self.train)/60:.4f} min/source-epoch)', end='\r')
                        j += y_batch.size(0)
                    nn += 1
                    np.save(f'{save_dir}/{self.run_id}/data_value_iter={nn}', data_vals)
                    elapsed.append(time.time()-tic)

                # step 3: optimize on validation gradient 
                optim.zero_grad()
                loss.backward()
                optim.step()
                losses.append(loss.item())
                iter += 1

            print(f'epoch {epoch} || avg loss: {np.mean(losses):.2f} || grad time elapsed: {np.mean(elapsed):.1f} s', end='\r')

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

