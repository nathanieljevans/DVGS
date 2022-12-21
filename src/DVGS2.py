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
from torchmetrics.functional import pairwise_cosine_similarity 
import copy 

class DVGS2(): 

    def __init__(self, source_dataset, target_dataset, model): 
        ''''''
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.model = model
    
    def _get_grad(self, loss, params): 
        ''''''
        dl = torch.autograd.grad(loss, params, create_graph=True)
        dl= torch.cat([x.view(-1) for x in dl])
        return dl

    def run(self, crit, optim=torch.optim.Adam, lr=1e-2, num_epochs=100, batch_size=512, num_workers=1, grad_params=None, verbose=True, use_cuda=True): 
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

        model = copy.deepcopy(self.model)
        optim = optim(model.parameters(), lr=lr)
        
        # if no grad params are provided
        if grad_params is None: 
            grad_params = [n for n,_ in model.named_parameters()]

        source_loader = torch.utils.data.DataLoader(self.source_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        target_loader = torch.utils.data.DataLoader(self.target_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

        if torch.cuda.is_available() & use_cuda: 
            device = 'cuda'
        else: 
            device = 'cpu'
            if verbose: print('source gradient calculations will use randomness="same" instead of "different" as it with GPU.')
        if verbose: print('using device:', device)

        model.to(device)
        #edge_index = torch.empty(size=(2,0), device=device, dtype=torch.long)

        A = torch.zeros((len(self.source_dataset), len(self.source_dataset)), device='cpu', dtype=torch.float32)

        elapsed = []
        for epoch in range(num_epochs): 
            losses = []
            tic = time.time()
            for x, y, idx in source_loader:

                x,y,idx = x.to(device), y.to(device), idx.to(device).squeeze()

                # compute sample gradients + pairwise similarity
                fmodel, params, buffers = make_functional_with_buffers(model)
                ft_compute_sample_grad = get_per_sample_grad_func(crit, fmodel, device)
                ft_per_sample_grads = ft_compute_sample_grad(params, buffers, x, y)
                batch_grads = torch.cat([_g.view(x.size(0), -1) for _g, (n,p) in zip(ft_per_sample_grads, model.named_parameters()) if n in grad_params], dim=1)
                batch_sims = pairwise_cosine_similarity(batch_grads) # (num_samples, num_samples); batch_sims[i,j] is sim between sample i,j

                iii = torch.meshgrid(idx.cpu(), idx.cpu(), indexing='ij')
                A[iii] += batch_sims.detach().cpu()

                '''
                # select top 90% most similar samples and add edges between them 
                _r,_c = torch.triu_indices(*batch_sims.size(), offset=1)
                t = torch.quantile(batch_sims[(_r,_c)], q=0.99) 
                assert t > 0, 't is less than or equal to zero'
                batch_sims = torch.triu(batch_sims)
                new_edge_index = idx[(1. * (batch_sims >= t)).nonzero()].T # (2, num edges)
                edge_index = torch.cat((edge_index, new_edge_index), dim=1)
                '''

                # update weights
                if False: 
                    optim.zero_grad()
                    yhat = model(x)
                    loss = crit(yhat, y)
                    loss.backward() 
                    optim.step()
                else: 
                    for x,y in target_loader: 
                        x,y = x.to(device), y.to(device)
                        # update weights
                        optim.zero_grad()
                        yhat = model(x)
                        loss = crit(yhat, y)
                        loss.backward() 
                        optim.step()

                # logging 
                losses.append(loss.item())
            
            elapsed = time.time()-tic
            print(f'epoch {epoch} || avg loss: {np.mean(losses):.2f} || elapsed: {np.mean(elapsed):.0f} s/epoch', end='\r')

        return A
        #return edge_index


def get_per_sample_grad_func(crit, fmodel, device): 

    def compute_loss_stateless_model (params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)
        predictions = fmodel(params, buffers, batch) 
        loss = crit(predictions, targets)
        return loss

    ft_compute_grad = grad(compute_loss_stateless_model)
    
    if device == 'cpu': 
        return vmap(ft_compute_grad, in_dims=(None, None, 0, 0), randomness='same')
    else: 
        return vmap(ft_compute_grad, in_dims=(None, None, 0, 0), randomness='different')

