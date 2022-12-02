'''

'''

import torch
import numpy as np
import time 
from functorch import make_functional_with_buffers, vmap, grad

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
                x,y =x.to(device), y.to(device)
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

    def run(self, crit, similarity=torch.nn.CosineSimilarity(dim=1), optim=torch.optim.Adam, lr=1e-2, num_epochs=100, compute_every=1, batch_size=512, num_workers=1, grad_params=None, verbose=True, use_cuda=True): 
        '''
        trains the model and returns data values 

        args: 
            crit                            loss criteria, e.g., torch.nn.BCELoss()
            similarity                      similarity metric
            optim                           optimization method 
            lr                              learning rate 
            num_epochs                      number of epochs to train the model 
            compute_every                   period to compute gradient similarities, value of 1 will compute every step. 
            batch_size                      batch_size to use for training; if batch_size > len(self.valid) then batches won't be used. 
            num_workers                     number of workers to use with the validation dataloader 
            grad_params                     list of parameter names to be used to compute gradient similarity; If None, then all parameters are used. 
            verbose                         if True, will print epoch loss and accuracy 
            use_cuda                        use cuda-enabled GPU if available

        output: 
            data values     (N_t,N_p)       N_t is number of training samples, N_p is the number of sampled data values (int(num_epochs/compute_every)) 
        '''
        model = self.model
        optim = optim(model.parameters(), lr=lr)
        
        # if no grad params are provided
        if grad_params is None: 
            grad_params = [n for n,_ in model.named_parameters()]

        valid_loader = torch.utils.data.DataLoader(self.valid, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        train_loader = torch.utils.data.DataLoader(self.train, batch_size=512, num_workers=num_workers, shuffle=False)

        # place holder for vals 
        sim_vals = torch.zeros((len(self.train), int(num_epochs*len(valid_loader)/compute_every)))
        grad_mag = torch.zeros((len(self.train), int(num_epochs*len(valid_loader)/compute_every)))

        if torch.cuda.is_available() & use_cuda: 
            device = 'cuda'
        else: 
            device = 'cpu'
        if verbose: print('using device:', device)

        model.to(device)

        nn = 0
        iter = 0
        elapsed = []
        for epoch in range(num_epochs): 
            losses = []
            for x_valid, y_valid in valid_loader:
                model.train()
                x_valid, y_valid = x_valid.to(device), y_valid.to(device)

                if iter%compute_every==0: 
                    tic = time.time()
                    # step 1: compute validation gradient
                    optim.zero_grad()
                    yhat_valid = model(x_valid)
                    loss = crit(yhat_valid, y_valid)
                    grad_valid = self._get_grad(loss, [p for n,p in model.named_parameters() if n in grad_params])

                    # step 2: for each train sample 
                    j = 0
                    # to speed up per-sample gradient calculations 
                    fmodel, params, buffers = make_functional_with_buffers(model)
                    ft_compute_sample_grad = get_per_sample_grad_func(crit, fmodel)
                    for x_batch, y_batch in train_loader: 
                        _tic = time.time()
                        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                        ft_per_sample_grads = ft_compute_sample_grad(params, buffers, x_batch, y_batch)
                        batch_grads = torch.cat([_g.view(x_batch.size(0), -1) for _g, (n,p) in zip(ft_per_sample_grads, model.named_parameters()) if n in grad_params], dim=1)
                        batch_sim = similarity(grad_valid.unsqueeze(0).expand(x_batch.size(0), -1), batch_grads).detach().cpu()
                        grad_norm = torch.norm(batch_grads, p=2, dim=1).detach().cpu()
                        sim_vals[j:int(j + x_batch.size(0)), nn] = batch_sim 
                        grad_mag[j:int(j + x_batch.size(0)), nn] = grad_norm
                        print(f'epoch {epoch} || computing sample gradients... [{j}/{len(self.train)}] ({time.time()-_tic:.4f}s/batch)', end='\r')
                        j += x_batch.size(0)
                    nn += 1
                    elapsed.append(time.time()-tic)

                # step 3: optimize on validation gradient 
                loss.backward()
                optim.step()
                losses.append(loss.item())
                iter += 1

            print(f'epoch {epoch} || avg loss: {np.mean(losses):.2f} || grad time elapsed: {np.mean(elapsed):.1f} s', end='\r')

        return sim_vals.detach().cpu().numpy(), grad_mag.detach().cpu().numpy()



def get_per_sample_grad_func(crit, fmodel): 

    def compute_loss_stateless_model (params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)
        predictions = fmodel(params, buffers, batch) 
        loss = crit(predictions, targets)
        return loss

    ft_compute_grad = grad(compute_loss_stateless_model)
    return vmap(ft_compute_grad, in_dims=(None, None, 0, 0), randomness='different')

