'''

'''

import torch
import numpy as np
import time 

class DVGS(): 

    def __init__(self, train_dataset, valid_dataset, test_dataset, model): 
        ''''''
        self.train = train_dataset 
        self.valid = valid_dataset 
        self.test = test_dataset 
        self.model = model


    def run(self, crit, similarity=torch.nn.CosineSimilarity(dim=0), optim=torch.optim.Adam, lr=1e-2, num_epochs=100, compute_every=1, batch_size=512, num_workers=1, grad_params=None, verbose=True, use_cuda=True): 
        '''
        trains the model and returns data values 

        args: 
            crit                            loss criteria, e.g., torch.nn.BCELoss()
            similarity                      similarity metric that takes two 1d torch vectors and computes a scalar metric; common options: torch.nn.CosineSimilarity(dim=0), torch.cdist, torch.dot
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

        valid_loader = torch.utils.data.DataLoader(self.valid, batch_size=batch_size, num_workers=num_workers)

        # place holder for vals 
        sim_vals = np.zeros((len(self.train), int(num_epochs*len(valid_loader)/compute_every)))

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
                x_valid, y_valid = x_valid.to(device), y_valid.to(device)

                if epoch%compute_every==0: 
                    tic = time.time()
                    # step 1: compute validation gradient
                    optim.zero_grad()
                    yhat_valid = model(x_valid)
                    loss = crit(yhat_valid, y_valid)
                    loss.backward()
                    grad_valid = torch.cat([p.grad.view(-1) for n,p in model.named_parameters() if n in grad_params])

                    # step 2: for each train sample 
                    for j in range(len(self.train)): 
                        optim.zero_grad()
                        x_j, y_j = self.train.__getitem__(j)
                        x_j, y_j = x_j.to(device), y_j.to(device)
                        
                        # a) compute sample gradient 
                        yhat_j = model(x_j)
                        loss_j = crit(yhat_j, y_j)
                        loss_j.backward()
                        grad_j = torch.cat([p.grad.view(-1) for n,p in model.named_parameters() if n in grad_params])
                        
                        # b compute cosine similarity with validation gradient 
                        sim_vals[j, nn] = similarity(grad_valid, grad_j).item()
                    nn += 1
                    elapsed.append(time.time()-tic)

                # step 3: optimize on validation gradient 
                optim.zero_grad()
                yhat_valid = model(x_valid)
                loss = crit(yhat_valid, y_valid)
                loss.backward()
                optim.step()
                losses.append(loss.item())

                iter += 1

            print(f'epoch {epoch} || avg loss: {np.mean(losses):.2f} || grad time elapsed: {np.mean(elapsed):.1f} s', end='\r')

        return sim_vals

