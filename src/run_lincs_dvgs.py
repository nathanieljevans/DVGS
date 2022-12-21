'''
usage: 

# should take ~8 hours on GPU with less than 4 GB vram 
```
(dvgs) $ python run_lincs_dvgs.py --data ../data/ --out ../lincs_results --epochs 1 --lr 1e-3 --compute_every 5 --target_batch_size 2000 --source_batch_size 100 --do 0. --num_layers 2 --latent_channels 64 --hidden_channels 500 --target_prop 0.25 --zscore_expr
```

'''
import pandas as pd 
import argparse
import numpy as np
import torch 
import os 
from AEDataset import AEDataset
from DVGS import DVGS
from AE import AE
import time 
import similarities


def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str,
                        help="path to the data dir")

    parser.add_argument("--out", type=str,
                        help="output dir; will save as rAPC.csv")

    parser.add_argument("--epochs", type=int,
                        help="number of epochs to train dvgs")           

    parser.add_argument("--lr", type=float,
                        help="learning rate") 

    parser.add_argument("--compute_every", type=int,
                        help="frequency to compute gradient similarities")

    parser.add_argument("--target_batch_size", type=int,
                        help="target (valid) batch size")  

    parser.add_argument("--source_batch_size", type=int,
                        help="source (train) batch size") 

    parser.add_argument("--do", type=float,
                        help="dropout") 

    parser.add_argument("--num_layers", type=int,
                        help="number of hidden layers in the AE encoder/decoder") 

    parser.add_argument("--latent_channels", type=int,
                        help="number of latent channels (bottleneck layer)")         

    parser.add_argument("--hidden_channels", type=int,
                        help="number of hidden channels in encoder/decoder")  

    parser.add_argument("--target_prop", type=float,
                        help="number of hidden channels in encoder/decoder [0,1]") 

    parser.add_argument('--zscore_expr', action='store_true', default=False,
                        help='whether to zscore the perturbed expression')

    return parser.parse_args()


if __name__ == '__main__': 

    args = get_args() 

    print('loading data...')

    # for testing...
    #data = torch.load(f'{args.data}/lincs/l1000_data_TEST.pt')
    #sample_ids = np.load(f'{args.data}/lincs/l1000_ids_TEST.npy', allow_pickle=True)

    data = torch.load(f'{args.data}/lincs/l1000_data_ALL.pt')
    sample_ids = np.load(f'{args.data}/lincs/l1000_ids_ALL.npy', allow_pickle=True)

    if args.zscore_expr: 
        print('z-scoring perturbed expression data...')
        data = (data - data.mean(dim=0))/data.std(dim=0)

    valid_idx = torch.tensor(np.random.randint(0,data.size(0), size=int(args.target_prop*data.size(0))), dtype=torch.long)
    train_idx = torch.tensor(np.delete(np.arange(data.size(0)), valid_idx), dtype=torch.long)

    valid_ids = sample_ids[valid_idx]
    train_ids = sample_ids[train_idx]

    x_train = data[train_idx, :]
    x_valid = data[valid_idx, :]

    print('total dataset size:', data.size(0))
    print('source dataset size:', x_train.size(0))
    print('valid dataset size:', x_valid.size(0))
    print('est. # of `source-epochs` to perform:', args.epochs*int(x_valid.size(0)/args.target_batch_size)/args.compute_every)

    train_dataset = AEDataset(x_train)
    valid_dataset = AEDataset(x_valid)

    print('initializing models...')
    model = AE(in_channels=978, latent_channels=args.latent_channels, hidden_channels=args.hidden_channels, norm=False, dropout=args.do, bias=True, act=torch.nn.Mish, num_layers=args.num_layers)
    dvgs = DVGS(train_dataset, valid_dataset, test_dataset=None, model=model)
    print('# of model params:', sum(p.numel() for p in model.parameters()))

    tic = time.time() 
    print('running dvgs...')
    run_id = dvgs.run(crit                  = torch.nn.MSELoss(), 
                      save_dir              = args.out,
                      similarity            = similarities.cosine_similarity(), 
                      optim                 = torch.optim.Adam, 
                      lr                    = args.lr, 
                      num_epochs            = args.epochs, 
                      compute_every         = args.compute_every, 
                      target_batch_size     = args.target_batch_size, 
                      source_batch_size     = args.source_batch_size, 
                      num_workers           = 1, 
                      grad_params           = None, 
                      verbose               = True, 
                      use_cuda              = True)

    # to avoid mem issues jic
    N = data.size(0)
    del data 
    del x_train 
    del x_valid
    del train_dataset 
    del valid_dataset 

    vals_source = dvgs.agg(f'{args.out}/{run_id}/', reduction='mean')

    vals = np.zeros(N)
    vals[train_idx] = vals_source 

    print(f'results are saved to: {args.out}/{run_id}/agg/')

    # full dataset data values are saved in `values.npy` in order of `sample_ids.npy`
    # target data values will be zero 

    os.mkdir(f'{args.out}/{run_id}/agg/')
    np.save(f'{args.out}/{run_id}/agg/values.npy', vals)
    np.save(f'{args.out}/{run_id}/agg/sample_ids.npy', sample_ids)
    np.save(f'{args.out}/{run_id}/agg/source_ids.npy', train_ids)
    np.save(f'{args.out}/{run_id}/agg/target_ids.npy', valid_ids)
    open(f'{args.out}/{run_id}/agg/meta.log', 'w').write(str(args))

    print()
    print(f'time elapsed: {(time.time() - tic)/60:.2f} min')