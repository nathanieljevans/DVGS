'''

usage: 

```
(dvgs) $ python run_adult_label_corruption.py --out ../adult_results/ --noise_rate 0.2 --train_num 1000 --valid_num 400 --num_layers 2 --hidden_channels 100 --do 0.25 --epochs 100 --lr 1e-3 --compute_every 1 --target_batch_size 400 --source_batch_size 1000 --dvrl_outer_iter 2000 --dvrl_inner_iter 100 --dvrl_outer_batch_size 1000 --dvrl_inner_batch_size 256 --dvrl_est_lr 1e-2 --dvrl_pred_lr 1e-3 --dvrl_T 20 --dvrl_entropy_beta 1e-5 --dvrl_entropy_decay 1. --dvrl_est_num_layers 2 --dvrl_est_hidden_channels 300 --dvrl_est_do 0. --dshap_epochs 100 --dshap_tol 0.03 --dshap_lr 1e-3
```


'''

import argparse 
import numpy as np
from data_loading import load_tabular_data, preprocess_data, corrupt_label
from GenDataset import GenDataset
from uuid import uuid4
from os import mkdir
from os.path import exists 
from DVGS import DVGS
from DVRL import DVRL
import DShap
from utils import get_corruption_scores
from NN import NN
import time 
import torch
import similarities
import pickle as pkl
from sklearn.metrics import roc_auc_score
from utils import get_filtered_scores
import copy

def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--out", type=str,
                        help="output dir")

    parser.add_argument("--noise_rate", type=float,
                        help="proportion of labels to corrupt")

    parser.add_argument("--train_num", type=int,
                        help="number of training (source) samples")

    parser.add_argument("--valid_num", type=int,
                        help="number of validation (target) samples")

    parser.add_argument("--num_layers", type=int,
                        help="number of NN layers")                

    parser.add_argument("--hidden_channels", type=int,
                        help="number of NN hidden channels") 

    parser.add_argument("--do", type=float,
                        help="dropout")

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

    parser.add_argument("--dvrl_est_hidden_channels", type=int,
                        help="[DVRL] estimator hidden channels")

    parser.add_argument("--dvrl_est_num_layers", type=int,
                        help="[DVRL] estimator number of layers")

    parser.add_argument("--dvrl_est_do", type=float,
                        help="[DVRL] estimator dropout")

    parser.add_argument("--dvrl_outer_iter", type=int,
                        help="[DVRL] outer loop iterations")

    parser.add_argument("--dvrl_inner_iter", type=int,
                        help="[DVRL] inner loop iterations")

    parser.add_argument("--dvrl_outer_batch_size", type=int,
                        help="[DVRL] outer loop batch size")

    parser.add_argument("--dvrl_inner_batch_size", type=int,
                        help="[DVRL] inner loop batch size")

    parser.add_argument("--dvrl_est_lr", type=float,
                        help="[DVRL] estimator learning rate")

    parser.add_argument("--dvrl_pred_lr", type=float,
                        help="[DVRL] predictor learning rate")

    parser.add_argument("--dvrl_T", type=int,
                        help="[DVRL] moving average window")

    parser.add_argument("--dvrl_entropy_beta", type=float,
                        help="[DVRL] starting entropy weight")

    parser.add_argument("--dvrl_entropy_decay", type=float,
                        help="[DVRL] entropy weight decay")

    parser.add_argument("--dshap_epochs", type=int,
                        help="[DVRL] number of training epochs")

    parser.add_argument("--dshap_tol", type=float,
                        help="[DShap] truncation performance tolerance")

    parser.add_argument("--dshap_lr", type=float,
                        help="[DShap] learning rate")

    return parser.parse_args()




if __name__ == '__main__': 

    args = get_args()
    run_id = uuid4()

    if not exists(args.out): mkdir(args.out)
    mkdir(f'{args.out}/{run_id}')

    print('downloading and processing data...')
    noise_idx = load_tabular_data('adult', {'train':args.train_num, 'valid':args.valid_num}, noise_rate=args.noise_rate, out=f'{args.out}/{run_id}/data_files/') # saves to disk
    x_train, y_train, x_valid, y_valid, x_test, y_test, col_names = preprocess_data('minmax', 'train.csv', 'valid.csv', 'test.csv', data=f'{args.out}/{run_id}/data_files/')
    train_dataset = GenDataset(x_train, y_train)
    test_dataset = GenDataset(x_test, y_test)
    valid_dataset = GenDataset(x_valid, y_valid)

    model = NN(in_channels=108, out_channels=2, num_layers=args.num_layers, hidden_channels=args.hidden_channels, norm=True, dropout=args.do, bias=True, act=torch.nn.Mish, out_fn=torch.nn.Softmax(dim=-1))
    dvgs = DVGS(train_dataset, valid_dataset, test_dataset, model)

    CEL = torch.nn.CrossEntropyLoss() 

    print('running dvgs...')
    tic = time.time() 
    run_id2 = dvgs.run(crit              = lambda x,y: CEL(x,y.squeeze(1).type(torch.long)), 
                    save_dir          = f'{args.out}/{run_id}/data_values/',
                    similarity        = similarities.cosine_similarity(),
                    optim             = torch.optim.Adam, 
                    lr                = args.lr, 
                    num_epochs        = args.epochs, 
                    compute_every     = args.compute_every, 
                    source_batch_size = args.source_batch_size, 
                    target_batch_size = args.target_batch_size,
                    num_workers       = 1, 
                    grad_params       = None, 
                    verbose           = True, 
                    use_cuda          = True)

    vals_dvgs = dvgs.agg(f'{args.out}/{run_id}/data_values/{run_id2}')
    np.save(f'{args.out}/{run_id}/dvgs_data_values.npy', vals_dvgs)

    print()
    print(f'time elapsed: {(time.time() - tic)/60:.2f} min')
    print('dvgs complete.')

    print('running dshap...')

    model = NN(in_channels=108, out_channels=2, num_layers=args.num_layers, hidden_channels=args.hidden_channels, norm=True, dropout=args.do, bias=True, act=torch.nn.Mish, out_fn=torch.nn.Softmax(dim=-1))
    CEL = torch.nn.CrossEntropyLoss() 

    dshap = DShap.DShap(model           = model.cpu(), 
                        crit            = lambda x,y: CEL(x,y.squeeze(1).type(torch.long)),
                        train_dataset   = copy.deepcopy(train_dataset), 
                        V               = DShap.V(copy.deepcopy(valid_dataset), roc_auc_score),
                        epochs          = args.dshap_epochs,
                        tol             = args.dshap_tol,
                        optim           = torch.optim.Adam,
                        lr              = args.dshap_lr)

    tic = time.time() 
    vals_shap = dshap.TMC(max_iterations=500, min_iterations=50, use_cuda=True, T=5, stopping_criteria=0.999)
    np.save(f'{args.out}/{run_id}/dshap_data_values.npy', vals_shap)
    print()
    print(f'time elapsed: {(time.time() - tic)/60:.2f} min')

    print('dshap complete.')

    print('running dvrl...')

    pred = NN(in_channels=108, out_channels=2, num_layers=args.num_layers, hidden_channels=args.hidden_channels, norm=True, dropout=args.do, bias=True, act=torch.nn.Mish, out_fn=torch.nn.Softmax(dim=-1))
    est = NN(in_channels=112, out_channels=1, num_layers=args.dvrl_est_num_layers, hidden_channels=args.dvrl_est_hidden_channels, norm=True, dropout=args.dvrl_est_do, bias=True, act=torch.nn.Mish, out_fn=torch.nn.Sigmoid())

    dvrl = DVRL(train_dataset, valid_dataset, test_dataset, predictor=pred, estimator=est, problem='classification')
    CEL = torch.nn.CrossEntropyLoss() 
    print()

    tic = time.time() 
    vals_dvrl = dvrl.run(perf_metric            = 'auroc', 
                        crit_pred              = lambda x,y: CEL(x,y.squeeze(1).type(torch.long)), 
                        outer_iter             = args.dvrl_outer_iter, 
                        inner_iter             = args.dvrl_inner_iter, 
                        outer_batch            = args.dvrl_outer_batch_size, 
                        outer_workers          = 1, 
                        inner_batch            = args.dvrl_inner_batch_size, 
                        estim_lr               = args.dvrl_est_lr, 
                        pred_lr                = args.dvrl_pred_lr, 
                        moving_average_window  = args.dvrl_T,
                        entropy_beta           = args.dvrl_entropy_beta, 
                        entropy_decay          = args.dvrl_entropy_decay,
                        fix_baseline           = False)

    if torch.is_tensor(vals_dvrl): 
        vals_dvrl = vals_dvrl.detach().cpu().numpy().ravel()

    np.save(f'{args.out}/{run_id}/dvrl_data_values.npy', vals_dvrl)

    print()
    print(f'time elapsed: {(time.time() - tic)/60:.2f} min')      

    print('dvrl complete.')
    
    print('calculating corruption identification rate...')
    pk, dvgs_corr, p_perfect, p_random = get_corruption_scores(vals_dvgs, noise_idx, train_size=args.train_num, noise_prop=args.noise_rate)
    pk, shap_corr, p_perfect, p_random = get_corruption_scores(vals_shap, noise_idx, train_size=args.train_num, noise_prop=args.noise_rate)
    pk, dvrl_corr, p_perfect, p_random = get_corruption_scores(vals_dvrl, noise_idx, train_size=args.train_num, noise_prop=args.noise_rate)

    pkl.dump({'pk':pk, 'dvgs_corr':dvgs_corr, 'dvrl_corr':dvrl_corr, 'shap_corr':shap_corr, 'p_perfect':p_perfect, 'p_random':p_random}, open(f'{args.out}/{run_id}/corruption_res_dict.pkl' , 'wb'))

    print('calculating filtered scores...')
    model = NN(in_channels=108, out_channels=2, num_layers=args.num_layers, hidden_channels=args.hidden_channels, norm=True, dropout=args.do, bias=True, act=torch.nn.Mish, out_fn=torch.nn.Softmax(dim=-1))
    CEL = torch.nn.CrossEntropyLoss() 
    crit = lambda x,y: CEL(x,y.squeeze(1).type(torch.long))
    metric = lambda y,yhat: roc_auc_score(y, yhat[:, 1]) 

    # Evaluation hyperparameters
    qs = np.linspace(0., 0.5, 10)
    bs = 256 
    nw = 1 
    lr = 1e-3 
    ep = 100
    rp = 3

    dvgs_low, dvgs_high = get_filtered_scores(vals_dvgs, 
                                              copy.deepcopy(model), 
                                              crit, 
                                              metric, 
                                              x_train, 
                                              y_train.reshape(-1, 1), 
                                              x_test, 
                                              y_test.reshape(-1,1), 
                                              qs=qs, 
                                              batch_size=bs, 
                                              num_workers=nw, 
                                              lr=lr, 
                                              epochs=ep, 
                                              repl=rp)

    shap_low, shap_high = get_filtered_scores(vals_shap, 
                                              copy.deepcopy(model), 
                                              crit, 
                                              metric, 
                                              x_train, 
                                              y_train.reshape(-1, 1), 
                                              x_test, 
                                              y_test.reshape(-1,1), 
                                              qs=qs, 
                                              batch_size=bs, 
                                              num_workers=nw, 
                                              lr=lr, 
                                              epochs=ep, 
                                              repl=rp)

    dvrl_low, dvrl_high = get_filtered_scores(vals_dvrl, 
                                              copy.deepcopy(model), 
                                              crit, 
                                              metric, 
                                              x_train, 
                                              y_train.reshape(-1, 1), 
                                              x_test, 
                                              y_test.reshape(-1,1), 
                                              qs=qs, 
                                              batch_size=bs, 
                                              num_workers=nw, 
                                              lr=lr, 
                                              epochs=ep, 
                                              repl=rp)

    pkl.dump({'qs':qs, 'dvgs_low':dvgs_low, 'dvgs_high':dvgs_high, 'dvrl_low':dvrl_low, 'dvrl_high':dvrl_high, 'shap_low':shap_low, 'shap_high':shap_high}, open(f'{args.out}/{run_id}/filtered_score_res_dict.pkl' , 'wb'))
    