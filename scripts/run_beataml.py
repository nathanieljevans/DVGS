import pandas as pd 
from matplotlib import pyplot as plt
import numpy as np 
import seaborn as sbn 
import pickle as pkl
from matplotlib import cm
import torch 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import percentileofscore
from shutil import rmtree
import argparse 
import uuid 
import copy 
import os

import sys 
sys.path.append('../src/')
from utils import get_filtered_scores
from AE import AE
from NN import NN
from LM import LM
from DVGS import DVGS
import similarities

###############
###############

dvgs_kwargs = lambda: { 
                    "target_crit"           : torch.nn.MSELoss(), 
                    "source_crit"           : torch.nn.MSELoss(),
                    "num_restarts"          : 1,
                    "similarity"            : similarities.C_dist(), 
                    "optim"                 : np.random.choice([torch.optim.Adam], size=1).item(), 
                    "lr"                    : np.random.choice([1e-3], size=1).item(), 
                    "num_epochs"            : np.random.choice([100], size=1).item(), 
                    "compute_every"         : 1, 
                    "source_batch_size"     : 100,
                    "target_batch_size"     : np.random.choice([25, 50, 75, 100], size=1).item(),
                    "grad_params"           : None, 
                    "verbose"               : False, 
                    "use_cuda"              : True,
                    "wd"                    : 0
                }

nn_kwargs = lambda: {
                "out_channels"      : 1, 
                "hidden_channels"   : np.random.randint(10, 100, size=1).item(), 
                "num_layers"        : np.random.choice([1,2], size=1).item(), 
                "norm"              : np.random.choice([True, False]).item(), 
                "dropout"           : np.random.choice([0, 0.1], size=1).item(), 
                "bias"              : True, 
                "act"               : np.random.choice([torch.nn.Mish, torch.nn.ELU], size=1).item()
                }

ae_kwargs = lambda: { 
                "hidden_channels"   : np.random.randint(50, 100, size=1).item(), 
                "latent_channels"   : np.random.randint(10, 50, size=1).item(), 
                "num_layers"        : 1, 
                "norm"              : np.random.choice([True, False]).item(), 
                "dropout"           : np.random.choice([0, 0.1], size=1).item(), 
                "bias"              : True, 
                "act"               : np.random.choice([torch.nn.Mish, torch.nn.ELU], size=1).item()
            }

filter_kwargs = lambda: {
                        "crit"          : torch.nn.MSELoss(),
                        "metric"        : lambda x,y: np.corrcoef(x.ravel(),y.ravel())[0,1], # r2_score(x.ravel(),y.ravel()) #lambda x,y: np.mean((x - y)**2)**0.5 , # lambda x,y: r2_score(x,y, multioutput='variance_weighted')
                        "qs"            : np.linspace(0., 0.9, 10), 
                        "batch_size"    : np.random.choice([25, 50, 75, 100], size=1).item(),
                        "lr"            : np.random.choice([1e-3], size=1).item(), 
                        "epochs"        : np.random.choice([100], size=1).item(), 
                        "repl"          : 1,
                        "reset_params"  : True,
                        'return_all_scores' : False
                    }
##############
##############


def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default='../../beatAML/', 
                        help="path to the data dir")

    parser.add_argument("--out", type=str, default='../results/beatAML/',
                        help="output dir")
    
    parser.add_argument("--use_percentile", action='store_true',
                        help="output dir")
    
    parser.add_argument("--variance_filter", type=int, default=None,
                        help="number of top variance features to select")
    
    parser.add_argument("--zscore", action='store_true',
                        help="output dir")
    
    parser.add_argument("--supervised", action='store_true',
                        help="output dir")
    
    parser.add_argument("--verbose", action='store_true',
                        help="output dir")
    
    parser.add_argument("--use_nn", action='store_true',
                        help="whether to use a NN (default: linear model)")
    
    parser.add_argument("--drug", type=str, default=None,
                        help="output dir")
    
    parser.add_argument("--num_pca_comp", type=int, default=25,
                        help="output dir")
    
    parser.add_argument("--nfolds", type=int, default=25,
                        help="output dir")
    
    parser.add_argument("--nsplits", type=int, default=25,
                        help="output dir")
    
    parser.add_argument("--source_p", type=float, default=0.6,
                        help="output dir")
    
    parser.add_argument("--test_p", type=float, default=0.1,
                        help="output dir")

    parser.add_argument("--pca", action='store_true',
                        help="output dir")
    

    
    args = parser.parse_args()

    return args

def load_data_unsupervised(args): 
    expr = pd.read_csv(f'{args.data}/beataml_waves1to4_norm_exp_dbgap.txt', sep='\t')
    
    X = expr.values[:, 4:].T.astype(float)
    ids = expr.columns[4:].tolist()

    return X, ids

def load_data_supervised(args): 

    expr = pd.read_csv(f'{args.data}/beataml_waves1to4_norm_exp_dbgap.txt', sep='\t')

    resp = pd.read_csv(f'{args.data}/beataml_probit_curve_fits_v4_dbgap.txt', sep='\t')

    drRes = resp[lambda x: (x.inhibitor == args.drug) & ~x.dbgap_rnaseq_sample.isna()]
    drRes = drRes[['dbgap_rnaseq_sample', 'inhibitor', 'auc']]

    if drRes.shape[0] == 0: raise Exception(f'Drug has no observations ({args.drug})')

    expr2 = expr.drop(['display_label', 'description', 'biotype'], axis=1).T
    expr2.columns = expr2.iloc[0]
    expr2 = expr2.iloc[1:]

    drRes = drRes.merge(expr2, left_on='dbgap_rnaseq_sample', right_index=True)

    X = drRes.values[:, 3:].astype(float)
    y = drRes.auc.values.astype(float)
    ids = drRes.dbgap_rnaseq_sample.values

    return X,y,ids

def normalize(X_train, X_test, y_train, y_test, args): 

    if args.variance_filter:
        # for EDA select high var samples 
        gene_idxs = np.argsort(X_train.std(axis=0).ravel())[-args.variance_filter:]
        #gene_idxs = (X_train.std(axis=0) > args.variance_filter).nonzero()[0]
        X_train = X_train[:, gene_idxs]
        X_test = X_test[:, gene_idxs]

    if args.zscore:
        # zscore 
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if args.pca: 
        reducer = PCA(args.num_pca_comp)
        X_train = reducer.fit_transform(X_train)
        X_test = reducer.transform(X_test)
    
    if args.supervised: 

        mean = y_train.mean() 
        std = y_train.std() + 1e-8

        # always z-score target 
        y_train = y_train - mean 
        y_train = y_train / std

        y_test = y_test - mean 
        y_test = y_test / std

    else:
        y_train = None
        y_test = None

    return X_train, X_test, y_train, y_test
            
def run_split(X, y, ids, args): 

    # random [statistically] unique run id 
    uid = uuid.uuid4()

    if args.supervised: 
        
        X_target, X_source, y_target, y_source, target_ids, source_ids = train_test_split(X, y, ids, test_size=args.source_p)

        # TODO: should we be normalizing based on target or source??? Currently using target. I think that makes sense. 
        X_target, X_source, y_target, y_source = normalize(X_train=X_target, X_test=X_source, y_train=y_target, y_test=y_source, args=args)

        X_target = torch.tensor(X_target, dtype=torch.float32)
        X_source = torch.tensor(X_source, dtype=torch.float32)
        y_source = torch.tensor(y_source, dtype=torch.float32).view(-1, 1)
        y_target = torch.tensor(y_target, dtype=torch.float32).view(-1, 1)

        if args.use_nn: 
            model = NN(in_channels=X_target.size(1), **nn_kwargs())
        else: 
            model = LM(in_channels=X_target.size(1), out_channels=1)
    else: 
        
        X_target, X_source, target_ids, source_ids = train_test_split(X, ids, test_size=args.source_p)

        X_target, X_source, y_target, y_source = normalize(X_target, X_source, None, None, args)

        X_target = torch.tensor(X_target, dtype=torch.float32)
        X_source = torch.tensor(X_source, dtype=torch.float32)
        y_target = X_target
        y_source = X_source

        model = AE(in_channels=X_target.size(1), **ae_kwargs())

    nparams = sum([p.numel() for p in model.parameters()])
    if args.verbose: print('\n\t# params:', nparams)

    dvgs = DVGS(x_source         = X_source,
                y_source         = y_source, 
                x_target         = X_target,  
                y_target         = y_target, 
                model            = model)

    if os.path.exists(f'{args.out}/{uid}'): rmtree(f'{args.out}/{uid}')

    kwargs = dvgs_kwargs()
    run_id = dvgs.run(**kwargs, save_dir=args.out, uid=uid)

    vals = dvgs.agg(f'{args.out}/{run_id}/').ravel()

    # convert vals to percentile for easier interpretation and aggregation 
    if args.use_percentile: vals = [percentileofscore(vals, x) for x in vals]

    res = pd.DataFrame({'source_id':source_ids, 'vals':vals})

    hparams = pd.DataFrame({k:str(v) for k,v in kwargs.items()}, index=[0])

    return res, hparams

def eval(X, y, ids, res, args, randomize=False):

    perfs_low = []; perfs_high = [] 

    for i in range(args.nfolds):
        if args.verbose: print(f'running eval fold: {i}', end='\r')

        if args.supervised: 

            X_train, X_test, y_train, y_test, source_ids, test_ids = train_test_split(X, y, ids, test_size=args.test_p)
            X_train, X_test, y_train, y_test = normalize(X_train, X_test, y_train, y_test, args)

            X_test = torch.tensor(X_test, dtype=torch.float32)
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

            if args.use_nn: 
                model = NN(in_channels=X_train.size(1), **nn_kwargs())
            else: 
                model = LM(in_channels=X_train.size(1), out_channels=1)

        else: 

            X_train, X_test, y_train, y_test, source_ids, test_ids = train_test_split(X, y, ids, test_size=args.test_p)
            X_train, X_test, _, _ = normalize(X_train, X_test, None, None, args)

            X_test = torch.tensor(X_test, dtype=torch.float32)
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = X_train
            y_test = X_test

            model = AE(in_channels=X_train.size(1), **ae_kwargs())

        vals = res.set_index('source_id').loc[source_ids].vals_mean.values

        if randomize: vals = np.random.permutation(vals)

        perf_filter_low, perf_filter_high = get_filtered_scores(x_train       = X_train,
                                                                y_train       = y_train,
                                                                x_test        = X_test,
                                                                y_test        = y_test, 
                                                                vals          = vals, 
                                                                model         = model, **filter_kwargs())

        perfs_low.append(np.array(perf_filter_low))
        perfs_high.append(np.array(perf_filter_high))

    return perfs_low, perfs_high


if __name__ == '__main__': 
    
    print()
    args = get_args()
    print(args)
    print()

    if os.path.exists(args.out): 
        if args.verbose: print(f'output directory exists ({args.out}), erasing contents...', end='')
        rmtree(args.out)
        if args.verbose: print('complete')
    
    os.mkdir(args.out)

    if args.supervised: 
        X,y,ids = load_data_supervised(args=args)
    else: 
        X,ids = load_data_unsupervised(args=args)
        y = copy.deepcopy(X) 

    if args.verbose: print('X size:', X.shape)
    if args.verbose: print('y size:', y.shape)

    print('Running data valuation splits...')
    ress = []
    hparamss = {}
    for i in range(args.nsplits): 
        print('running split:', i, end='\r')
        res, hparams = run_split(X, y, ids, args)
        ress.append(res)
        hparamss[i] = hparams

    res = ress[0].rename({'vals':'vals0'}, axis=1)
    for i, res2 in enumerate(ress[1:]):
        res2 = res2.rename({'vals':f'vals{i+1}'}, axis=1) 
        res = res.merge(res2, on='source_id', how='outer')

    res = res.assign(vals_mean=[np.nanmean([res.values[i, 1:]]) for i in range(res.shape[0])])

    print('Running performance evaluation folds...')
    perfs_low, perfs_high = eval(X, y, ids, res, args)

    print('Running performance evaluation folds (random data values)...')
    perfs_low_random, perfs_high_random = eval(X, y, ids, res, args, randomize=True)

    print('Saving results to file...', end='')

    res.to_csv(args.out + '/data_values.csv', sep=',', index=False)
    
    with open(args.out + '/args.txt', 'w') as f: 
        f.write(str(args))

    perfs = {'perfs_low':perfs_low, 
             'perfs_high':perfs_high,
             'perfs_low_random':perfs_low_random,
             'perfs_high_random':perfs_high_random}
    
    #if args.verbose: print('\n', perfs)

    with open(args.out + '/perfs.pkl', 'wb') as f: 
        pkl.dump(obj=perfs, file=f)

    try: 
        with open(args.out + '/hparams.pkl', 'wb') as f: 
            pkl.dump(obj=hparamss, file=f)
    except: 
        print('saving hyper parameters failed.')

    if args.supervised: 
        # TODO: save latent embeddings... 
        pass

    print('save complete.')    
    print()





    