'''

'''

import argparse 
import uuid 
import torch 
from sklearn.utils.class_weight import compute_class_weight
import time
import numpy as np
import pandas as pd
import hashlib
import os
import copy 

import sys 
sys.path.append('../src/')
from utils import load_data, load_config, Logger, get_filtered_scores
from DVGS import DVGS
from DVRL import DVRL
from DShap import DShap
from LOO import LOO

def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str,
                        help="path to config file")

    parser.add_argument("--method", type=str,
                        help="data valuation method to perform")

    return parser.parse_args()


if __name__ == '__main__': 

    args = get_args()

    # LOAD CONFIG 
    config = load_config(args.config)

    # random [statistically] unique run id 
    uid = uuid.uuid4()

    # fork console output to log file
    sys.stdout = Logger(f'{config.out_dir}/logs/{uid}/' )

    print('-'*100)
    print('-'*100)
    print('args:')
    print(args)
    print('-'*100)
    print('-'*100)

    # LOAD DATA  
    print('loading data...')
    x_train, y_train, x_valid, y_valid, x_test, y_test, exog_noise, endog_noise = load_data(dataset     = config.dataset, 
                                                                                            train_num   = config.train_num, 
                                                                                            valid_num   = config.valid_num, 
                                                                                            exog_noise  = config.exog_noise, 
                                                                                            endog_noise = config.endog_noise, 
                                                                                            save_dir    = f'{config.out_dir}/data/{uid}', 
                                                                                            clean_up    = config.cleanup_data)

    print('train size:')    
    print('\tx:', x_train.shape)                                                                                    
    print('\ty:', y_train.shape)   
    print('valid size:')    
    print('\tx:', x_valid.shape)                                                                                    
    print('\ty:', y_valid.shape)   
    print('test size:')    
    print('\tx:', x_test.shape)                                                                                    
    print('\ty:', y_test.shape)   
    print()                                                                                 

    # VALUATION  
    print('running data valuation...')
    tic = time.time() 
    if args.method == 'dvgs': 

        dvgs = DVGS(x_source         = x_train,
                    y_source         = y_train, 
                    x_target         = x_train,  
                    y_target         = y_train, 
                    model            = copy.deepcopy(config.model))

        if config.dvgs_balance_class_weights: 
            class_weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=list(np.sort(np.unique(y_train.detach().numpy()))), y=y_train.detach().numpy().ravel()), dtype=torch.float).to('cuda')
            CEL = torch.nn.CrossEntropyLoss(weight=class_weights) 
            config.dvgs_kwargs["target_crit"] = lambda x,y: CEL(x,y.squeeze(1).type(torch.long))
            config.dvgs_kwargs["source_crit"] = lambda x,y: torch.nn.functional.cross_entropy(x,y.squeeze(1).type(torch.long))

        run_id = dvgs.run(**config.dvgs_kwargs)
        vals = dvgs.agg(f'{config.out_dir}/dvgs/{run_id}/').ravel()

        if config.dvgs_clean_gradient_sims:   
            dvgs.clean(f'{config.out_dir}/dvgs/{run_id}/')

    elif args.method == 'dvrl': 
        
        dvrl = DVRL(x_train       = x_train,
                    y_train       = y_train,
                    x_valid       = x_valid,
                    y_valid       = y_valid, **config.dvrl_init)

        CEL = torch.nn.CrossEntropyLoss() 
        print()

        tic = time.time() 
        vals = dvrl.run(**config.dvrl_run).detach().cpu().numpy().ravel()

    elif args.method == 'dshap': 

        dshap =  DShap(x_train       = x_train,
                       y_train       = y_train,
                       x_valid       = x_valid,
                       y_valid       = y_valid, **config.dshap_init)

        vals = dshap.TMC(**config.dshap_run).ravel()

    elif args.method == 'loo': 
        
        loo = LOO(x_train       = x_train,
                  y_train       = y_train,
                  x_valid       = x_valid,
                  y_valid       = y_valid, **config.loo_kwargs)

        vals = np.array(loo.run()).ravel()

    elif args.method == 'random': 

        vals = np.random.randn(x_train.size(0))

    else: 
        raise Exception('unrecognized argument `method`; options: [dvgs, dvrl, dshap, loo, random]')

    print(f'time elapsed: {(time.time() - tic)/60:.2f} min')
    print()

    print('filtering data and recording model performance...')
    # FILTER & TRAIN  
    perf_filter_low, perf_filter_high = get_filtered_scores(x_train       = x_train,
                                                            y_train       = y_train,
                                                            x_test        = x_test,
                                                            y_test        = y_test, 
                                                            vals          = vals, **config.filter_kwargs)
    print()

    # SAVE 
    print(f'saving results to file ({config.out_dir}/results.csv)')
    print(f'data_values, exog_noise, endog_noise will be saved in the respective data_values/uid dir')

    res = pd.DataFrame({'uid'               : uid,
                        'config'            : args.config,
                        'config-checksum'   : hashlib.md5(open(args.config,'rb').read()).hexdigest(),
                        'method'            : args.method,
                        'perf_filter_low'   : [perf_filter_low],
                        'perf_filter_high'  : [perf_filter_high], 
                        'runtime_s'         : time.time() - tic}, index=[0])

    if not os.path.exists(config.out_dir + '/data_values/'):
        os.mkdir(config.out_dir + '/data_values/')
    
    os.mkdir(config.out_dir + '/data_values/' + str(uid))
    np.save(config.out_dir + '/data_values/' + str(uid) + '/data_values.npy', vals)
    np.save(config.out_dir + '/data_values/' + str(uid) + '/endog_noise.npy', endog_noise)
    np.save(config.out_dir + '/data_values/' + str(uid) + '/exog_noise.npy', exog_noise)

    if os.path.exists(config.out_dir + '/results.csv'): 
        res.to_csv(config.out_dir + '/results.csv', mode='a', sep='\t', index=False, header=False)
    else: 
        res.to_csv(config.out_dir + '/results.csv', mode='w', sep='\t', index=False)
    print()

    print('data valuation complete.')