'''
use: 

```
(dvgs) $ 
```
'''

import pandas as pd 
import argparse
import h5py 
import numpy as np
import torch 
import os 

def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_values", type=str,
                        help="path to the `data_values`")


    parser.add_argument("--config", type=str,
                        help="path to config file")

    return parser.parse_args()

def train_embedding_model(): 
    ''''''
    pass 



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

    # Load Data 

    # Load data values 
    
    # FOR q in qs: 
        # filter quantile data based on data values 

        # train embedding model 

        # extract drug embeddings 

        # train log reg 

        # record res 

    # save to disk 



