import torch 
import torchvision
import sys 
import copy 
from sklearn.metrics import roc_auc_score, r2_score
import numpy as np
from data_loading import load_tabular_data

sys.path.append('../src/')
from NN import NN 
from CNN import CNN
from CNNAE import CNNAE
from Estimator import Estimator
import similarities
from utils import train_model


##################################
# Experiment summary 
##################################

summary="This experiment measures the ability of (2) methods for capturing exogenous noise in the cifar dataset (unsupervised)."

#################
# General params 
#################

# options: "adult", "blog", "cifar10", 'cifar10-unsupervised'
dataset = "cifar10-unsupervised"

encoder_model = None 
transforms =  torchvision.transforms.Compose([
                torchvision.transforms.Resize(32),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2469, 0.2434, 0.2615])
            ])

# learning algorithm to use 
model = CNNAE(in_channels=3, hidden_channels=32, latent_channels=10, act=torch.nn.Mish, dropout=0.)

# label corruption of endogenous variable (y)
endog_noise = 0.

# max guassian noise rate (standard deviation) in exogenous variable (x) 
# each sample will be assigned a random gaussian std noise rate sampled from a uniform dist between [0, exog_noise]
exog_noise = 5.

## number of training/source observations 
train_num = 20000 

# number of validation/target observations 
valid_num = 2000

# output paths 
out_dir = '../results/exp7/'

# whether to delete the data on disk after reading into memory 
cleanup_data = False

##################################
# Filtered performance params
##################################

filter_kwargs = {
                # model to be trained; will be re-initialized every run 
                "model"         : copy.deepcopy(model), 

                # optimizer loss criteria 
                "crit"          : torch.nn.MSELoss(),

                # reported performance metric  
                #"metric"        : lambda y,yhat: -(np.mean((y - yhat)**2)**(0.5)) ,  # -rmse, bigger is better
                "metric"        : lambda y,yhat: r2_score(y.ravel(), yhat.ravel()) , 

                # filter quantiles 
                "qs"            : np.linspace(0., 0.9, 10), 

                # mini-batch size for SGD 
                "batch_size"    : 256,

                # learning rate 
                "lr"            : 1e-3, 

                # number of training epochs 
                "epochs"        : 100, 

                # number of technical replicates at each quantile (re-init of model)
                "repl"          : 3,

                # whether to re-initialize model parameters prior to each train - if repl > 1, should be True
                "reset_params"  : True
            }

####################################################################
# Data valuation with reinfocement learning (DVRL) params 
####################################################################


estimator = Estimator(xin=50, 
                      yin=0, 
                      y_cat_dim=50, 
                      num_layers=3, 
                      hidden_channels=100, 
                      norm=False, 
                      dropout=0.0, 
                      act=torch.nn.ReLU, 
                      cnn=CNN(in_conv=3, out_conv=64, out_channels=50, kernel_size=3, act=torch.nn.ReLU))

dvrl_init = { 
                "predictor"         : copy.deepcopy(model), 
                "estimator"         : estimator, 
                "problem"           : 'regression',
                "include_marginal"  : False
            }


dvrl_run = { 
                "perf_metric"            : 'r2', 
                "crit_pred"              : torch.nn.MSELoss(), 
                "outer_iter"             : 2000, 
                "inner_iter"             : 100, 
                "outer_batch"            : 3000, 
                "inner_batch"            : 256, 
                "estim_lr"               : 5e-3, 
                "pred_lr"                : 1e-3, 
                "moving_average_window"  : 100,
                "fix_baseline"           : False,
                "use_cuda"               : True,
            }

####################################################################
# Data valuation with gradient similarity (DVGS) params
#################################################################### 

# only relevant for classification problems and assumes cross entropy loss; will override 'target_crit' and 'source_crit'
dvgs_balance_class_weights = False

# remove interim gradient similarities 
dvgs_clean_gradient_sims = False

dvgs_kwargs = { 
                "target_crit"           : torch.nn.MSELoss(), 
                "source_crit"           : torch.nn.MSELoss(),
                "num_restarts"          : 1,
                "save_dir"              : f'{out_dir}/dvgs/',
                "similarity"            : similarities.cosine_similarity(),
                "optim"                 : torch.optim.Adam, 
                "lr"                    : 1e-3, 
                "num_epochs"            : 1000, 
                "compute_every"         : 1, 
                "source_batch_size"     : 50, 
                "target_batch_size"     : 1000,
                "grad_params"           : None, 
                "verbose"               : True, 
                "use_cuda"              : True
            }
