import torch 
import sys 
import copy 
from sklearn.metrics import roc_auc_score, r2_score
import numpy as np
from data_loading import load_tabular_data

sys.path.append('../src/')
from NN import NN 
from deprecated.NNEst import NNEst
from CNNAE import CNNAE
from MyResNet18 import MyResNet18
import similarities


##################################
# Experiment summary 
##################################

summary="This experiment measures the ability of (2) methods for capturing exogenous noise in the cifar dataset (unsupervised)."

#################
# General params 
#################

# options: "adult", "blog", "cifar10", 'cifar10-unsupervised'
dataset = "cifar10-unsupervised"

# learning algorithm to use 
model = CNNAE(in_channels=3, hidden_channels=32, latent_channels=10, act=torch.nn.Mish)

# label corruption of endogenous variable (y)
endog_noise = 0.

# max guassian noise rate (standard deviation) in exogenous variable (x) 
# each sample will be assigned a random gaussian std noise rate sampled from a uniform dist between [0, exog_noise]
exog_noise = 3.

## number of training/source observations 
train_num = 40000 

# number of validation/target observations 
valid_num = 10000

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
                "metric"        : lambda y,yhat: r2_score(y.ravel(), yhat.ravel()) , 

                # filter quantiles 
                "qs"            : np.linspace(0., 0.5, 10), 

                # mini-batch size for SGD 
                "batch_size"    : 1000,

                # learning rate 
                "lr"            : 1e-3, 

                # number of training epochs 
                "epochs"        : 25, 

                # number of technical replicates at each quantile (re-init of model)
                "repl"          : 2,

                # whether to re-initialize model parameters prior to each train - if repl > 1, should be True
                "reset_params"  : True
            }

####################################################################
# Data valuation with reinfocement learning (DVRL) params 
####################################################################

resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights="ResNet18_Weights.DEFAULT")
resnet.fc = torch.nn.Linear(512, 100)
estimator = NNEst(xin=100, yin=20, y_cat_dim=100, out_channels=1, num_layers=4, hidden_channels=100, norm=False, dropout=0.0, bias=True, act=torch.nn.ReLU, cnn=resnet)

dvrl_init = { 
                "predictor"         : copy.deepcopy(model), 
                "estimator"         : estimator, 
                "problem"           : 'classification',
                "include_marginal"  : True
            }


dvrl_run = { 
                "perf_metric"            : 'mse', 
                "crit_pred"              : torch.nn.MSELoss(), 
                "outer_iter"             : 1000, 
                "inner_iter"             : 25, 
                "outer_batch"            : 50000, 
                "inner_batch"            : 1000, 
                "estim_lr"               : 1e-4, 
                "pred_lr"                : 1e-3, 
                "moving_average_window"  : 50,
                "entropy_beta"           : 0., 
                "entropy_decay"          : 1.,
                "fix_baseline"           : True,
                "noise_labels"           : None,
                "use_cuda"               : True,
                "center_logits"          : True
            }

####################################################################
# Data valuation with gradient similarity (DVGS) params
#################################################################### 

# only relevant for classification problems and assumes cross entropy loss; will override 'target_crit' and 'source_crit'
dvgs_balance_class_weights = False

# remove interim gradient similarities 
dvgs_clean_gradient_sims = True

dvgs_kwargs = { 
                "target_crit"           : torch.nn.MSELoss(), 
                "source_crit"           : torch.nn.MSELoss(),
                "num_restarts"          : 1,
                "save_dir"              : f'{out_dir}/dvgs/',
                "similarity"            : similarities.cosine_similarity(),
                "optim"                 : torch.optim.Adam, 
                "lr"                    : 1e-3, 
                "num_epochs"            : 25, 
                "compute_every"         : 1, 
                "source_batch_size"     : 50, 
                "target_batch_size"     : 2000,
                "grad_params"           : None, 
                "verbose"               : True, 
                "use_cuda"              : True
            }
