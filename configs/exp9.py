import torch 
import sys 
import copy 
from sklearn.metrics import roc_auc_score, r2_score
import numpy as np
from data_loading import load_tabular_data

sys.path.append('../src/')
from Estimator import Estimator
from AE import AE
import similarities


##################################
# Experiment summary 
##################################

summary="This experiment measures the ability of (1) methods for data valuation on the LINCS L1000 dataset"

#################
# General params 
#################

# options: "adult", "blog", "cifar10", 'cifar10-unsupervised', 'lincs-hi-apc-target'
dataset = "lincs"

transforms = None 
encoder_model = None

# learning algorithm to use 
model = AE(in_channels      = 978, 
           num_layers       = 2, 
           hidden_channels  = 256, 
           latent_channels  = 64,
           norm             = False, 
           dropout          = 0.0, 
           act              = torch.nn.Mish)

# label corruption of endogenous variable (y)
endog_noise = 0.

# max guassian noise rate (standard deviation) in exogenous variable (x) 
# each sample will be assigned a random gaussian std noise rate sampled from a uniform dist between [0, exog_noise]
exog_noise = 0.

## number of training/source observations 
train_num = -1 

# number of validation/target observations 
valid_num = 5000

# output paths 
out_dir = '../results/exp9/'

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
                "metric"        : lambda x,y: r2_score(x,y, multioutput='uniform_average') , 

                # filter quantiles 
                "qs"            : np.linspace(0., 0.95, 10), 

                # mini-batch size for SGD 
                "batch_size"    : 1024,

                # learning rate 
                "lr"            : 1e-3, 

                # number of training epochs 
                "epochs"        : 50, 

                # number of technical replicates at each quantile (re-init of model)
                "repl"          : 1,

                # whether to re-initialize model parameters prior to each train - if repl > 1, should be True
                "reset_params"  : True
            }

####################################################################
# Data valuation with reinfocement learning (DVRL) params 
####################################################################

estimator = Estimator(xin               = 978, 
                      yin               = 0, 
                      y_cat_dim         = 10, 
                      num_layers        = 3, 
                      hidden_channels   = 100, 
                      norm              = False, 
                      dropout           = 0., 
                      act               = torch.nn.ReLU)
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
                "outer_batch"            : 10000, 
                "inner_batch"            : 500, 
                "estim_lr"               : 1e-3, 
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
dvgs_clean_gradient_sims = True

dvgs_kwargs = { 
                "target_crit"           : torch.nn.MSELoss(), 
                "source_crit"           : torch.nn.MSELoss(),
                "num_restarts"          : 1,
                "save_dir"              : f'{out_dir}/dvgs/',
                "similarity"            : similarities.cosine_similarity(),
                "optim"                 : torch.optim.Adam, 
                "lr"                    : 1e-3, 
                "num_epochs"            : 500, 
                "compute_every"         : 1, 
                "source_batch_size"     : 250, 
                "target_batch_size"     : 5000,
                "grad_params"           : None, 
                "verbose"               : True, 
                "use_cuda"              : True
            }
