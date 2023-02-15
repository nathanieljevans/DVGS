import torch 
import sys 
import copy 
from sklearn.metrics import roc_auc_score
import numpy as np
from data_loading import load_tabular_data

sys.path.append('../src/')
from NN import NN 
from deprecated.NNEst import NNEst
import similarities


##################################
# Experiment summary 
##################################

summary="This experiment measures the ability of (4) methods for capturing exogenous noise in the adult dataset (supervised)."

#################
# General params 
#################

# options: "adult", "blog", "cifar10",
dataset = "adult"

# learning algorithm to use 
model = NN(in_channels=108, out_channels=2, num_layers=2, hidden_channels=100, norm=True, dropout=0.5, bias=True, act=torch.nn.Mish, out_fn=None)

# label corruption of endogenous variable (y)
endog_noise = 0.

# guassian noise rate (standard deviation) in exogenous variable (x) 
exog_noise = 3.

## number of training/source observations 
train_num = 1000 

# number of validation/target observations 
valid_num = 400

# output paths 
out_dir = '../results/exp4/'

# whether to delete the data on disk after reading into memory 
cleanup_data = True

##################################
# Filtered performance params
##################################

filter_kwargs = {
                # model to be trained; will be re-initialized every run 
                "model"         : copy.deepcopy(model), 

                # optimizer loss criteria 
                "crit"          : lambda x,y: torch.nn.functional.cross_entropy(x,y.squeeze(1).type(torch.long)),

                # reported performance metric  
                "metric"        : lambda y,yhat: roc_auc_score(y, yhat[:, 1]) , 

                # filter quantiles 
                "qs"            : np.linspace(0., 0.5, 10), 

                # mini-batch size for SGD 
                "batch_size"    : 250,

                # learning rate 
                "lr"            : 1e-4, 

                # number of training epochs 
                "epochs"        : 200, 

                # number of technical replicates at each quantile (re-init of model)
                "repl"          : 2,

                # whether to re-initialize model parameters prior to each train - if repl > 1, should be True
                "reset_params"  : True
            }

##################################
# Leave-one-out (LOO) params 
##################################

loo_kwargs = {
                # learning algorithm
                "model"           : copy.deepcopy(model),

                # performance metric 
                "metric"        : lambda y,yhat: roc_auc_score(y, yhat[:, 1]),    

                # loss criteria
                "crit"          : lambda x,y: torch.nn.functional.cross_entropy(x,y.squeeze(1).type(torch.long)),  

                # optimizer 
                "optim"         : torch.optim.Adam, 

                # number of optimizer iterations 
                "epochs"        : 100, 

                # optimizer step size 
                "lr"            : 1e-3,

                # batch size for stochastic gradient descent 
                "batch_size"    : 250, 

                # whether to use nvidia GPU, if available 
                "use_cuda"      : True, 

                # console verbosity 
                "verbose"       : True, 

                # number of "baseline" (e.g., trained on all data) models trained to get "baseline" performance estimate
                "baseline_repl" : 10,

                # number of technical replicates for each LOO model 
                "n_repl"        : 1
            }

##################################
# Data Shapley (dshap) params 
##################################

dshap_init = {
                # learning algorithm
                "model"           : copy.deepcopy(model),

                # loss criteria 
                "crit"            : lambda x,y: torch.nn.functional.cross_entropy(x,y.squeeze(1).type(torch.long)),

                # performance metric 
                "perf_metric"     : lambda y, yhat: roc_auc_score(y, yhat[:, 1]),

                # number of epochs to train each model for 
                "epochs"          : 100,

                # Truncation criteria; stops the chain when performance is within `tol` of vD (valid perf) for T iterations in a row.
                "tol"             : 0.03,

                # optimizer algorithm 
                "optim"           : torch.optim.Adam,

                # optimizer learning-rate/step-size 
                "lr"              : 1e-3,

                # console verbosity 
                "verbose"         : True
            }

dshap_run = {   
                # max number of iterations to run TMC shapley 
                "max_iterations"      : 1000, 

                # minimum number of iterations to run TMC shapley
                "min_iterations"      : 200, 

                # to use GPU if available 
                "use_cuda"            : True, 

                # 
                "T"                   : 5, 

                # TMC stopping criteria; iterative runs have rank correlation greater than `stopping criteria`, e.g., when ranking stops changing significantly 
                "stopping_criteria"   : 0.999
            }


####################################################################
# Data valuation with reinfocement learning (DVRL) params 
####################################################################

estimator = NNEst(xin=108, yin=4, y_cat_dim=200, out_channels=1, num_layers=4, hidden_channels=100, norm=False, dropout=0.0, bias=True, act=torch.nn.ReLU)

dvrl_init = { 
                "predictor"         : copy.deepcopy(model), 
                "estimator"         : estimator, 
                "problem"           : 'classification',
                "include_marginal"  : True
            }


dvrl_run = { 
                "perf_metric"            : 'auroc', 
                "crit_pred"              : lambda x,y: torch.nn.functional.cross_entropy(x,y.squeeze(1).type(torch.long)), 
                "outer_iter"             : 1000, 
                "inner_iter"             : 100, 
                "outer_batch"            : 1000, 
                "inner_batch"            : 250, 
                "estim_lr"               : 1e-4, 
                "pred_lr"                : 5e-3, 
                "moving_average_window"  : 50,
                "entropy_beta"           : 0.1, 
                "entropy_decay"          : 0.999,
                "fix_baseline"           : True,
                "noise_labels"           : None,
                "use_cuda"               : True,
                "center_logits"          : True
            }

####################################################################
# Data valuation with gradient similarity (DVGS) params
#################################################################### 

# only relevant for classification problems and assumes cross entropy loss; will override 'target_crit' and 'source_crit'
dvgs_balance_class_weights = True 

# remove interim gradient similarities 
dvgs_clean_gradient_sims = True

dvgs_kwargs = { 
                "target_crit"           : None, 
                "source_crit"           : None,
                "num_restarts"          : 3,
                "save_dir"              : f'{out_dir}/dvgs/',
                "similarity"            : similarities.cosine_similarity(),
                "optim"                 : torch.optim.SGD, 
                "lr"                    : 5e-2, 
                "num_epochs"            : 200, 
                "compute_every"         : 1, 
                "source_batch_size"     : 500, 
                "target_batch_size"     : 400,
                "grad_params"           : None, 
                "verbose"               : True, 
                "use_cuda"              : True
            }
