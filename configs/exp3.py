import torch 
import torchvision
import sys 
import copy 
from sklearn.metrics import roc_auc_score
import numpy as np
from data_loading import load_tabular_data

sys.path.append('../src/')
from NN import NN 
from Estimator import Estimator
from MyResNet import MyResNet
from CNN import CNN
import similarities


##################################
# Experiment summary 
##################################

summary="This experiment measures the ability of (2) methods for capturing label corruption in the cifar10 dataset."

#################
# General params 
#################

# options: "adult", "blog", "cifar10",
dataset = "cifar10"

# encode x into reduced representation;
encoder_model = MyResNet('inception', None, dropout=0.).eval()

# transformations to apply to cifar 
transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(299),
                torchvision.transforms.CenterCrop(299),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

# learning algorithm to use 
#model = MyResNet('resnet50', 10, dropout=0.1)
model = NN(in_channels      = 2048, 
           out_channels     = 10, 
           num_layers       = 2, 
           hidden_channels  = 100, 
           norm             = True, 
           dropout          = 0.5, 
           bias             = True, 
           act              = torch.nn.Mish, 
           out_fn           = None)

# label corruption of endogenous variable (y)
endog_noise = 0.2

# guassian noise rate (standard deviation) in exogenous variable (x) 
exog_noise = 0.

## number of training/source observations 
train_num = 4000 

# number of validation/target observations 
valid_num = 1000

# output paths 
out_dir = '../results/exp3/'

# whether to delete the data on disk after reading into memory 
cleanup_data = False

##################################
# Filtered performance params
##################################

filter_kwargs = {
                # model to be trained; will be re-initialized every run 
                "model"         : copy.deepcopy(model), 

                # optimizer loss criteria 
                "crit"          : lambda x,y: torch.nn.functional.cross_entropy(x,y.squeeze(1)),

                # reported performance metric  
                "metric"        : lambda y,yhat: (1.*(yhat.argmax(axis=1).ravel() == y.ravel())).mean() , 
                #"metric"        : lambda y,yhat: roc_auc_score(y, torch.softmax(torch.tensor(yhat), dim=-1), multi_class='ovr') , 

                # filter quantiles 
                "qs"            : np.linspace(0., 0.5, 10), 

                # mini-batch size for SGD  
                "batch_size"    : 256,

                # learning rate 
                "lr"            : 1e-3, 

                # number of training epochs 
                "epochs"        : 100, 

                # number of technical replicates at each quantile (re-init of model)
                "repl"          : 2,

                # whether to re-initialize model parameters prior to each train - if repl > 1, should be True
                "reset_params"  : True
            }

####################################################################
# Data valuation with reinfocement learning (DVRL) params 
####################################################################

estimator = Estimator(xin               = 2048, 
                      yin               = 20, 
                      y_cat_dim         = 10, 
                      num_layers        = 5, 
                      hidden_channels   = 100, 
                      norm              = False, 
                      dropout           = 0., 
                      act               = torch.nn.ReLU)

dvrl_init = { 
                "predictor"         : copy.deepcopy(model), 
                "estimator"         : estimator, 
                "problem"           : 'classification',
                "include_marginal"  : True
            }


dvrl_run = { 
                "perf_metric"            : 'acc', 
                "crit_pred"              : lambda yhat,y: torch.nn.functional.cross_entropy(yhat, y.squeeze(1)), 
                "outer_iter"             : 2000, 
                "inner_iter"             : 100,  # 250
                "outer_batch"            : 4000, 
                "inner_batch"            : 256, 
                "estim_lr"               : 1e-2, 
                "pred_lr"                : 1e-3, 
                "moving_average_window"  : 50,
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
                "target_crit"           : lambda yhat,y: torch.nn.functional.cross_entropy(yhat,y.squeeze(1).type(torch.long)), 
                "source_crit"           : lambda yhat,y: torch.nn.functional.cross_entropy(yhat,y.squeeze(1).type(torch.long)),
                "num_restarts"          : 1,
                "save_dir"              : f'{out_dir}/dvgs/',
                "similarity"            : similarities.cosine_similarity(),
                "optim"                 : torch.optim.Adam, 
                "lr"                    : 1e-3, 
                "num_epochs"            : 100, 
                "compute_every"         : 1, 
                "source_batch_size"     : 100, 
                "target_batch_size"     : 500,
                "grad_params"           : None, 
                "verbose"               : True, 
                "use_cuda"              : True
            }
