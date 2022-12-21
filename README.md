# Data Valuation with Gradient Similarity (DVGS)



## environment 

we recommend using the [mamba](https://mamba.readthedocs.io/en/latest/installation.html) package manager, although `conda` should also work.  

```bash 
$ mamba env create -f environment.yml
$ conda activate dvgs 
(dvgs) $
```

## example notebook 

To explore dvgs, dvrl or Dshap... use `example.ipynb` 

## adult label corruption 

```bash 
(dvgs) $ python run_adult_label_corruption.py --out ../adult_results/ --noise_rate 0.2 --train_num 1000 --valid_num 400 --num_layers 2 --hidden_channels 100 --do 0.25 --epochs 100 --lr 1e-3 --compute_every 1 --target_batch_size 400 --source_batch_size 1000 --dvrl_outer_iter 2000 --dvrl_inner_iter 100 --dvrl_outer_batch_size 1000 --dvrl_inner_batch_size 256 --dvrl_est_lr 1e-2 --dvrl_pred_lr 1e-3 --dvrl_T 20 --dvrl_entropy_beta 1e-5 --dvrl_entropy_decay 1. --dvrl_est_num_layers 2 --dvrl_est_hidden_channels 300 --dvrl_est_do 0. --dshap_epochs 100 --dshap_tol 0.03 --dshap_lr 1e-3
```

## blog label corruption 

```bash 
python blog_adult_label_corruption.py --out ../blog_results/ --noise_rate 0.2 --train_num 1000 --valid_num 400 --num_layers 2 --hidden_channels 100 --do 0.25 --epochs 100 --lr 1e-3 --compute_every 1 --target_batch_size 400 --source_batch_size 1000 --dvrl_outer_iter 2000 --dvrl_inner_iter 100 --dvrl_outer_batch_size 1000 --dvrl_inner_batch_size 256 --dvrl_est_lr 1e-2 --dvrl_pred_lr 1e-3 --dvrl_T 20 --dvrl_entropy_beta 1e-5 --dvrl_entropy_decay 1. --dvrl_est_num_layers 2 --dvrl_est_hidden_channels 300 --dvrl_est_do 0. --dshap_epochs 100 --dshap_tol 0.03 --dshap_lr 1e-3
```

## cifar label corruption 



## mnist->usps domain adaptation 



## `LINCS` data valuation 

Should take about ~8 hours on a GPU (requires less than 4GB vram).

```bash

(dvgs) $ ./lincs_setup.sh 
(dvgs) $ cd src 
(dvgs) $ python run_lincs_dvgs.py --data ../data/ --out ../lincs_results --epochs 25 --lr 1e-3 --compute_every 5 --target_batch_size 2000 --source_batch_size 50 --do 0.2 --num_layers 2 --latent_channels 64 --hidden_channels 500

```