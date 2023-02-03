# Data Valuation with Gradient Similarity (DVGS)



## environment 

we recommend using the [mamba](https://mamba.readthedocs.io/en/latest/installation.html) package manager, although `conda` should also work.  

```bash 
$ mamba env create -f environment.yml
$ conda activate dvgs 
(dvgs) $
```

## example notebook 

To explore dvgs, dvrl or dshap... use `example.ipynb` 

## run an experiment 

see `/configs` for details. 

```bash
(dvgs) $ python run_valuation --config ../configs/exp1.py --method dvgs 
```

## `LINCS` data valuation 

Should take about ~8 hours on a GPU (requires less than 4GB vram).

```bash

(dvgs) $ ./lincs_setup.sh 
(dvgs) $ cd src 
(dvgs) $ python run_lincs_dvgs.py --data ../data/ --out ../lincs_results --epochs 25 --lr 1e-3 --compute_every 5 --target_batch_size 2000 --source_batch_size 50 --do 0.2 --num_layers 2 --latent_channels 64 --hidden_channels 500

```