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

# References 

In this project we implement versions of: 

## Data Shapley (DShap)

[arxiv](https://arxiv.org/abs/1904.02868)
[proceedings](https://proceedings.mlr.press/v97/ghorbani19c/ghorbani19c.pdf)
[github](https://github.com/amiratag/DataShapley)

```
@inproceedings{ghorbani2019data,
  title={Data Shapley: Equitable Valuation of Data for Machine Learning},
  author={Ghorbani, Amirata and Zou, James},
  booktitle={International Conference on Machine Learning},
  pages={2242--2251},
  year={2019}
}
```

## Data Valuation with Reinforcement Learning (DVRL) 

[arxiv](https://arxiv.org/abs/1909.11671)
[proceedings](https://proceedings.mlr.press/v119/yoon20a.html)
[github](https://github.com/google-research/google-research/tree/master/dvrl)

```
@misc{https://doi.org/10.48550/arxiv.1909.11671,
  doi = {10.48550/ARXIV.1909.11671},
  
  url = {https://arxiv.org/abs/1909.11671},
  
  author = {Yoon, Jinsung and Arik, Sercan O. and Pfister, Tomas},
  
  keywords = {Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Data Valuation using Reinforcement Learning},
  
  publisher = {arXiv},
  
  year = {2019},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
