# Data Valuation with Gradient Similarity (DVGS)

High-quality data is crucial for accurate machine learning and actionable analytics, however, mislabeled or noisy data is a common problem in many domains. Distinguishing low- from high-quality data can be challenging, often requiring expert knowledge and considerable manual intervention. Data Valuation algorithms are a class of methods that seek to quantify the value of each sample in a dataset based on its contribution or importance to a given predictive task. These data values have shown an impressive ability to identify mislabeled observations, and filtering low-value data can boost machine learning performance. In this work, we present a simple alternative to existing methods, termed Data Valuation with Gradient Similarity (DVGS). This approach can be easily applied to any gradient descent learning algorithm, scales well to large datasets, and performs comparably or better than baseline valuation methods for tasks such as corrupted label discovery and noise quantification. We evaluate the DVGS method on tabular, image and RNA expression datasets to show the effectiveness of the method across domains. Our approach has the ability to rapidly and accurately identify low-quality data, which can reduce the need for expert knowledge and manual intervention in data cleaning tasks.

```
@misc{evans2024data,
      title={Data Valuation with Gradient Similarity}, 
      author={Nathaniel J. Evans and Gordon B. Mills and Guanming Wu and Xubo Song and Shannon McWeeney},
      year={2024},
      eprint={2405.08217},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## examples

To explore dvgs, dvrl or dshap... use `example.ipynb` 

## run an experiment 

see `/configs` for details. 

```bash
(dvgs) $ python run_valuation --config ../configs/exp1.py --method dvgs 
```

## `LINCS` data valuation 

```bash
(dvgs) $ ./lincs_setup.sh 
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
