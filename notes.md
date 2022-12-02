# Notes 

## 11/28/22 

Trying to get [DVRL](https://github.com/google-research/google-research/tree/master/dvrl) environment set up ... 

as suggested [here](https://github.com/google-research/google-research)... 
```bash 
$ svn export https://github.com/google-research/google-research/trunk/dvrl
```

setting up env ... had to change `sklearn` to `scikit-learn`
```bash 
$ conda create -n dvrl_tb python==3.8 
$ conda activate dvrl_tb 
(dvrl_tb) $ pip3 install -r requirements.txt 
```


adult... 
```bash 
(dvrl_tb) $ python main_data_valuation.py --data_name adult --train_no 1000 --valid_no 400 --hidden_dim 100 --comb_dim 10 --iterations 2000 --layer_number 5 --batch_size 2000 --inner_iterations 100 --batch_size_predictor 256  --learning_rate 0.01 --n_exp 5 --checkpoint_file_name ./tmp/model.ckpt
```


error: 

```
...

File "/home/teddy/local/dvrl/dvrl.py", line 28, in <module>
    from dvrl import dvrl_metrics
ImportError: cannot import name 'dvrl_metrics' from partially initialized module 'dvrl' (most likely due to a circular import) (/home/teddy/local/dvrl/dvrl.py)
```

changed line 28 to: import dvrl_metrics


error: 

tensorflow.contrib was removed from tensorflow 2.0.x

```
pip uninstall tensorflow
pip install tensorflow==1.13.2
```