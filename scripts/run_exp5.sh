#!/bin/sh

# dvgs ~ 170 min [actual]
# dvrl ~ 20s/iter * 1000 iter ~ 333 min 

python run_valuation.py --config ../configs/exp5.py --method dvgs
python run_valuation.py --config ../configs/exp5.py --method dvrl
python run_valuation.py --config ../configs/exp5.py --method dshap
python run_valuation.py --config ../configs/exp5.py --method loo
python run_valuation.py --config ../configs/exp5.py --method random