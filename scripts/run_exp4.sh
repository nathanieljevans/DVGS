#!/bin/sh

python run_valuation.py --config ../configs/exp4.py --method dvgs
python run_valuation.py --config ../configs/exp4.py --method dvrl
python run_valuation.py --config ../configs/exp4.py --method dshap
python run_valuation.py --config ../configs/exp4.py --method loo
python run_valuation.py --config ../configs/exp4.py --method random