#!/bin/sh


num_repl=5

for i in $(seq 1 $num_repl); 
do 
    python run_valuation.py --config ../configs/exp2.py --method dvgs
    python run_valuation.py --config ../configs/exp2.py --method dvrl
    python run_valuation.py --config ../configs/exp2.py --method dshap
    python run_valuation.py --config ../configs/exp2.py --method loo
    python run_valuation.py --config ../configs/exp2.py --method random
done 