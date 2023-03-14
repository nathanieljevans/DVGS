#!/bin/sh

num_repl=10

for i in $(seq 1 $num_repl); 
do 
    python run_valuation.py --config ../configs/exp10.py --method dvgs
    python run_valuation.py --config ../configs/exp10.py --method apc
    #python run_valuation.py --config ../configs/exp9.py --method dvrl
done