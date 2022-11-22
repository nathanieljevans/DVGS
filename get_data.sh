#!/bin/sh

# Download datasets and document sources
# author: nathaniel evans
# email: evansna@ohsu.edu
# --------------------------------------------------------------------
ROOT=./data/

# INFO: https://archive.ics.uci.edu/ml/datasets/adult 
ADULT_TRAIN=https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
ADULT_TEST=https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test


########
[ ! -d "$ROOT" ] && mkdir -p $ROOT

[ ! -f "$ROOT/adult.data" ] && wget $ADULT_TRAIN -O $ROOT/adult.data
[ ! -f "$ROOT/adult.test" ] && wget $ADULT_TEST -O $ROOT/adult.test

echo 'downloads complete'