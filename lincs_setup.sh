#!/bin/sh

# Download datasets and document sources
# author: nathaniel evans
# email: evansna@ohsu.edu
# --------------------------------------------------------------------

# PATHS # 
ROOT=./data/
APC_DIR=$ROOT
LINCS_DIR=$ROOT/lincs/

l1000_phaseII_lvl34_meta=https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/instinfo_beta.txt
l1000_phaseII_compoundinfo=https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/compoundinfo_beta.txt
l1000_phaseII_cellineinfo=https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/cellinfo_beta.txt
l1000_phaseII_geneinfo=https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/geneinfo_beta.txt
lvl4_lincs=https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level4/level4_beta_trt_cp_n1805898x12328.gctx

########
echo 'beginning lincs setup...'
echo 'downloading lincs data...'

[ ! -d "$ROOT" ] && mkdir -p $ROOT

[ ! -f "$ROOT/compoundinfo_beta.txt" ] && wget $l1000_phaseII_compoundinfo -O $ROOT/compoundinfo_beta.txt
[ ! -f "$ROOT/cellinfo_beta.txt" ] && wget $l1000_phaseII_cellineinfo -O $ROOT/cellinfo_beta.txt
[ ! -f "$ROOT/geneinfo_beta.txt" ] && wget $l1000_phaseII_geneinfo -O $ROOT/geneinfo_beta.txt
[ ! -f "$ROOT/instinfo_beta.txt" ] && wget $l1000_phaseII_lvl34_meta -O $ROOT/instinfo_beta.txt
[ ! -f "$ROOT/level4_beta_trt_cp_n1805898x12328.gctx" ] && wget $lvl4_lincs -O $ROOT/level4_beta_trt_cp_n1805898x12328.gctx

echo 'downloads complete.'

echo 'calculating rAPC values...this may take a while...'

# calculate APC values 
[ ! -d "$APC_DIR" ] && mkdir $APC_DIR
[ ! -f "$APC_DIR/rAPC.csv" ] && python ./src/get_lvl4_APC.py --data $ROOT --out $APC_DIR

echo 'rAPC calc complete.'

echo 'starting lincs preprocessing...'

[ ! -d "$LINCS_DIR" ] && python ./src/lincs_preproc.py --data $ROOT --out $ROOT/lincs/ --prop_test 0.25 --thresh_rAPC 0.7 --seed 0

echo 'lincs setup complete.'

