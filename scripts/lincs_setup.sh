#!/bin/sh

# Download datasets and document sources
# author: nathaniel evans
# email: evansna@ohsu.edu
# --------------------------------------------------------------------

# PATHS # 
## OUT
ROOT=../data/
APC_DIR=$ROOT/processed/
LINCS_DIR=$ROOT/processed/

## data source url
l1000_phaseII_lvl34_meta=https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/instinfo_beta.txt
l1000_phaseII_compoundinfo=https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/compoundinfo_beta.txt
l1000_phaseII_cellineinfo=https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/cellinfo_beta.txt
l1000_phaseII_geneinfo=https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/geneinfo_beta.txt
lvl4_lincs=https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level4/level4_beta_trt_cp_n1805898x12328.gctx

# Level 5 LINCS 
LINCS_LEVEL5_META=https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/siginfo_beta.txt
LINCS_LEVEL5_CP=https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level5/level5_beta_trt_cp_n720216x12328.gctx

########
echo 'beginning lincs setup...'
echo 'downloading lincs data...'

[ ! -d "$ROOT" ] && mkdir -p $ROOT

[ ! -f "$ROOT/compoundinfo_beta.txt" ] && wget $l1000_phaseII_compoundinfo -O $ROOT/compoundinfo_beta.txt
[ ! -f "$ROOT/cellinfo_beta.txt" ] && wget $l1000_phaseII_cellineinfo -O $ROOT/cellinfo_beta.txt
[ ! -f "$ROOT/geneinfo_beta.txt" ] && wget $l1000_phaseII_geneinfo -O $ROOT/geneinfo_beta.txt
[ ! -f "$ROOT/instinfo_beta.txt" ] && wget $l1000_phaseII_lvl34_meta -O $ROOT/instinfo_beta.txt
[ ! -f "$ROOT/level4_beta_trt_cp_n1805898x12328.gctx" ] && wget $lvl4_lincs -O $ROOT/level4_beta_trt_cp_n1805898x12328.gctx

#LEVEL 5 
[ ! -f "$ROOT/level5_beta_trt_cp_n720216x12328.gctx" ] && wget $LINCS_LEVEL5_CP -O $ROOT/level5_beta_trt_cp_n720216x12328.gctx
[ ! -f "$ROOT/siginfo_beta.txt" ] && wget $LINCS_LEVEL5_META -O $ROOT/siginfo_beta.txt

echo 'downloads complete.'

echo 'calculating APC values...this may take a while...'

# calculate APC values 
[ ! -d "$APC_DIR" ] && mkdir $APC_DIR
[ ! -f "$APC_DIR/APC.csv" ] && python ./calc_APC.py --data $ROOT --out $APC_DIR

echo 'APC calc complete.'

echo 'starting lincs preprocessing...'

# NOTE: remove `--zscore` to use raw data.
[ ! -d "$LINCS_DIR/data.h5" ] && python ./lincs_preproc.py --data $ROOT --apc_dir $APC_DIR --out $LINCS_DIR --zscore

echo 'lincs setup complete.'

