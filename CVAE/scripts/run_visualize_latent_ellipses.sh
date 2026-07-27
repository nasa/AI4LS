#!/bin/bash
PRETRAIN_FILE=$1
CKPT_DIR=$2
python -u ../src/visualize_latent_ellipses.py --data $PRETRAIN_FILE --ckpt $CKPT_DIR 
