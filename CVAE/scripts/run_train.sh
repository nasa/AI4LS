#!/bin/bash

python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/pretrain_archs4.py \
	--data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/archs4_pretrain_no_unknown_no_other.h5 \
	--output_dir /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/new_tissue_v3 \
	--beta 0.005 \
	--conditions tissue \
        --kl_anneal_epochs 300 \
	--latent_dim 64 \
        --patience 40 \
	--dropout 0.3

