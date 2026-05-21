#!/bin/bash

#python -u train.py --data DATA/subset_final.h5 --output_dir ./gpu_checkpoints
#python -u train.py --data DATA/subset_final.h5 --lambda_adv 1.0 --grl_alpha 2.0 --output_dir ./checkpoints_v2
#python -u train.py --data DATA/subset_final.h5 --lambda_adv 0.0 --grl_alpha 0.0 --patience 40 --output_dir ./checkpoints_v3
python -u /home/jcasalet/nobackup/CVAE/src/train.py \
	--data /home/jcasalet/nobackup/CVAE/DATA/subset_final.h6 \
	--output_dir /home/jcasalet/nobackup/CVAE/checkpoints/v9 \
	--beta 0.005 \
        --kl_anneal_epochs 300 \
        --lambda_cls 2.0 \
	--latent_dim 64 \
        --patience 30 \
	--dropout 0.3

