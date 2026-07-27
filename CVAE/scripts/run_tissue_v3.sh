#!/bin/bash

python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/pretrain_archs4.py  \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/archs4_pretrain_no_unknown_no_other.h5 \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/pretrain/tissue_v3_redo \
    --epochs 500 \
    --conditions tissue \
    --latent_dim 32 \
    --tissue_emb_dim 32 \
    --epochs 500 \
    --kl_anneal_epochs 100 \
    --patience 40 \
    --batch_size 128 \
    --beta 0.01 \
    --dropout 0.2 \
    --lr 0.001

python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/train.py \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --pretrain_checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/pretrain/tissue_v3_redo/pretrain_best.pt \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/tissue_v3_redo/ \
    --conditions tissue \
    --latent_dim 32 \
    --tissue_emb_dim 32 \
    --lr 5e-5 \
    --epochs 500 \
    --beta 0.005 \
    --lambda_cls 2.0 \
    --lambda_adv 0.1 \
    --patience 40 \
    --new_lr_mult 10.0 \
    --dropout 0.3 \
    --grl_alpha 1.0 \
    --batch_size 32 \
    --kl_anneal_epochs 200 
    #--reinit_latent_heads \

