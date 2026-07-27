#!/bin/bash
#python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/finetune_zeropad.py \
#    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
#    --pretrain_checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/pretrain/zeropad/pretrain_best.pt \
#    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/zeropad/ \
#    --freeze_decoder_epochs 0 \
#    --lr 5e-5 \
#    --beta 0.005 \
#    --kl_anneal_epochs 200 \
#    --lambda_cls 2.0 \
#    --patience 40 \
#    --dropout 0.3 \
#    --lr 1e-4

#python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/train.py \
#    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
#    --pretrain_checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/pretrain/zeropad/pretrain_best.pt \
#    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/v5/ \
#    --lr 5e-5 \
#    --new_lr_mult 20.0 \
#    --beta 0.005 \
#    --kl_anneal_epochs 200 \
#    --lambda_cls 5.0 \
#    --freeze_decoder_epochs 0 \
#    --patience 9000 \
#    --dropout 0.3 \
#    --epochs 1000 \
#    --tissue_emb_dim 16 \
#    --strain_emb_dim 4 \
#    --sex_emb_dim 2 \
#    --study_emb_dim 4 \
#    --euth_emb_dim 2

python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/train.py \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --pretrain_checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/pretrain/tissue_64_32/pretrain_best.pt \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/tissue_64_32_noreinit \
    --conditions tissue \
    --latent_dim 64 \
    --tissue_emb_dim 32 \
    --lr 5e-5 \
    --new_lr_mult 10.0 \
    --beta 0.005 \
    --lambda_cls 4.0 \
    --patience 80 \
    --dropout 0.3 \
    --grl_alpha 0.0 \
    --freeze_decoder_epochs 20
    #--reinit_latent_heads \
