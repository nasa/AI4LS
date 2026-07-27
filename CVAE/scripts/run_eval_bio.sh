#!/bin/bash
python eval_biology_retention.py \
    --probe_data held_out_pretrain_slice.h5 \
    --pretrain_checkpoint pretrain_best.pt \
    --finetuned_checkpoint checkpoints/best_model.pt \
    --latent_dim 32 --hidden_dims 512 256
