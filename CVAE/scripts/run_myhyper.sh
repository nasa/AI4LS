#!/bin/bash

SRC_DIR=/home/jcasalet/nobackup/AI4LS/CVAE/src/
DATA_BASEDIR=/home/jcasalet/nobackup/AI4LS/CVAE/DATA/
#ARCHS4_DATA=archs4_pretrain.h5
ARCHS4_DATA=archs4_top1024cv.h5
#ARCHS4_DATA=archs4_top8192cv.h5
#OSDR_DATA=osdr_mouse.h5
OSDR_DATA=osdr_mouse_top1024.h5
#OSDR_DATA=osdr_mouse_top8192.h5
PRETRAIN_DATA=${DATA_BASEDIR}/$ARCHS4_DATA
FINETUNE_DATA=${DATA_BASEDIR}/$OSDR_DATA
OUTPUT_BASEDIR=/home/jcasalet/nobackup/AI4LS/CVAE/
CKPT_DIR=${OUTPUT_BASEDIR}/checkpoints
RESULT_DIR=${OUTPUT_BASEDIR}/results

EPOCHS=500
CONDITIONS='tissue study'
KL_ANNEAL_EPOCHS=100
BATCH_SIZE=64
PATIENCE=40
LATENT_DIM=128
DROPOUT=0.3
FREEZE_DECODER_EPOCHS=40
GRL_ALPHA=0
LAMBDA_CLS=4
STUDY_EMB_DIM=8
TISSUE_EMB_DIM=32

#for hidden_dims in '4096 2048 1024 512 256' '2048 1024 512 256' '1024 512 256'
#for hidden_dims in '16384 8192 4096 2048 1024 512 256' '8192 4096 2048 1024 512 256' '4096 2048 1024 512 256'
#for hidden_dims in '2048 1024 512 256' '1024 512 256' '512 256'
#for hidden_dims in '1024 512 256' 
for hidden_dims in '4096 2048 1024 512 256'
do
	echo $hidden_dims
	#for train_script in train.py loso_cv.py
	for train_script in train.py
	do
		train=$(echo $train_script | cut -d. -f1)
		echo $train_script
		#for detach in --detach_cls_head
		for detach in "" 
		do
			echo $detach
			for reinit_latent_heads in --reinit_latent_heads ""
			do
				hidden="${hidden_dims// /-}"
				experiment=${ARCHS4_DATA}_${OSDR_DATA}_hidden_${hidden}_latent_${LATENT_DIM}_tissue_emb_dim_${TISSUE_EMB_DIM}_study_emb_dim_${STUDY_EMB_DIM}_grl_alpha_${GRL_ALPHA}_lambda_cls_${LAMBDA_CLS}_reinit_latent_heads_${reinit_latent_heads}_detach_${detach}_train_${train}
				echo running experiment: $experiment

				# pretrain
				echo "#!/bin/bash" > pretrain.sh 
				echo "python -u ${SRC_DIR}/pretrain_archs4.py  --data ${PRETRAIN_DATA}  --output_dir ${CKPT_DIR}/pretrain/${experiment} --epochs $EPOCHS  --conditions $CONDITIONS  --latent_dim $LATENT_DIM  --tissue_emb_dim $TISSUE_EMB_DIM --study_emb_dim $STUDY_EMB_DIM --kl_anneal_epochs $KL_ANNEAL_EPOCHS  --patience $PATIENCE  --batch_size $BATCH_SIZE  --beta 0.01 --hidden_dims $hidden_dims" >> pretrain.sh

				chmod +x ./pretrain.sh
				PRETRAIN_ID=$(sbatch -t 1-0 --gres=gpu:1 --parsable ./pretrain.sh) 

				# finetune
				echo "#!/bin/bash" > finetune.sh
				echo "python -u ${SRC_DIR}/$train_script  --data $FINETUNE_DATA  --pretrain_checkpoint ${CKPT_DIR}/pretrain/${experiment}/pretrain_best.pt  --output_dir ${CKPT_DIR}/finetune/${experiment}  --conditions $CONDITIONS  --latent_dim $LATENT_DIM  --tissue_emb_dim $TISSUE_EMB_DIM --study_emb_dim $STUDY_EMB_DIM --lr 5e-5  --new_lr_mult 10.0  --beta 0.005  --lambda_cls $LAMBDA_CLS  --patience 80  --dropout $DROPOUT  --freeze_decoder_epochs $FREEZE_DECODER_EPOCHS  $reinit_latent_heads --hidden_dims $hidden_dims $detach " >> finetune.sh

				chmod +x ./finetune.sh
				FINETUNE_ID=$(sbatch -t 1-0 --gres=gpu:1 --dependency=afterok:$PRETRAIN_ID --parsable ./finetune.sh)

				echo "#!/bin/bash" > check_auroc.sh
				echo "python -u ${SRC_DIR}/check_auroc.py  --checkpoint ${CKPT_DIR}/finetune/${experiment}/best_model.pt  --data $FINETUNE_DATA  --output_dir ${RESULT_DIR}/check_auroc/finetune/${experiment} --hidden_dims $hidden_dims" >> check_auroc.sh
				chmod +x check_auroc.sh
				sbatch --dependency=afterok:$FINETUNE_ID --gres=gpu:1 ./check_auroc.sh
				
				echo "#!/bin/bash" > inference.sh
				echo "python -u ${SRC_DIR}/inference.py  --checkpoint ${CKPT_DIR}/finetune/${experiment}/best_model.pt  --data $FINETUNE_DATA  --output_dir ${RESULT_DIR}/inference/finetune/${experiment}/  --skip_enrichment --hidden_dims $hidden_dims" >> inference.sh
				chmod +x inference.sh
				sbatch --dependency=afterok:$FINETUNE_ID --gres=gpu:1 ./inference.sh 

				echo "#!/bin/bash" > check_latent_dims.sh
				echo "python -u ${SRC_DIR}/check_latent_dims.py  --checkpoint ${CKPT_DIR}/finetune/${experiment}/best_model.pt  --data $FINETUNE_DATA  --output_dir ${RESULT_DIR}/check_latent_dims/finetune/${experiment}/ --hidden_dims $hidden_dims" >> check_latent_dims.sh
				chmod +x check_latent_dims.sh
				sbatch --dependency=afterok:$FINETUNE_ID --gres=gpu:1 ./check_latent_dims.sh

				echo "#!/bin/bash" > visualize_latent.sh
				echo "python ${SRC_DIR}/visualize_latent.py  --checkpoint ${CKPT_DIR}/finetune/${experiment}/best_model.pt  --data $FINETUNE_DATA  --output_dir ${RESULT_DIR}/visualize/${experiment}/ --hidden_dims $hidden_dims" >> visualize_latent.sh
				chmod +x visualize_latent.sh
				sbatch --dependency=afterok:$FINETUNE_ID --gres=gpu:1 ./visualize_latent.sh

				echo "#!/bin/bash" > latent_gene_predictor.sh
				echo "python -u ${SRC_DIR}/latent_gene_predictor.py --checkpoint ${CKPT_DIR}/finetune/${experiment}/best_model.pt --data $FINETUNE_DATA --output_dir ${RESULT_DIR}/latent_gene_pred/${experiment}/by_tissue/ --by_tissue --min_samples 5 --validate --hidden_dims $hidden_dims" >> latent_gene_predictor.sh
				chmod +x latent_gene_predictor.sh
				sbatch --dependency=afterok:$FINETUNE_ID --gres=gpu:1 ./latent_gene_predictor.sh

				echo "#!/bin/bash" > pretrain_latent_ellipses.sh
				echo "python -u ${SRC_DIR}/visualize_latent_ellipses.py --ckpt ${CKPT_DIR}/pretrain/${experiment}/pretrain_best.pt --data $PRETRAIN_DATA --output ${RESULT_DIR}/latent_ellipses/pretrain/${experiment}/"  >> pretrain_latent_ellipses.sh
				chmod +x pretrain_latent_ellipses.sh
				sbatch --dependency=afterok:$PRETRAIN_ID --gres=gpu:1 ./pretrain_latent_ellipses.sh

				echo "#!/bin/bash" > finetune_latent_ellipses.sh
				echo "python -u ${SRC_DIR}/visualize_latent_ellipses.py --ckpt ${CKPT_DIR}/finetune/${experiment}/best_model.pt --data $PRETRAIN_DATA --output ${RESULT_DIR}/latent_ellipses/finetune/${experiment}/"  >> finetune_latent_ellipses.sh
				chmod +x finetune_latent_ellipses.sh
				sbatch --dependency=afterok:$FINETUNE_ID --gres=gpu:1 ./finetune_latent_ellipses.sh


			done
		done
	done
done

