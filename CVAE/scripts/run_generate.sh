#!/bin/bash

# 100 spaceflight liver samples
python generate.py \
    --checkpoint checkpoints_v6/best_model.pt \
    --data subset_final.h5 \
    --n 100 \
    --tissue Liver --strain C57BL/6J --sex Female \
    --flight 1 --euth Isoflurane \
    --seed 42


# conditions.csv:
# n,tissue,strain,sex,flight,euth
# 100,Liver,C57BL/6J,Female,1,Isoflurane
# 100,Liver,C57BL/6J,Female,0,Isoflurane
# 50,Soleus,C57BL/6J,Male,1,Ketamine_Xylazine

python generate.py \
    --checkpoint checkpoints_v6/best_model.pt \
    --data subset_final.h5 \
    --config conditions.csv \
    --seed 42
