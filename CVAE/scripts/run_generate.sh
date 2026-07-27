#!/bin/bash
# 100 spaceflight liver samples
python /home/jcasalet/nobackup/AI4LS/CVAE/src/generate.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/v2/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --n 100 \
    --tissue Liver --strain C57BL/6J --sex Female \
    --flight 1 --euth Isoflurane \
    --seed 42 \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/generate/finetune/v2/


# conditions.csv:
# n,tissue,strain,sex,flight,euth
# 100,Liver,C57BL/6J,Female,1,Isoflurane
# 100,Liver,C57BL/6J,Female,0,Isoflurane
# 50,Soleus,C57BL/6J,Male,1,Ketamine_Xylazine

python /home/jcasalet/nobackup/AI4LS/CVAE/src/generate.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/v2/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --config /home/jcasalet/nobackup/AI4LS/CVAE/config/conditions.csv \
    --seed 42 \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/generate/finetune/v2/
