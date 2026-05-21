#!/bin/bash

# 100 spaceflight liver samples
python /home/jcasalet/nobackup/CVAE/src/generate.py \
    --checkpoint /home/jcasalet/nobackup/CVAE/checkpoints/v9/best_model.pt \
    --data /home/jcasalet/nobackup/CVAE/DATA/subset_final.h6 \
    --n 100 \
    --tissue Liver --strain C57BL/6J --sex Female \
    --flight 1 --euth Isoflurane \
    --seed 42 \
    --output_dir /home/jcasalet/nobackup/CVAE/results/generate/v9


# conditions.csv:
# n,tissue,strain,sex,flight,euth
# 100,Liver,C57BL/6J,Female,1,Isoflurane
# 100,Liver,C57BL/6J,Female,0,Isoflurane
# 50,Soleus,C57BL/6J,Male,1,Ketamine_Xylazine

python /home/jcasalet/nobackup/CVAE/src/generate.py \
    --checkpoint /home/jcasalet/nobackup/CVAE/checkpoints/v9/best_model.pt \
    --data /home/jcasalet/nobackup/CVAE/DATA/subset_final.h6 \
    --config /home/jcasalet/nobackup/CVAE/config/conditions.csv \
    --seed 42 \
    --output_dir /home/jcasalet/nobackup/CVAE/results/generate/v9
