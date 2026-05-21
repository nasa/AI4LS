#!/bin/bash

# Strain × flight interaction — does spaceflight affect C57BL/6J liver differently than BALB/c liver?
python src/whatif.py \
    --checkpoint checkpoints/v9/best_model.pt \
    --data DATA/subset_final.h6 \
    --mode interaction \
    --tissue Liver \
    --interact_condition strain \
    --interact_value_a C57BL/6J \
    --interact_value_b BALB/c \
    --output_dir whatif_results/v9/a

# Euthanasia artifact analysis — which spaceflight genes are robust vs euthanasia-confounded?
python src/whatif.py \
    --checkpoint checkpoints/v9/best_model.pt \
    --data DATA/subset_final.h6 \
    --mode artifact \
    --tissue Liver \
    --euth_a Isoflurane \
    --euth_b CO2 \
    --output_dir whatif_results/v9/b

# what if CO2-euthanized liver samples had been euthanized with isoflurane?
python /home/jcasalet/nobackup/CVAE/src/whatif.py \
    --checkpoint /home/jcasalet/nobackup/CVAE/checkpoints/v9/best_model.pt \
    --data /home/jcasalet/nobackup/CVAE/DATA/subset_final.h6 \
    --mode counterfactual \
    --tissue Liver \
    --euth CO2 \
    --change_condition euth \
    --change_from CO2 \
    --change_to Isoflurane \
    --output_dir /home/jcasalet/nobackup/CVAE/results/whatif/v9/1


# just tissue, any strain/sex
python /home/jcasalet/nobackup/CVAE/src/whatif.py \
    --checkpoint /home/jcasalet/nobackup/CVAE/checkpoints/v9/best_model.pt \
    --data /home/jcasalet/nobackup/CVAE/DATA/subset_final.h6 \
    --mode population \
    --tissue Soleus \
    --output_dir /home/jcasalet/nobackup/CVAE/results/whatif/v9/2

# tissue + euthanasia method
python /home/jcasalet/nobackup/CVAE/src/whatif.py \
    --checkpoint /home/jcasalet/nobackup/CVAE/checkpoints/v9/best_model.pt \
    --data /home/jcasalet/nobackup/CVAE/DATA/subset_final.h6 \
    --mode population \
    --tissue Kidney \
    --euth Isoflurane \
    --output_dir /home/jcasalet/nobackup/CVAE/results/whatif/v9/3

# no filters at all — uses all 2080 samples
python /home/jcasalet/nobackup/CVAE/src/whatif.py \
    --checkpoint /home/jcasalet/nobackup/CVAE/checkpoints/v9/best_model.pt \
    --data /home/jcasalet/nobackup/CVAE/DATA/subset_final.h6 \
    --mode population\
    --output_dir /home/jcasalet/nobackup/CVAE/results/whatif/v9/4
