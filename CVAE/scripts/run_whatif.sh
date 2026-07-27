#!/bin/bash

# Strain × flight interaction — does spaceflight affect C57BL/6J liver differently than BALB/c liver?
python /home/jcasalet/nobackup/AI4LS/CVAE/src/whatif.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/v9/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --mode interaction \
    --tissue Liver \
    --interact_condition strain \
    --interact_value_a C57BL/6J \
    --interact_value_b BALB/c \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/whatif/v9/b

# Euthanasia artifact analysis — which spaceflight genes are robust vs euthanasia-confounded?
python /home/jcasalet/nobackup/AI4LS/CVAE/src/whatif.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/v9/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --mode artifact \
    --tissue Liver \
    --euth_a Isoflurane \
    --euth_b CO2 \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/whatif/v9/0

# what if CO2-euthanized liver samples had been euthanized with isoflurane?
python /home/jcasalet/nobackup/AI4LS/CVAE/src/whatif.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/v9/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --mode counterfactual \
    --tissue Liver \
    --euth CO2 \
    --change_condition euth \
    --change_from CO2 \
    --change_to Isoflurane \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/whatif/v9/1


# just tissue, any strain/sex
python /home/jcasalet/nobackup/AI4LS/CVAE/src/whatif.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/v9/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --mode population \
    --tissue Soleus \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/whatif/v9/2

# tissue + euthanasia method
python /home/jcasalet/nobackup/AI4LS/CVAE/src/whatif.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/v9/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --mode population \
    --tissue Kidney \
    --euth Isoflurane \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/whatif/v9/3

# no filters at all — uses all 2080 samples
python /home/jcasalet/nobackup/AI4LS/CVAE/src/whatif.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/v9/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --mode population\
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/whatif/v9/4
