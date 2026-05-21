#!/bin/bash

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
