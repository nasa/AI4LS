#!/bin/bash

# synthetic samples
CUDA_VISIBLE_DEVICES=""  python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/generate.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/v9/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --n 100 \
    --tissue Liver --strain C57BL/6J --sex Female \
    --flight 0 --euth Isoflurane \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/synthetic_samples/

# OR counterfactual samples
CUDA_VISIBLE_DEVICES=""  python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/whatif.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/v9/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --mode population \
    --tissue Liver \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/whatif_results/


# evaluate the synthetic counts from step 1
CUDA_VISIBLE_DEVICES=""  python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/evaluate_generated.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/v9/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --generated /home/jcasalet/nobackup/AI4LS/CVAE/results/synthetic_samples/Liver_C57BL-6J_Female_0_Isoflurane_n100/synthetic_counts.csv \
    --mode synthetic \
    --tissue Liver --strain C57BL/6J --sex Female --flight 1 \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/eval_results/generated

# evaluate the synthetic counts from step 1
CUDA_VISIBLE_DEVICES=""  python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/evaluate_generated.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/v9/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --generated /home/jcasalet/nobackup/AI4LS/CVAE/results/whatif_results/Liver_population/population_flight_expression.csv \
    --mode synthetic \
    --tissue Liver --strain C57BL/6J --sex Female --flight 1 \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/eval_results/counterfactual
