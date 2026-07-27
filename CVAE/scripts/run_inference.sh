#!/bin/bash

# integrated gradients (slower but more accurate and tissue-specific)
python -u  /home/jcasalet/nobackup/AI4LS/CVAE/src/inference.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/v5/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/inference/finetune_ig/ \
    --attribution_method integrated_gradients \
    --ig_steps 50 \
    --enrichment_genes 100 \
    --enrichment_cutoff 0.05

# vanilla gradients (fast, existing behavior — default unchanged)
python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/inference.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/v5/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/inference/finetune/v5/ \
    --enrichment_genes 100 \
    --enrichment_cutoff 0.05

# integrated gradients (slower but more accurate and tissue-specific)
python -u  /home/jcasalet/nobackup/AI4LS/CVAE/src/inference.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/v5/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/inference/finetune/v5_ig/ \
    --attribution_method integrated_gradients \
    --ig_steps 50 \
    --enrichment_genes 100 \
    --enrichment_cutoff 0.05

# vanilla gradients (fast, existing behavior — default unchanged)
python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/inference.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/v5/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/inference/finetune/v5/ \
    --enrichment_genes 100 \
    --enrichment_cutoff 0.05
