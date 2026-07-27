#!/bin/bash
#python -u prepare_archs4.py \
#    --archs4 ~/nobackup/FM/BIOFM_MOUSE/data/archs4/mouse_gene_v2.5.h5 \
#    --genelab /home/jcasalet/nobackup/CVAE/DATA/osdr_mouse.h5 \
#    --output /home/jcasalet/nobackup/CVAE/DATA/archs4_pretrain.h5 \
#    --max_samples 200000

#python -u ~/nobackup/AI4LS/CVAE/src/prepare_archs4.py \
    #--archs4 ~/nobackup/FM/BIOFM_MOUSE/data/archs4/mouse_gene_v2.5.h5 \
    #--genelab ~/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    #--output ~/nobackup/AI4LS/CVAE/DATA/archs4_pretrain.h5  \
    #--max_samples 500000

#python -u ~/nobackup/AI4LS/CVAE/src/prepare_archs4_fast.py \
    #--archs4 ~/nobackup/FM/BIOFM_MOUSE/data/archs4/mouse_gene_v2.5.h5 \
    #--genelab ~/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    #--output ~/nobackup/AI4LS/CVAE/DATA/archs4_pretrain_nometadata.h5 \
    #--max_samples 200000 \
    #--sample_block_size=2048

#python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/prepare_archs4_v5.py \
    #--archs4 /home/jcasalet/nobackup/FM/BIOFM_MOUSE/data/archs4/mouse_gene_v2.5.h5 \
    #--genelab /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    #--output /home/jcasalet/nobackup/AI4LS/CVAE/DATA/archs4_pretrain_v5.h5 \
    #--sc_threshold 0.1

#python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/prepare_archs4_v5.py \
        #--archs4 /home/jcasalet/nobackup/FM/BIOFM_MOUSE/data/archs4/mouse_gene_v2.5.h5 \
        #--genelab /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
        #--output /home/jcasalet/nobackup/AI4LS/CVAE/DATA/archs4_formatted_metadata.h5
        ##--block_size

python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/prepare_archs4.py \
	--archs4 /home/jcasalet/nobackup/FM/BIOFM_MOUSE/data/archs4/mouse_gene_v2.5.h5 \
	--genelab /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
        --output /home/jcasalet/nobackup/AI4LS/CVAE/DATA/archs4_osdr_subset.h5

