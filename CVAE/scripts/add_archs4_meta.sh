#!/bin/bash


python -u ../src/add_archs4_metadata.py \
    --archs4 /home/jcasalet/nobackup/FM/BIOFM_MOUSE/data/archs4/mouse_gene_v2.5.h5 \
    --genelab ../DATA/osdr_mouse.h5 \
    --pretrain ../DATA/archs4_pretrain_final.h5 \
    --output ../DATA/archs4_pretrain_metadata_final.h5
