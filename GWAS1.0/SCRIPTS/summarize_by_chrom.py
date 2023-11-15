import pandas as pd
import json
import sys
import os


all_json={}
for i in range(1, 23):
    all_json[str(i)]=[]
    json_file="/home/jcasalet/nobackup/GWAS/DATA/VCF/OUT/" + str(i) + "/FILTER_BEFORE_MERGE/MERGED/FILTER_AFTER_MERGE/GWAS/SIG_FDR_BH/results.json"
    with open(json_file, 'r') as f:
        for json_obj in f:
            variants = json.loads(json_obj)
            if len(variants) != 0:
                all_json[str(i)].append(variants)
        f.close()

print(json.dumps(all_json))

