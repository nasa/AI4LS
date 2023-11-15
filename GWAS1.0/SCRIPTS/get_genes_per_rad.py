import sys
import json
from pyensembl import EnsemblRelease

INPUT_JSON="/home/jcasalet/nobackup/GWAS/DATA/RESULTS/snp-summary-per-chrom.json"

data=EnsemblRelease(75)
all_json={}
with open(INPUT_JSON, 'r') as f:
    json_data=json.load(f)
    for chrom in json_data:
        #print("chrom: ", chrom)
        for rad_dict in json_data[chrom]:
            for rad in rad_dict:
                if not rad in all_json:
                    all_json[rad] = list() 
                for var in rad_dict[rad]:
                    v=eval(var)
                    #print("v: ", v)
                    pos=int(v[2])
                    genes=data.gene_names_at_locus(contig=chrom, position=pos)
                    for gene in genes:
                        all_json[rad].append(gene)
f.close()
for rad in all_json:
    all_json[rad] = list(all_json[rad])


all_json_dict=dict()
for rad in all_json:
    all_json_dict[rad] = dict()
    gene_list=all_json[rad]
    for gene in set(gene_list):
        all_json_dict[rad][gene] = all_json[rad].count(gene)
#print(json.dumps(all_json))
print(json.dumps(all_json_dict))
