import sys
import json
from pyensembl import EnsemblRelease
import subprocess

INPUT_JSON="/home/jcasalet/nobackup/GWAS/DATA/RESULTS/snp-summary-per-chrom.json"
INPUT_VCF="/home/jcasalet/nobackup/GWAS/DATA/all_merged.vcf"

data=EnsemblRelease(75)
snp_list=dict()
with open(INPUT_JSON, 'r') as f:
    json_data=json.load(f)
    for chrom in json_data:
        #print("chrom: ", chrom)
        for rad_dict in json_data[chrom]:
            for rad in rad_dict:
                if not rad in snp_list:
                    snp_list[rad]=dict()
                for var in rad_dict[rad]:
                    v=eval(var)
                    #print("v: ", v)
                    pos=int(v[2])
                    genes=data.gene_names_at_locus(contig=chrom, position=pos)
                    for gene in genes:
                        if not gene in snp_list[rad]:
                            snp_list[rad][gene]=list()
                        snp_list[rad][gene].append(var)
f.close()

final_report=dict()

for rad in snp_list:
    if not rad in final_report:
        final_report[rad]=dict()
    for gene in snp_list[rad]:
        if not gene in final_report[rad]:
            final_report[rad][gene]=dict()
        for snp in snp_list[rad][gene]:
            if not snp in final_report[rad][gene]:
                final_report[rad][gene][snp]=dict()
            snpID=eval(snp)[0]
            #print(snpID)
            # grep -e $rs2084106\t /home/jcasalet/nobackup/GWAS/DATA/all_merged.vcf
            homo_ref_command="grep -e $" + "\'" +  snpID  + "\\t\' " +  INPUT_VCF + " | xargs | tr ' ' '\n' | grep 0/0 | wc -l"
            homo_alt_command="grep -e $" + "\'" +  snpID  + "\\t\' " +  INPUT_VCF + " | xargs | tr ' ' '\n' | grep 1/1 | wc -l"
            het_command=     "grep -e $" + "\'" +  snpID  + "\\t\' " +  INPUT_VCF + " | xargs | tr ' ' '\n' | grep 0/1 | wc -l"
            snp_count_homo_ref=int(subprocess.check_output(homo_ref_command, shell=True).decode("utf-8").strip())
            snp_count_homo_alt=int(subprocess.check_output(homo_alt_command, shell=True).decode("utf-8").strip())
            snp_count_het=int(subprocess.check_output(het_command, shell=True).decode("utf-8").strip())
            final_report[rad][gene][snp]=[int(snp_count_homo_ref), int(snp_count_het), int(snp_count_homo_alt)]
            #final_report[rad][gene][snp]=[int(snp_count_homo_ref)]
            #final_report[rad][gene][snp]=[int(snp_count_homo_ref), int(snp_count_homo_alt)]
            print(final_report[rad][gene][snp], flush=True)

print(json.dumps(final_report))
