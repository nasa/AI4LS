import sys
import json
import subprocess


def main():
    all_genes=dict()
    json_file=sys.argv[1]
    chrom=sys.argv[2]
    offset=int(sys.argv[3])


    with open(json_file, 'r') as f:
        results=json.load(f)
        for rads in results[chrom]:
            for rad in rads.keys():
                for snp in rads[rad]:
                    snp=eval(snp)
                    c=snp[1]
                    pos=snp[2]
                    left_pos=pos-offset
                    right_pos=pos+offset
                    command="bcftools view -r " + str(c) + ":" + str(left_pos) + "-" + str(right_pos) + " /home/jcasalet/nobackup/GWAS/DATA/VCF/OUT/" + str(c) + "/FILTER_BEFORE_MERGE/MERGED/FILTER_AFTER_MERGE/filteraftermerge.vcf.gz" + "| grep GENE= | awk '{print $8}' | grep GENE= | cut -d\; -f1"
                    list_of_genes=subprocess.check_output(command, shell=True).decode("utf-8").strip()
                    for gene in list_of_genes.split("GENE="):
                        gene=gene.strip()
                        if not gene in all_genes:
                            all_genes[gene]=0
                        all_genes[gene] += 1 
    f.close()
    print(all_genes)
    return all_genes

if __name__ == "__main__":
    main()
