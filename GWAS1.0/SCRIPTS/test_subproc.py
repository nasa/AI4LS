import subprocess

x=subprocess.check_output(
    "bcftools view -r 19:11153896-11183896 /home/jcasalet/nobackup/GWAS/DATA/VCF/OUT/19/FILTER_BEFORE_MERGE/MERGED/FILTER_AFTER_MERGE/filteraftermerge.vcf.gz | grep GENE | awk '{print $8}' | cut -d\; -f1" ,
    shell=True
)
print(x)
