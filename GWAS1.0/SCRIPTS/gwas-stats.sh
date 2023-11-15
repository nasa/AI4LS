#!/bin/bash



TS=$(date +%s)
if [ -d /home/jcasalet/nobackup/GWAS/DATA/RESULTS/STATS ]
then
	mv /home/jcasalet/nobackup/GWAS/DATA/RESULTS/STATS /home/jcasalet/nobackup/GWAS/DATA/RESULTS/STATS-$TS
fi
mkdir /home/jcasalet/nobackup/GWAS/DATA/RESULTS/STATS
cd /home/jcasalet/nobackup/GWAS/DATA/RESULTS/STATS
# bcftools stats
echo "{" > bcftools_stats.json
for i in $(seq 1 22)
do
	samples=$(bcftools stats /home/jcasalet/nobackup/GWAS/DATA/VCF/OUT/$i/FILTER_BEFORE_MERGE/MERGED/FILTER_AFTER_MERGE/filteraftermerge.vcf.gz | grep ^SN | grep samples | cut -d: -f2)
	records=$(bcftools stats /home/jcasalet/nobackup/GWAS/DATA/VCF/OUT/$i/FILTER_BEFORE_MERGE/MERGED/FILTER_AFTER_MERGE/filteraftermerge.vcf.gz | grep ^SN | grep records | cut -d: -f2)
	snps=$(bcftools stats /home/jcasalet/nobackup/GWAS/DATA/VCF/OUT/$i/FILTER_BEFORE_MERGE/MERGED/FILTER_AFTER_MERGE/filteraftermerge.vcf.gz | grep ^SN | grep SNPs | cut -d: -f2) 
	echo \"$i\": { >> bcftools_stats.json
	echo -n \"samples\":${samples}, >> bcftools_stats.json
	echo -n \"records\":${records}, >> bcftools_stats.json
	echo -n \"snps\":${snps}} >> bcftools_stats.json
	if [ $i -ne 22 ]
	then
		echo "," >> bcftools_stats.json
	fi
done
echo "}" >> bcftools_stats.json

# plink2 missing stats
for i in $(seq 1 22)
do
	plink2 --vcf /home/jcasalet/nobackup/GWAS/DATA/VCF/OUT/$i/FILTER_BEFORE_MERGE/MERGED/FILTER_AFTER_MERGE/filteraftermerge.vcf.gz --missing --out ${i}_missing
done




