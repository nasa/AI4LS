#!/bin/bash

TS=$(date +%s)

if [ -d /home/jcasalet/nobackup/GWAS/DATA/RESULTS/PLOTS/ ]
then
	mv /home/jcasalet/nobackup/GWAS/DATA/RESULTS/PLOTS/ /home/jcasalet/nobackup/GWAS/DATA/RESULTS/PLOTS-$TS
fi

mkdir -p /home/jcasalet/nobackup/GWAS/DATA/RESULTS/PLOTS/

for i in $(seq 1 22)
do 
	for rad in Ar Fe Gamma Si
	do
		for dur in 4h_Slope 24h_Slope Residual
		do
			cat /home/jcasalet/nobackup/GWAS/DATA/VCF/OUT/$i/FILTER_BEFORE_MERGE/MERGED/FILTER_AFTER_MERGE/GWAS/out_filename.${rad}_${dur}.glm.linear >> /home/jcasalet/nobackup/GWAS/DATA/RESULTS/PLOTS/all_${rad}_${dur}.glm.linear
			# cat /home/jcasalet/nobackup/GWAS/DATA/VCF/OUT/$i/FILTER_BEFORE_MERGE/MERGED/FILTER_AFTER_MERGE/GWAS/out_filename.${rad}_${dur}.glm.linear.adjusted >> /home/jcasalet/nobackup/GWAS/DATA/RESULTS/PLOTS/all_${rad}_${dur}.glm.linear.adjusted
		done

	done
done


cd /home/jcasalet/nobackup/GWAS/DATA/RESULTS/PLOTS/
for rad in Ar Fe Gamma Si
do
	for dur in 4h_Slope 24h_Slope Residual
	do
		head -1 all_${rad}_${dur}.glm.linear > header
		grep -v CHROM all_${rad}_${dur}.glm.linear > temp-$TS
		cat header temp-$TS >> temp2-$TS
		mv temp2-$TS all_${rad}_${dur}.glm.linear
		rm -f temp-$TS
		rm -f header
		qmplot -I all_${rad}_${dur}.glm.linear -T ${rad}_${dur} -M ID --dpi 300 -O ${rad}_${dur}
		#qmplot -I all_${rad}_${dur}.glm.linear.adjusted -T ${rad}_${dur}.adjusted -M ID --dpi 300 -O ${rad}_${dur}.adjusted
	done
done
