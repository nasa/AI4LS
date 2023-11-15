#!/bin/bash -x

sbatch /home/jcasalet/nobackup/GWAS/DATA/SCRIPTS/separate_by_chrom.sh /home/jcasalet/nobackup/GWAS/DATA/VCF X X 
#sbatch -t 03-00:00:00 /home/jcasalet/nobackup/GWAS/DATA/SCRIPTS/separate_by_chrom.sh /home/jcasalet/nobackup/GWAS/DATA/VCF 19 19
#sbatch  -t 03-00:00:00 /home/jcasalet/nobackup/GWAS/DATA/SCRIPTS/separate_by_chrom.sh /home/jcasalet/nobackup/GWAS/DATA/VCF 2 6 
#sbatch -t 03-00:00:00 /home/jcasalet/nobackup/GWAS/DATA/SCRIPTS/separate_by_chrom.sh /home/jcasalet/nobackup/GWAS/DATA/VCF 7 12 
#sbatch -t 03-00:00:00 /home/jcasalet/nobackup/GWAS/DATA/SCRIPTS/separate_by_chrom.sh /home/jcasalet/nobackup/GWAS/DATA/VCF 13 18  
#sbatch -t 03-00:00:00 /home/jcasalet/nobackup/GWAS/DATA/SCRIPTS/separate_by_chrom.sh /home/jcasalet/nobackup/GWAS/DATA/VCF 19 22
