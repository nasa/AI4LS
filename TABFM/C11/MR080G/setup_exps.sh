#!/bin/bash


for exp in $(ls DATA/post*.csv)
do	
	echo $exp
	exp_base=$(echo $exp | cut -d. -f1 | cut -d/ -f2)
	echo '#!/bin/bash' > run_${exp_base}.sh
	echo "python -u c11_regression.py DATA/${exp_base}.csv RESULTS/reg_rf_vs_tabpfn_${exp_base}.csv" >> run_${exp_base}.sh
	chmod +x run_${exp_base}.sh 
done
