#!/bin/bash

# -e 200 -ld 8 -bs 1 -nl 2 -hd 64 -lr 5e-03 -nb 5 -ng 0  -ie ./Proj2_Normalized_Counts.csv -im ./all_metadata_Proj2.csv -cd checkpoints/ -gf top-liver-genes.txt
# gamma_0.983_ld_8_bs_2_nl_2_hd_64_lr_1e-04.h5

max_gamma=0
epochs=50
seed=23
num_samples=112
num_genes=8000
expr_data=expanded_20_0.05_expr.csv
meta_data=expanded_20_0.05_meta.csv
output_dir=/tmp
re='^[0-9]+([.][0-9]+)?$'
hostname=$(hostname)
outputFile=$output_dir/myout_$hostname
pathToPython=.

for ld in 16 32
do
  for bs in 4 8
    do
      for nl in 2 4 
      do
        for hd in 128 256
        do
	  for lr in 5e-04 1e-03
	  do
	     echo "new job: ld=$ld bs=$bs nl=$nl hd=$hd lr=$lr"
             gamma=$(python -u ${pathToPython}/my_synthetic.py -g 0 -e $epochs -ld $ld -bs $bs -nl $nl -hd $hd -lr $lr -nb 5 -ng $num_genes -pg False -s $seed -ns $num_samples -ie $expr_data -im $meta_data -od $output_dir | tee $outputFile | grep "Gamma(Dx, Dz):" | awk -F: '{print $2}' | xargs)
	     if ! [[ $gamma =~ $re ]]
	     then
	        rm -rf checkpoints
	        mkdir checkpoints
		echo "got a nan ... continuing to next"
                continue 
 	     fi
	     echo "got a number $gamma ... seeing if it's max"
	     gamma_ten=$(echo $gamma | cut -d. -f2)
             echo "gamma:" $gamma "ld:" $ld "bs:" $bs "nl:" $nl "hd:" $hd "lr:" $lr >> ./params.txt
             if [ $gamma_ten -gt $max_gamma ]
             then
		echo "found max $gamma"
                echo "gamma:" $gamma "ld:" $ld "bs:" $bs "nl:" $nl "hd:" $hd "lr:" $lr >> ./best_params.txt
		cp checkpoints/models/gen_liver.h5 MODELS/gamma_${gamma}_ld_${ld}_bs_${bs}_nl_${nl}_hd_${hd}_lr_${lr}_ng_${num_genes}.h5
                max_gamma=$gamma_ten
             fi
           done
        done
      done
    done
done
