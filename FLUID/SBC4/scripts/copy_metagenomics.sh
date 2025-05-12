#!/bin/bash

NUM_NODES=$1

if [ -z "$NUM_NODES" ]
then
	echo usage $0 num_nodes
	exit 1
fi

HOSTS="COLAB_EARTH AGG_EARTH COLAB_SHIM_ISS AGG_ISS COLAB_ISS COLAB_SHIM_MOON AGG_MOON COLAB_MOON"
for host in $HOSTS 
do

	if [ $NUM_NODES -eq 1 ]
	then
		cp ../../data/METAGENOMICS/combined_test.csv ../../data/WORKSPACE/${host}/col_0/test/data.csv	
		cp ../../data/METAGENOMICS/combined_train.csv ../../data/WORKSPACE/${host}/col_0/train/data.csv	
	else
		echo $host
		mkdir -p ../../data/WORKSPACE/${host}/col_0/train
		mkdir -p ../../data/WORKSPACE/${host}/col_0/test
		mkdir -p ../../data/WORKSPACE/${host}/col_1/train
		mkdir -p ../../data/WORKSPACE/${host}/col_1/test
		mkdir -p ../../data/WORKSPACE/${host}/col_2/train
		mkdir -p ../../data/WORKSPACE/${host}/col_2/test

		cp ../../data/METAGENOMICS/SRA_PRJEB45093_Combined-gene-level-taxonomy-coverages-CPM_GLmetagenomics_subsetSpecies_ariLabel_test.csv ../../data/WORKSPACE/${host}/col_0/test/data.csv 
		cp ../../data/METAGENOMICS/SRA_PRJEB45093_Combined-gene-level-taxonomy-coverages-CPM_GLmetagenomics_subsetSpecies_ariLabel_train.csv ../../data/WORKSPACE/${host}/col_0/train/data.csv 

		cp ../../data/METAGENOMICS/GLDS-564_GMetagenomics_Combined-gene-level-taxonomy-coverages-CPM_GLmetagenomics_subsetSpecies_ariLabel_test.csv ../../data/WORKSPACE/${host}/col_1/test/data.csv 
		cp ../../data/METAGENOMICS/GLDS-564_GMetagenomics_Combined-gene-level-taxonomy-coverages-CPM_GLmetagenomics_subsetSpecies_ariLabel_train.csv ../../data/WORKSPACE/${host}/col_1/train/data.csv 

		cp ../../data/METAGENOMICS/GLDS-564_GMetagenomics_Combined-gene-level-taxonomy-coverages-CPM_GLmetagenomics_subsetSpecies_ariLabel_test.csv ../../data/WORKSPACE/${host}/col_2/test/data.csv 
		cp ../../data/METAGENOMICS/GLDS-564_GMetagenomics_Combined-gene-level-taxonomy-coverages-CPM_GLmetagenomics_subsetSpecies_ariLabel_train.csv ../../data/WORKSPACE/${host}/col_2/train/data.csv 
	fi
done

