#!/bin/bash

TOPDIR='/Users/jcasalet/Desktop/CODES/NASA/AI4LS/FLUID/SBC4/PUSH_BUTTON'

PATH_TO_SCRIPTS=${TOPDIR}/scripts
PATH_TO_CRISP=${TOPDIR}/crisp/fl_src
PATH_TO_WORKSPACE=${TOPDIR}/data/WORKSPACE
MODEL=irm # erm or irm
DATA=METAGENOMIC #SYNTHETIC or METAGENOMIC 
REG=1.0
NUMROUNDS=3
NUMEPOCHS=2
HIDDEN_DIM=1
LR=1e-03
LAYERS=3 # 1 or 3 or 5
#colabs='export COLABS="colab-earth"'
#colabs='export COLABS="colab-earth colab-iss"'
colabs='export COLABS="colab-earth colab-iss colab-moon"'
#aggs='export AGGS="agg-earth"'
#aggs='export AGGS="agg-earth agg-iss"'
aggs='export AGGS="agg-earth agg-iss agg-moon"'
#shims='export COLAB_SHIMS="colab-shim-iss"'
shims='export COLAB_SHIMS="colab-shim-iss colab-shim-moon"'

#============================
# copy DATA
#============================
echo "copying $DATA data"
for h in AGG_EARTH COLAB_EARTH COLAB_SHIM_ISS COLAB_SHIM_MOON AGG_ISS COLAB_ISS AGG_MOON COLAB_MOON
do
	if [ ! -d ${PATH_TO_WORKSPACE}/$h/col_0/train/ ]
	then
		mkdir -p ${PATH_TO_WORKSPACE}/$h/col_0/train/
	fi

	cp DATA/${DATA}/col_0_train.csv ${PATH_TO_WORKSPACE}/$h/col_0/train/data.csv
	cp DATA/${DATA}/col_1_train.csv ${PATH_TO_WORKSPACE}/$h/col_1/train/data.csv
	cp DATA/${DATA}/col_2_train.csv ${PATH_TO_WORKSPACE}/$h/col_2/train/data.csv
	cp DATA/${DATA}/col_0_test.csv ${PATH_TO_WORKSPACE}/$h/col_0/test/data.csv
	cp DATA/${DATA}/col_1_test.csv ${PATH_TO_WORKSPACE}/$h/col_1/test/data.csv
	cp DATA/${DATA}/col_2_test.csv ${PATH_TO_WORKSPACE}/$h/col_2/test/data.csv
done
#============================

#============================
# edit crisp-config.sh
#============================
echo "updating crisp-config.sh"
cp CONFIG/crisp-config.sh ${PATH_TO_SCRIPTS}/crisp-config.sh
#gsed -i "1,$ s/TOP_DIR=.*/TOP_DIR=${TOP_DIR}/" ${PATH_TO_SCRIPTS}/crisp-config.sh
gsed -i "1,$ s:TOP_DIR=.*:TOP_DIR=$TOPDIR:" ${PATH_TO_SCRIPTS}/crisp-config.sh
gsed -i "1,$ s/ROUNDS=.*/ROUNDS=$NUMROUNDS/" ${PATH_TO_SCRIPTS}/crisp-config.sh
gsed -i "1,$ s/EPOCHS=.*/EPOCHS=$NUMEPOCHS/" ${PATH_TO_SCRIPTS}/crisp-config.sh
gsed -i "1,$ s/COLABS/$colabs/" ${PATH_TO_SCRIPTS}/crisp-config.sh 
gsed -i "1,$ s/AGGS/$aggs/" ${PATH_TO_SCRIPTS}/crisp-config.sh 
gsed -i "1,$ s/SHIMS/$shims/" ${PATH_TO_SCRIPTS}/crisp-config.sh 
#============================

#============================
# edit crisp_task_runner.py
#============================
echo "updating crisp_task_runner.py"
cp CODE/crisp_task_runner.py new_crisp_task_runner.py
if [ "$MODEL" == "erm" ]
then
	model='"nlerm" : NLERMModule(logger=self.logger, device=self.device, reg=REG, prefix="nlerm_"),'
elif [ "$MODEL" == "irm" ]
then
	model='"nlirm" : NLIRMModule(logger=self.logger, device=self.device, reg=REG, prefix="nlirm_"),'
else
	echo "no such model $MODEL"
	exit 1
fi
gsed -i "1,$ s/MODEL/$model/" new_crisp_task_runner.py
gsed -i "1,$ s/reg=REG/reg=$REG/" new_crisp_task_runner.py

mv new_crisp_task_runner.py ${PATH_TO_CRISP}/crisp_task_runner.py
#============================


#============================
# edit erm_module.py or irm_module.py
#============================
echo "updating ${MODEL}_module.py" 

ONE_LAYER='setattr(self, self.phi_name, torch.nn.Sequential(torch.nn.Linear(self.inputSize, HIDDEN_DIM * self.inputSize, bias=True), torch.nn.Tanh(), torch.nn.Linear(HIDDEN_DIM * self.inputSize, self.inputSize, bias=True)))'
        
THREE_LAYER='setattr(self, self.phi_name, torch.nn.Sequential(torch.nn.Linear(self.inputSize, HIDDEN_DIM * self.inputSize, bias=True), torch.nn.Tanh(), torch.nn.Linear(HIDDEN_DIM * self.inputSize, self.inputSize, bias=True), torch.nn.Tanh(), torch.nn.Linear(self.inputSize, HIDDEN_DIM * self.inputSize, bias=True), torch.nn.Tanh(), torch.nn.Linear(HIDDEN_DIM * self.inputSize, self.inputSize, bias=True)))'
        
FIVE_LAYER='setattr(self, self.phi_name, torch.nn.Sequential(torch.nn.Linear(self.inputSize, HIDDEN_DIM * self.inputSize, bias=True), torch.nn.Tanh(), torch.nn.Linear(HIDDEN_DIM * self.inputSize, self.inputSize, bias=True), torch.nn.Tanh(), torch.nn.Linear(self.inputSize, HIDDEN_DIM * self.inputSize, bias=True), torch.nn.Tanh(), torch.nn.Linear(HIDDEN_DIM * self.inputSize, self.inputSize, bias=True), torch.nn.Linear(self.inputSize, HIDDEN_DIM * self.inputSize, bias=True), torch.nn.Tanh(), torch.nn.Linear(HIDDEN_DIM * self.inputSize, self.inputSize, bias=True)))'
	
cp CODE/${MODEL}_module.py new_${MODEL}_module.py

if [ $LAYERS == 1 ]
then
	gsed -i "s/SETATTR/$ONE_LAYER/" new_${MODEL}_module.py
elif [ $LAYERS == 3 ]
then
	gsed -i "s/SETATTR/$THREE_LAYER/" new_${MODEL}_module.py
else
	gsed -i "s/SETATTR/$FIVE_LAYER/" new_${MODEL}_module.py
fi
	
gsed -i "1,$ s/HIDDEN_DIM/$HIDDEN_DIM/g" new_${MODEL}_module.py

gsed -i "1,$ s/lr=.*/lr=${LR}\)/" new_${MODEL}_module.py

mv new_${MODEL}_module.py ${PATH_TO_CRISP}/${MODEL}_module.py

