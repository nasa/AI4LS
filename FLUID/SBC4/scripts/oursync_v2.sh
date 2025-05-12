#!/bin/bash -x

ALL_LOCAL=true
TO_TDS_FROM_BRIDGE=/home/fluid/TO_TDS_FROM_BRIDGE
TO_BRIDGE_FROM_TDS=/home/fluid/TO_BRIDGE_FROM_TDS
TO_ISS_FROM_TDS=/data/work/fluid/project/release/TO_ISS
TO_TDS_FROM_ISS=/data/work/fluid/project/release/TO_TDS
AGG_ISS_WORKSPACE_PATH=/home/ec2-user/data/AGG_ISS/WORKSPACE/workspace
COLAB_SHIM_WORKSPACE_PATH=/home/ec2-user/data/COLAB_SHIM/WORKSPACE/workspace
COLAB_SHIM=54.226.73.143
#TDS=192.48.188.134
TDS=54.210.26.155
BRIDGE=52.5.52.204
AGG_ISS=54.226.73.143
SLEEP_INTERVAL=5

#######################################
# run bridge on bridge
# run tds on tds (for testing purposes)
#######################################


if [ $# -ne 2 ]
then
	echo "usage: $0 bridge|tds start|continue"
	exit 1
fi

MYHOSTNAME=$1
OPERATION=$2

if [ $MYHOSTNAME != "bridge" -a $MYHOSTNAME != "tds" ]
then
	echo "usage: $0 bridge|tds start|continue"
	exit 1
fi



idempotentize() {
	d=$1
	if [ -d $d -a $OPERATION == 'start' ]
	then
		mv $d ~/DONE/$TS
		mkdir -p $d
	elif [ ! -d $d ] 
	then
		mkdir -p $d
	fi	
				
}




if [ $MYHOSTNAME == "bridge" ]
then
	TS=$(date +%s)
	mkdir -p ~/DONE/$TS

	idempotentize $TO_BRIDGE_FROM_TDS
	idempotentize $TO_TDS_FROM_BRIDGE

	while true
	do
		ssh fluid@${COLAB_SHIM} [ -d $COLAB_SHIM_WORKSPACE_PATH ] 
		colabshim_running=$(echo $?)
		ssh fluid@${TDS} [ -d $TO_ISS_FROM_TDS -a -d $TO_TDS_FROM_ISS ]
		tds_running=$(echo $?)
		if [ $colabshim_running == 0 -a $tds_running == 0 ]
		then
			break
		else
			echo waiting for colab-shim and tds nodes
			sleep 3
		fi
	done
	

	while true
	do
		# pull requests from tds 
		rsync -a fluid@${TDS}:${TO_TDS_FROM_ISS}/ ${TO_BRIDGE_FROM_TDS}/ 
		# and push requests to colab-shim
		rsync -a ${TO_BRIDGE_FROM_TDS}/ fluid@${COLAB_SHIM}:${COLAB_SHIM_WORKSPACE_PATH}/request_path
		# pull responses from colab-shim
		rsync -a fluid@${COLAB_SHIM}:${COLAB_SHIM_WORKSPACE_PATH}/response_path/ ${TO_TDS_FROM_BRIDGE}/
		# and push responses to tds 
		rsync -a  ${TO_TDS_FROM_BRIDGE}/ fluid@${TDS}:${TO_ISS_FROM_TDS}/

		sleep $SLEEP_INTERVAL 
	done


elif [ $MYHOSTNAME == "tds" ]
then
	TS=$(date +%s)
	mkdir -p ~/DONE/$TS

	idempotentize $TO_TDS_FROM_BRIDGE
	idempotentize $TO_BRIDGE_FROM_TDS

	while true
	do
		# pull responses from TDS 
		echo "pull from tds"
		rsync -a fluid@${TDS}:${TO_ISS_FROM_TDS}/ ${TO_TDS_FROM_BRIDGE}/
		# and push responses to ISS
		echo "push to iss"
		rsync -a ${TO_TDS_FROM_BRIDGE}/ fluid@${AGG_ISS}:${AGG_ISS_WORKSPACE_PATH}/response_path

		# pull requests from ISS 
		echo "pull from iss"
		rsync -a  fluid@${AGG_ISS}:${AGG_ISS_WORKSPACE_PATH}/request_path/ ${TO_BRIDGE_FROM_TDS}/ 
		# and push requests to TDS
		echo "push to tds"
		rsync -a ${TO_BRIDGE_FROM_TDS}/ fluid@${TDS}:${TO_TDS_FROM_ISS}/
		sleep $SLEEP_INTERVAL
	done

fi
