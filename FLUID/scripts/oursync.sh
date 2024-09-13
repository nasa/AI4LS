#!/bin/bash -x

TO_TDS_FROM_BRIDGE=/home/fluid/TO_TDS_FROM_BRIDGE
TO_BRIDGE_FROM_TDS=/home/fluid/TO_BRIDGE_FROM_TDS

WORKSPACE_PATH=/home/fluid/data/WORKSPACE/workspace

MYHOSTNAME=$1
OPERATION=$2

if [ $MYHOSTNAME != "bridge" -a $MYHOSTNAME != "tds" ]
then
	echo "usage: $0 bridge|tds"
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

if [ $# -ne 2 ]
then
	echo "usage: $0 bridge|tds start|continue"
	exit 1
fi



if [ $MYHOSTNAME == "bridge" ]
then
	TS=$(date +%s)
	mkdir -p ~/DONE/$TS

	idempotentize $TO_BRIDGE_FROM_TDS
	idempotentize $TO_TDS_FROM_BRIDGE

	while true
	do
		ssh fluid@colab-shim [ -d $WORKSPACE_PATH ] 
		colabshim_running=$(echo $?)
		ssh fluid@tds [ -d $TO_TDS_FROM_BRIDGE -a -d $TO_BRIDGE_FROM_TDS ]
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
		# get stuff from colab-shim
		rsync -a fluid@colab-shim:${WORKSPACE_PATH}/response_path/ ${TO_TDS_FROM_BRIDGE}/
		rsync -a fluid@colab-shim:${WORKSPACE_PATH}/request_path/ ${TO_TDS_FROM_BRIDGE}/

		# then push that to tds
		rsync -a ${TO_TDS_FROM_BRIDGE}/ fluid@tds:${TO_TDS_FROM_BRIDGE}/

		# get stuff from tds and push to colab-shim
		rsync -a ${TO_BRIDGE_FROM_TDS}/ fluid@colab-shim:${WORKSPACE_PATH}/request_path/
		rsync -a ${TO_BRIDGE_FROM_TDS}/ fluid@colab-shim:${WORKSPACE_PATH}/response_path
		sleep 5 
	done

fi

if [ $MYHOSTNAME == "tds" ]
then
	TS=$(date +%s)
	mkdir -p ~/DONE/$TS

	idempotentize $TO_TDS_FROM_BRIDGE
	idempotentize $TO_BRIDGE_FROM_TDS

	while true
	do
		ssh fluid@bridge [ -d $TO_BRIDGE_FROM_TDS -a -d $TO_TDS_FROM_BRIDGE ]
		bridge_running=$(echo $?)
		if [ $bridge_running == 0 ]
		then
			break
		else
			echo waiting for bridge node
			sleep 3
		fi
	done
	

	while true
	do
		# get stuff from bridge 
		rsync -a fluid@bridge:${TO_TDS_FROM_BRIDGE}/ ${TO_TDS_FROM_BRIDGE}/

		# and push it to ISS
		rsync -a ${TO_TDS_FROM_BRIDGE}/ fluid@agg-iss:${WORKSPACE_PATH}/request_path/
		rsync -a ${TO_TDS_FROM_BRIDGE}/ fluid@agg-iss:${WORKSPACE_PATH}/response_path/

		# get stuff from ISS
		rsync -a fluid@agg-iss:${WORKSPACE_PATH}/response_path/ ${TO_BRIDGE_FROM_TDS}/
		rsync -a fluid@agg-iss:${WORKSPACE_PATH}/request_path/ ${TO_BRIDGE_FROM_TDS}/

		# and push it to bridge
		rsync -a ${TO_BRIDGE_FROM_TDS}/ fluid@bridge:${TO_BRIDGE_FROM_TDS}/

		sleep 5 
	done

fi

