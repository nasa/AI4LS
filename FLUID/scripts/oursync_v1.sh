#!/bin/bash -x

FLUID_HOME=/home/fluid
TO_BRIDGE_FROM_COLABSHIM=$FLUID_HOME/TO_BRIDGE_FROM_COLABSHIM   # this will be a symbolic link and used to simulate comms failure for data being downlinked from the ISS
TO_BRIDGE_FROM_TDS=$FLUID_HOME/TO_BRIDGE_FROM_TDS   # this will be a symbolic link and used to simulate comms failure for data being uplinked to the ISS

TO_TDS_FROM_ISS=$FLUID_HOME/FROM_ISS
TO_TDS_FROM_BRIDGE=$FLUID_HOME/FROM_BRIDGE
COMMS_UP=_COMMS_UP   # directory name post_fix to append to enable symbolic link switching
COMMS_DOWN=_COMMS_DOWN   # directory name post_fix to append to enable symbolic link switching

WORKSPACE_PATH=/home/fluid/data/WORKSPACE/workspace
this_proc=$(basename $0 .sh)

if [ $# -ne 2 ]
then
	echo "usage: $this_proc bridge|tds start|continue"
	exit 1
fi

MYHOSTNAME=$1
OPERATION=$2

if [ $MYHOSTNAME != "bridge" -a $MYHOSTNAME != "tds" ]
then
	echo "usage: $this_proc bridge|tds start|continue"
	exit 1
fi

if [ $OPERATION != "start" -a $OPERATION != "continue" ]
then
	echo "usage: $this_proc bridge|tds start|continue"
	exit 1
fi


idempotentize() {
	d=$1
	if [ -d $d -a $OPERATION == 'start' ]
	then
		mkdir -p ~/DONE/$TS
		mv $d ~/DONE/$TS
		mkdir -p $d
	elif [ ! -d $d ] 
	then
		mkdir -p $d
	fi					
}

if [ $MYHOSTNAME == "bridge" ]
then
	# Check if comms_updown simulation utility is installed, which means that the FLUID directories on the BRIDGE node will be symbolic links.
	# If yes, then point $TO_BRIDGE_FROM_COLABSHIM to the COMMS_UP subdirectory, otherwise leave it pointing to the normal directory.
	if [ -L $TO_BRIDGE_FROM_COLABSHIM ]; then   # if this is a symbolic link, the comms_updown is installed
		echo "NOTIFICATION: FLUID comms_updown simulation has been installed."
		echo "              Symbolic directory links will be used and existing files will not be archived."
		TO_BRIDGE_FROM_COLABSHIM=$TO_BRIDGE_FROM_COLABSHIM$COMMS_UP
		TO_BRIDGE_FROM_TDS=$TO_BRIDGE_FROM_TDS$COMM_UP
	else
		TS=$(date +%s)
		idempotentize $TO_BRIDGE_FROM_TDS
		idempotentize $TO_TDS_FROM_BRIDGE
	fi
	
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
		rsync -a fluid@colab-shim:${WORKSPACE_PATH}/response_path/ ${TO_TDS_FROM_BRIDGE}/response_path/
		rsync -a fluid@colab-shim:${WORKSPACE_PATH}/request_path/ ${TO_TDS_FROM_BRIDGE}/request_path/

		# then push that to tds
		rsync -a ${TO_TDS_FROM_BRIDGE}/ fluid@tds:${TO_TDS_FROM_BRIDGE}/

		# get stuff from tds and push to colab-shim
		rsync -a ${TO_BRIDGE_FROM_TDS}/request_path/ fluid@colab-shim:${WORKSPACE_PATH}/request_path/
		rsync -a ${TO_BRIDGE_FROM_TDS}/response_path/ fluid@colab-shim:${WORKSPACE_PATH}/response_path
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
		rsync -a ${TO_TDS_FROM_BRIDGE}/request_path/ fluid@agg-iss:${WORKSPACE_PATH}/request_path/
		rsync -a ${TO_TDS_FROM_BRIDGE}/response_path/ fluid@agg-iss:${WORKSPACE_PATH}/response_path/

		# get stuff from ISS
		rsync -a fluid@agg-iss:${WORKSPACE_PATH}/response_path/ ${TO_BRIDGE_FROM_TDS}/response_path/
		rsync -a fluid@agg-iss:${WORKSPACE_PATH}/request_path/ ${TO_BRIDGE_FROM_TDS}/request_path/

		# and push it to bridge
		rsync -a ${TO_BRIDGE_FROM_TDS}/ fluid@bridge:${TO_BRIDGE_FROM_TDS}/

		sleep 5 
	done

fi


