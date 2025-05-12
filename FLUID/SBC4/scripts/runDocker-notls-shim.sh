#!/bin/bash -x

#configure the paths and IP addresses of the FLUID network
source ./crisp-config.sh

create_network() {
	MYNET=$1
	network_exists=$(docker network list | grep $MYNET)
	if [ -z "$network_exists" ]
	then
		docker network create --subnet=$SUBNET $MYNET
	fi
}

process_args() {
	ARGS=$1
	for arg in $ARGS
	do
		case $arg in
			-r |--role)
			ROLE="$2"
        		shift # Remove argument name from processing
        		shift # Remove argument value from processing
			;;

       			*)
			echo "wrong usage: $ARGS"
			exit 1
       			;;
		esac
	done	
	echo $ROLE
}

ROLE=$(process_args $@)
echo my role in runDocker: $ROLE
create_network $MYNET

case $ROLE in
	agg-iss)
		export HOSTNAME=agg-iss
		export IP=$AGG_ISS_IP
		RUNDOCKER_OPTS="-p ${AGG_ISS_PORT}:${AGG_ISS_PORT}"
		export DATA_PATH=$AGG_ISS_DATA_PATH
		;;
	agg-moon)
		export HOSTNAME=agg-moon
		export IP=$AGG_MOON_IP
		RUNDOCKER_OPTS="-p ${AGG_MOON_PORT}:${AGG_MOON_PORT}"
		export DATA_PATH=$AGG_MOON_DATA_PATH
		;;
	agg-earth)
		export HOSTNAME=agg-earth
		export IP=$AGG_EARTH_IP
		RUNDOCKER_OPTS="-p ${AGG_EARTH_PORT}:${AGG_EARTH_PORT}"
		export DATA_PATH=$AGG_EARTH_DATA_PATH
		;;
	colab-iss)
		export HOSTNAME=colab-iss
		export IP=$COLAB_ISS_IP
		export DATA_PATH=$COLAB_ISS_DATA_PATH
		;;
	colab-earth)
		export HOSTNAME=colab-earth
		export IP=$COLAB_EARTH_IP
		export DATA_PATH=$COLAB_EARTH_DATA_PATH
		;;
	colab-moon)
		export HOSTNAME=colab-moon
		export IP=$COLAB_MOON_IP
		export DATA_PATH=$COLAB_MOON_DATA_PATH
		;;
	colab-shim-iss)
		export HOSTNAME=colab-shim-iss
		export IP=$COLAB_SHIM_ISS_IP
		export DATA_PATH=$COLAB_SHIM_ISS_DATA_PATH
		;;
	colab-shim-moon)
		export HOSTNAME=colab-shim-moon
		export IP=$COLAB_SHIM_MOON_IP
		export DATA_PATH=$COLAB_SHIM_MOON_DATA_PATH
		;;
	root)
		docker run -it --rm --user 0:0 $IMAGE_NAME /bin/bash
		;;
	*)
		echo "wrong usage: $ROLE"
		exit 1
		;;
esac

echo about to run container with role = $ROLE


#docker run -p 8888:8888 -h ${HOSTNAME}  --user=fluid:fluid  -v ${DATA_PATH}:/data:rw  -v ${SCRIPT_PATH}:/scripts:ro  --add-host agg-iss:$AGG_ISS_IP --add-host agg-earth:$AGG_EARTH_IP --add-host colab-iss:$COLAB_ISS_IP --add-host colab-shim:$COLAB_SHIM_IP --add-host colab-earth:$COLAB_EARTH_IP --restart always  $IMAGE_NAME /scripts/runCrisp-notls-shim.sh -r $ROLE 

# docker run -p 8888:8888 -h ${HOSTNAME}  --user=fluid:fluid  -v ${DATA_PATH}:/data:rw  -v ${SCRIPT_PATH}:/scripts:ro  --add-host agg-iss:$AGG_ISS_IP --add-host agg-earth:$AGG_EARTH_IP --add-host colab-iss:$COLAB_ISS_IP --add-host colab-shim:$COLAB_SHIM_IP --add-host colab-earth:$COLAB_EARTH_IP --restart no  $IMAGE_NAME /scripts/runCrisp-notls-shim.sh -r $ROLE 

#docker run -p 8888:8888 -h ${HOSTNAME}  -v ${DATA_PATH}:/data:rw  -v ${SCRIPT_PATH}:/scripts:ro  --add-host agg-iss:$AGG_ISS_IP --add-host agg-earth:$AGG_EARTH_IP --add-host colab-iss:$COLAB_ISS_IP --add-host colab-shim:$COLAB_SHIM_IP --add-host colab-earth:$COLAB_EARTH_IP --restart no  $IMAGE_NAME /scripts/runCrisp-notls-shim.sh -r $ROLE 

#docker run --user=${USER_NAME}:${GROUP_NAME} $RUNDOCKER_OPTS -h ${HOSTNAME}  -v ${DATA_PATH}:/data:rw  -v ${SCRIPT_PATH}:/scripts:ro  --add-host agg-iss:$AGG_ISS_IP --add-host agg-earth:$AGG_EARTH_IP --add-host colab-iss:$COLAB_ISS_IP --add-host colab-shim:$COLAB_SHIM_IP --add-host colab-earth:$COLAB_EARTH_IP --ip $IP --network $MYNET  --restart no --name $ROLE --detach $IMAGE_NAME /scripts/runCrisp-notls-shim.sh -r $ROLE 

#docker run --user=${USER_NAME}:${GROUP_NAME} $RUNDOCKER_OPTS -h ${HOSTNAME}  -v ${DATA_PATH}:/data:rw  -v ${SCRIPT_PATH}:/scripts:ro  --add-host agg-iss:$AGG_ISS_IP --add-host agg-earth:$AGG_EARTH_IP --add-host colab-iss:$COLAB_ISS_IP --add-host colab-shim:$COLAB_SHIM_IP --add-host colab-earth:$COLAB_EARTH_IP --ip $IP --network $MYNET  --restart no --name $ROLE -it $IMAGE_NAME 


#docker run --user=${USER_NAME}:${GROUP_NAME} $RUNDOCKER_OPTS -h ${HOSTNAME}  -v ${DATA_PATH}:/data:rw  -v ${SCRIPT_PATH}:/scripts:ro  --add-host agg-iss:$AGG_ISS_IP --add-host agg-earth:$AGG_EARTH_IP --add-host colab-iss:$COLAB_ISS_IP --add-host colab-shim:$COLAB_SHIM_IP --add-host colab-earth:$COLAB_EARTH_IP --ip $IP --network $MYNET  --restart no --name $ROLE --detach $IMAGE_NAME /scripts/runCrisp-notls-shim.sh -r $ROLE 

#docker run --user=${USER_NAME}:${GROUP_NAME} $RUNDOCKER_OPTS -h ${HOSTNAME}  -v ${CRISP_FLSRC_PATH}:/home/fluid/crisp/fl_src/erm_module.py:ro -v ${DATA_PATH}:/data:rw  -v ${SCRIPT_PATH}:/scripts:ro  --add-host agg-iss:$AGG_ISS_IP --add-host agg-earth:$AGG_EARTH_IP --add-host colab-iss:$COLAB_ISS_IP --add-host colab-shim:$COLAB_SHIM_IP --add-host colab-earth:$COLAB_EARTH_IP --ip $IP --network $MYNET  --restart no --name $ROLE --detach  $IMAGE_NAME /scripts/runCrisp-notls-shim.sh -r $ROLE

#docker run --user=${USER_NAME}:${GROUP_NAME} $RUNDOCKER_OPTS -h ${HOSTNAME}  -v ${CRISP_FLSRC_PATH}:/mnt -v ${DATA_PATH}:/data:rw  -v ${SCRIPT_PATH}:/scripts:ro  --add-host agg-iss:$AGG_ISS_IP --add-host agg-earth:$AGG_EARTH_IP --add-host colab-iss:$COLAB_ISS_IP --add-host colab-shim:$COLAB_SHIM_IP --add-host colab-earth:$COLAB_EARTH_IP --ip $IP --network $MYNET  --restart no --name $ROLE -it  $IMAGE_NAME 

docker run --user=${USER_NAME}:${GROUP_NAME} $RUNDOCKER_OPTS -h ${HOSTNAME}  -v ${CRISP_FLSRC_PATH}:/mnt -v ${DATA_PATH}:/data:rw  -v ${SCRIPT_PATH}:/scripts:ro  --add-host agg-moon:$AGG_MOON_IP --add-host agg-iss:$AGG_ISS_IP --add-host agg-earth:$AGG_EARTH_IP --add-host colab-iss:$COLAB_ISS_IP --add-host colab-shim-iss:$COLAB_SHIM_ISS_IP --add-host colab-shim-moon:$COLAB_SHIM_MOON_IP --add-host colab-earth:$COLAB_EARTH_IP --ip $IP --network $MYNET  --restart no --name $ROLE --detach  $IMAGE_NAME /scripts/runCrisp-notls-shim.sh -r $ROLE
