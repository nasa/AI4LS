#!/bin/bash


if [ $# -ne 1 ]
then
	echo "usage $0 iss | moon | both"
	exit 1
fi

SYNC=$1

while true
do
	if [ $SYNC == "iss" ]
	then
		rsync -a ../data/WORKSPACE/AGG_ISS/WORKSPACE/workspace/request_path/ ../data/WORKSPACE/COLAB_SHIM_ISS/WORKSPACE/workspace/request_path
		rsync -a ../data/WORKSPACE/AGG_ISS/WORKSPACE/workspace/response_path/ ../data/WORKSPACE/COLAB_SHIM_ISS/WORKSPACE/workspace/response_path

		rsync -a ../data/WORKSPACE/COLAB_SHIM_ISS/WORKSPACE/workspace/request_path/ ../data/WORKSPACE/AGG_ISS/WORKSPACE/workspace/request_path/
		rsync -a ../data/WORKSPACE/COLAB_SHIM_ISS/WORKSPACE/workspace/response_path/ ../data/WORKSPACE/AGG_ISS/WORKSPACE/workspace/response_path/

	elif [ $SYNC == "moon" ]
	then
		rsync -a ../data/WORKSPACE/AGG_MOON/WORKSPACE/workspace/request_path/ ../data/WORKSPACE/COLAB_SHIM_MOON/WORKSPACE/workspace/request_path
		rsync -a ../data/WORKSPACE/AGG_MOON/WORKSPACE/workspace/response_path/ ../data/WORKSPACE/COLAB_SHIM_MOON/WORKSPACE/workspace/response_path

		rsync -a ../data/WORKSPACE/COLAB_SHIM_MOON/WORKSPACE/workspace/request_path/ ../data/WORKSPACE/AGG_MOON/WORKSPACE/workspace/request_path/
		rsync -a ../data/WORKSPACE/COLAB_SHIM_MOON/WORKSPACE/workspace/response_path/ ../data/WORKSPACE/AGG_MOON/WORKSPACE/workspace/response_path/

	elif [ $SYNC == "both" ]
	then
		rsync -a ../data/WORKSPACE/AGG_ISS/WORKSPACE/workspace/request_path/ ../data/WORKSPACE/COLAB_SHIM_ISS/WORKSPACE/workspace/request_path
		rsync -a ../data/WORKSPACE/AGG_ISS/WORKSPACE/workspace/response_path/ ../data/WORKSPACE/COLAB_SHIM_ISS/WORKSPACE/workspace/response_path

		rsync -a ../data/WORKSPACE/COLAB_SHIM_ISS/WORKSPACE/workspace/request_path/ ../data/WORKSPACE/AGG_ISS/WORKSPACE/workspace/request_path/
		rsync -a ../data/WORKSPACE/COLAB_SHIM_ISS/WORKSPACE/workspace/response_path/ ../data/WORKSPACE/AGG_ISS/WORKSPACE/workspace/response_path/

		rsync -a ../data/WORKSPACE/AGG_MOON/WORKSPACE/workspace/request_path/ ../data/WORKSPACE/COLAB_SHIM_MOON/WORKSPACE/workspace/request_path
		rsync -a ../data/WORKSPACE/AGG_MOON/WORKSPACE/workspace/response_path/ ../data/WORKSPACE/COLAB_SHIM_MOON/WORKSPACE/workspace/response_path

		rsync -a ../data/WORKSPACE/COLAB_SHIM_MOON/WORKSPACE/workspace/request_path/ ../data/WORKSPACE/AGG_MOON/WORKSPACE/workspace/request_path/
		rsync -a ../data/WORKSPACE/COLAB_SHIM_MOON/WORKSPACE/workspace/response_path/ ../data/WORKSPACE/AGG_MOON/WORKSPACE/workspace/response_path/

	else
		echo "usage $0 iss | moon | both"
		exit 1

	fi

	sleep 5
done
