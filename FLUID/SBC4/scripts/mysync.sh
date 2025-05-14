#!/bin/bash



while true
do
	rsync -a ../../data/WORKSPACE/AGG_ISS/WORKSPACE/workspace/request_path/ ../../data/WORKSPACE/COLAB_SHIM_ISS/WORKSPACE/workspace/request_path
	rsync -a ../../data/WORKSPACE/AGG_ISS/WORKSPACE/workspace/response_path/ ../../data/WORKSPACE/COLAB_SHIM_ISS/WORKSPACE/workspace/response_path

	rsync -a ../../data/WORKSPACE/AGG_MOON/WORKSPACE/workspace/request_path/ ../../data/WORKSPACE/COLAB_SHIM_MOON/WORKSPACE/workspace/request_path
	rsync -a ../../data/WORKSPACE/AGG_MOON/WORKSPACE/workspace/response_path/ ../../data/WORKSPACE/COLAB_SHIM_MOON/WORKSPACE/workspace/response_path

	rsync -a ../../data/WORKSPACE/COLAB_SHIM_ISS/WORKSPACE/workspace/request_path/ ../../data/WORKSPACE/AGG_ISS/WORKSPACE/workspace/request_path/
	rsync -a ../../data/WORKSPACE/COLAB_SHIM_ISS/WORKSPACE/workspace/response_path/ ../../data/WORKSPACE/AGG_ISS/WORKSPACE/workspace/response_path/

	rsync -a ../../data/WORKSPACE/COLAB_SHIM_MOON/WORKSPACE/workspace/request_path/ ../../data/WORKSPACE/AGG_MOON/WORKSPACE/workspace/request_path/
	rsync -a ../../data/WORKSPACE/COLAB_SHIM_MOON/WORKSPACE/workspace/response_path/ ../../data/WORKSPACE/AGG_MOON/WORKSPACE/workspace/response_path/


	sleep 5
done
