#!/bin/bash

while true
do
	cp ../data/WORKSPACE/AGG_ISS/WORKSPACE/workspace/request_path/* ../data/WORKSPACE/COLAB_SHIM/WORKSPACE/workspace/request_path
	cp ../data/WORKSPACE/AGG_ISS/WORKSPACE/workspace/response_path/* ../data/WORKSPACE/COLAB_SHIM/WORKSPACE/workspace/response_path

	cp ../data/WORKSPACE/COLAB_SHIM/WORKSPACE/workspace/request_path/* ../data/WORKSPACE/AGG_ISS/WORKSPACE/workspace/request_path/
	cp ../data/WORKSPACE/COLAB_SHIM/WORKSPACE/workspace/response_path/* ../data/WORKSPACE/AGG_ISS/WORKSPACE/workspace/response_path/


	sleep 5
done
