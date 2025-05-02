#!/bin/bash

source crisp-config.sh

for container in $COLABS $AGGS $COLAB_SHIM
do
	docker container stop $container
	docker container rm $container
done 


docker volume prune -f
docker builder prune -f
