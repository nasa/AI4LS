#!/bin/bash

source crisp-config.sh

for c in $COLABS $AGGS $COLAB_SHIMS
do
	docker container stop $c
	docker container rm $c
done 


docker volume prune -f
docker builder prune -f
