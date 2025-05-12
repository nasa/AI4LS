#!/bin/bash

source crisp-config.sh

for container in colab-shim-moon colab-shim-iss agg-moon agg-iss agg-earth colab-moon colab-iss colab-earth 
do
	docker container stop $container
	docker container rm $container
done 


docker volume prune -f
docker builder prune -f
