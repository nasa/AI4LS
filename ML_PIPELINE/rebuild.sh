#!/bin/bash

if [ $# -eq 1 ]
then
	svc=$1
        echo rebuilding $svc
	docker-compose down $svc
	image=$(docker image list | grep "$svc" | awk '{print $3}')
	docker image rm -f $image 

else
	docker-compose down 
	images=$(docker image list | grep "$svc" | awk '{print $3}' | sed -n '2,$ p' | xargs)	
	docker image rm -f $images
fi

docker system prune -f
docker-compose build $svc 
docker-compose up -d $svc
