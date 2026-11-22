#!/bin/bash

if [ $# -ne 1 ]
then
	echo "usage $0 agg-earth | colab-earth | agg-iss | colab-iss | agg-moon | colab-moon"
	exit 1
fi

HOST=$1

docker logs -f $HOST > ${HOST}-log.txt 2>& 1
