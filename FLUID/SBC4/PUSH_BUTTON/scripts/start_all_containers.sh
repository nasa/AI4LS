#!/bin/bash

source crisp-config.sh

for c in $COLABS $AGGS $COLAB_SHIMS
do
	./runDocker-notls-shim.sh -r $c
done
