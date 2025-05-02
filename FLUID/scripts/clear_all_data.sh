#!/bin/bash

cd /Users/jcasalet/Desktop/CODES/NASA/AI4LS/FLUID/data/WORKSPACE

for d in AGG_EARTH AGG_ISS COLAB_EARTH COLAB_ISS COLAB_SHIM
do
	rm -rf ${d}/workspace
	rm -rf ${d}/col_0 ${d}/col_1
done
