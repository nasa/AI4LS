#!/bin/basdh

export ROUNDS=5
export EPOCHS=1

export CRISP_FLSRC_PATH=/Users/jcasalet/Desktop/CODES/NASA/AI4LS/FLUID/crisp/fl_src

export COLABS="colab-iss colab-earth colab-moon"
#export COLABS="colab-iss colab-earth"
export AGGS="agg-earth agg-iss agg-moon"
#export AGGS="agg-earth agg-iss"
export COLAB_SHIMS="colab-shim-iss colab-shim-moon"
#export COLAB_SHIMS="colab-shim-iss"

export COLABORATOR_COUNT=0
for colab in $COLABS
do
	((COLABORATOR_COUNT=$COLABORATOR_COUNT+1))
done


export DATA_PATH=/Users/jcasalet/Desktop/CODES/NASA/AI4LS/FLUID/data
export SCRIPT_PATH=/Users/jcasalet/Desktop/CODES/NASA/AI4LS/FLUID/SBC4/scripts


export AGG_ISS_DATA_PATH=/Users/jcasalet/Desktop/CODES/NASA/AI4LS/FLUID/data/WORKSPACE/AGG_ISS/
export AGG_EARTH_DATA_PATH=/Users/jcasalet/Desktop/CODES/NASA/AI4LS/FLUID/data/WORKSPACE/AGG_EARTH
export AGG_MOON_DATA_PATH=/Users/jcasalet/Desktop/CODES/NASA/AI4LS/FLUID/data/WORKSPACE/AGG_MOON

export COLAB_ISS_DATA_PATH=/Users/jcasalet/Desktop/CODES/NASA/AI4LS/FLUID/data/WORKSPACE/COLAB_ISS
export COLAB_EARTH_DATA_PATH=/Users/jcasalet/Desktop/CODES/NASA/AI4LS/FLUID/data/WORKSPACE/COLAB_EARTH
export COLAB_MOON_DATA_PATH=/Users/jcasalet/Desktop/CODES/NASA/AI4LS/FLUID/data/WORKSPACE/COLAB_MOON

export COLAB_SHIM_ISS_DATA_PATH=/Users/jcasalet/Desktop/CODES/NASA/AI4LS/FLUID/data/WORKSPACE/COLAB_SHIM_ISS
export COLAB_SHIM_MOON_DATA_PATH=/Users/jcasalet/Desktop/CODES/NASA/AI4LS/FLUID/data/WORKSPACE/COLAB_SHIM_MOON

export IMAGE_NAME=fluid
export USER_NAME=fluid
export GROUP_NAME=fluid

export WORKSPACE_ISS_AGG_DIR=/data/WORKSPACE
export WORKSPACE_ISS_COLAB_DIR=/data/WORKSPACE
export WORKSPACE_ISS_COLAB_SHIM_DIR=/data/WORKSPACE
export WORKSPACE_EARTH_AGG_DIR=/data/WORKSPACE
export WORKSPACE_EARTH_COLAB_DIR=/data/WORKSPACE
export WORKSPACE_EARTH_COLAB_SHIM_DIR=/data/WORKSPACE
export WORKSPACE_MOON_AGG_DIR=/data/WORKSPACE
export WORKSPACE_MOON_COLAB_DIR=/data/WORKSPACE
export WORKSPACE_MOON_COLAB_SHIM_DIR=/data/WORKSPACE

export AGG_EARTH_HOST=agg-earth
export AGG_ISS_HOST=agg-iss
export AGG_MOON_HOST=agg-moon
export AGG_EARTH_IP=192.168.1.10
export AGG_ISS_IP=192.168.1.11
export AGG_MOON_IP=192.168.1.12
export COLAB_EARTH_IP=192.168.1.13
export COLAB_ISS_IP=192.168.1.14
export COLAB_MOON_IP=192.168.1.15
export COLAB_SHIM_ISS_IP=192.168.1.16
export COLAB_SHIM_MOON_IP=192.168.1.17

export AGG_ISS_PORT=8888
export AGG_EARTH_PORT=8889
export AGG_MOON_PORT=8890

export MYNET=mynet
export SUBNET=192.168.1.0/24
