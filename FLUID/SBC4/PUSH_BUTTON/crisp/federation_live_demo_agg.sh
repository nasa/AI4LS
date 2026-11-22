#set -e
# Script for the Aggregator node

FED_WORKSPACE=${1:-'workspace'}   # This can be whatever unique directory name you want
FQDN=${2:-$(hostname --all-fqdns | awk '{print $1}')}
TEMPLATE=${2:-'torch_cnn_mnist'}

echo "MACHINE NAME: " $FQDN
echo "SIMULATED COLABS: " $N_COLABS

### Totally collaborator agnostic :)

# START
# =====

# 2) Create FL workspace
rm -rf ${FED_WORKSPACE}
fx workspace create --prefix ${FED_WORKSPACE} --template ${TEMPLATE}
chmod 777 ${FED_WORKSPACE}
cd ${FED_WORKSPACE}

FED_DIRECTORY=`pwd`     # Get the absolute directory path for the workspace
touch requirements.txt  ## PD: absolute hack, to remove/clear the requirements in the workspace :)

# 3) Initialize our FL plan
cp -r ../fl_plan/* plan/   # This is where all the collab detils should be
cp -r ../fl_src/* src/
fx plan initialize -a ${FQDN}

# 4) Create certificate authority for workspace
fx workspace certify

# 5) Create aggregator certificate
fx aggregator generate-cert-request --fqdn ${FQDN}

# 6) Sign aggregator certificate
fx aggregator certify --fqdn ${FQDN} --silent # Remove '--silent' if you run this manually

#7) Export the workspace so that it can be imported to the col nodes
fx workspace export
