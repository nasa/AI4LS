#set -e
# Test the pipeline

N_COLABS=${1:-'5'}
TEMPLATE=${2:-'torch_cnn_mnist'}
FED_WORKSPACE=${3:-'fl_workspace'}   # This can be whatever unique directory name you want
FQDN=${4:-$(hostname --all-fqdns | awk '{print $1}')}

echo "MACHINE NAME: " $FQDN
echo "SIMULATED COLABS: " $N_COLABS

USER=user
COL_DATA_PATH=/home/${USER}/data

rm -rf ${COL_DATA_PATH}/save # remove if already exists
rm -rf ${COL_DATA_PATH}/plan # remove if already exists
mkdir ${COL_DATA_PATH}/plan
mkdir ${COL_DATA_PATH}/save
mkdir ${COL_DATA_PATH}/save/log

help() {
    echo "Usage: test_hello_federation.sh TEMPLATE FED_WORKSPACE COL1 COL2 [OPTIONS]"
    echo
    echo "Options:"
    echo "--rounds-to-train     rounds to train"
    echo "--col1-data-path      data path for collaborator 1"
    echo "--col2-data-path      data path for collaborator 2"
    echo "-h, --help            display this help and exit"
}

    echo $FED_WORKSPACE
    echo $FED_DIRECTORY
    echo $COL
    echo $COL_DIRECTORY
    echo $DATA_PATH

create_collaborator() {

    FED_WORKSPACE=$1
    FED_DIRECTORY=$2
    COL=$3
    COL_DIRECTORY=$4
    DATA_PATH=$5

    ARCHIVE_NAME="${FED_WORKSPACE}.zip"

    echo ""
    echo $FED_WORKSPACE
    echo $FED_DIRECTORY
    echo $COL
    echo $COL_DIRECTORY
    echo $DATA_PATH


    # Copy workspace to collaborator directories (these can be on different machines)
    rm -rf ${COL_DIRECTORY}    # Remove any existing directory
    mkdir -p ${COL_DIRECTORY}  # Create a new directory for the collaborator

    cd ${COL_DIRECTORY}
    fx workspace import --archive ${FED_DIRECTORY}/${ARCHIVE_NAME} # Import the workspace to this collaborator

    # mkdir ${FED_DIRECTORY}/${COL}/${FED_WORKSPACE}/data/${COL}
    # cp ${FED_DIRECTORY}/../data/${COL}/* ${FED_DIRECTORY}/${COL}/${FED_WORKSPACE}/data/${COL}/

    # Create collaborator certificate request
    cd ${COL_DIRECTORY}/${FED_WORKSPACE}
    fx collaborator generate-cert-request -d ${DATA_PATH}/${COL} -n ${COL} --silent # Remove '--silent' if you run this manually

    # Sign collaborator certificate
    cd ${FED_DIRECTORY}  # Move back to the Aggregator
    fx collaborator certify --request-pkg ${COL_DIRECTORY}/${FED_WORKSPACE}/col_${COL}_to_agg_cert_request.zip --silent # Remove '--silent' if you run this manually

    #Import the signed certificate from the aggregator
    cd ${COL_DIRECTORY}/${FED_WORKSPACE}
    fx collaborator certify --import ${FED_DIRECTORY}/agg_to_col_${COL}_signed_cert.zip

}

# START
# =====
# Make sure you are in a Python virtual environment with the FL package installed.

# Create FL workspace
rm -rf ${FED_WORKSPACE}
fx workspace create --prefix ${FED_WORKSPACE} --template ${TEMPLATE}

cd ${FED_WORKSPACE}
FED_DIRECTORY=`pwd`     # Get the absolute directory path for the workspace
touch requirements.txt  ## PD: absolute hack, to remove/clear the requirements in the workspace :)

# Copy the data over to the workspace
#cp -r ../data/* data/

# Initialize our FL plan
cp -r ../fl_plan/* plan/
cp -r ../fl_src/* src/

fx plan initialize -a ${FQDN}

# Set rounds to train if given
if [[ ! -z "$ROUNDS_TO_TRAIN" ]]
then
    sed -i "/rounds_to_train/c\    rounds_to_train: $ROUNDS_TO_TRAIN" plan/plan.yaml
fi

# Create certificate authority for workspace
fx workspace certify

# Export FL workspace
# cd ${FED_DIRECTORY}
# rm requirments.txt ## not working?
# touch requirements.txt

fx workspace export

# Create aggregator certificate
fx aggregator generate-cert-request --fqdn ${FQDN}

# Sign aggregator certificate
fx aggregator certify --fqdn ${FQDN} --silent # Remove '--silent' if you run this manually

counter=0
while [ $counter -lt ${N_COLABS} ]
do
    echo $counter

    # Create collaborator #0
    COL_DIRECTORY=${FED_DIRECTORY}/'col_'$counter
    create_collaborator ${FED_WORKSPACE} ${FED_DIRECTORY} 'col_'$counter ${COL_DIRECTORY} ${COL_DATA_PATH}
    counter=$((counter + 1))
done
