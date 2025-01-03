#set -e
# Second Aggregator Script

N_COLABS=${1:-'1'}
FED_WORKSPACE=${2:-'workspace'}   # This can be whatever unique directory name you want
FQDN=${4:-$(hostname --all-fqdns | awk '{print $1}')}

echo "MACHINE NAME: " $FQDN
echo "SIMULATED COLABS: " $N_COLABS


# CONTINUE
# =====
cd ${FED_WORKSPACE}

# 14) Sign collaborators certificates
fx collaborator certify -s --request-pkg col_bob_to_agg_cert_request.zip # automatically adds "col"
#fx collaborator certify -s --request-pkg col_betty_to_agg_cert_request.zip


#15) Start the Aggregator server
fx aggregator start
