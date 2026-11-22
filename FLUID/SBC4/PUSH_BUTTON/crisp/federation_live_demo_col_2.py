#!/usr/bin/env python

from paramiko import SSHClient, AutoAddPolicy
from scp import SCPClient
from pathlib import Path
from getpass import getpass
import paramiko
import logging
import os


## BEFORE YOU RUN THIS DO THE FOLLOWING IN A TERMINAL:
# `git clone -b live_demo https://gitlab.com/frontierdevelopmentlab/astronaut-health/crisp.git`
# ssh-keygen -t rsa -b 4096 -C "user@email_address"
# Add your sshkey to MetaData:GCP registry before turning on agg machine: `cp /home/user/.ssh/id_rsa.pub .`

## GET DATA FROM SOMEWHERE
# gcsfuse --implicit-dirs -only-dir validation/Example0b_dim_inv_5_dim_spu_5_dim_unc_0_n_exp_3000_n_env_3/seed_0 ah21_data /home/user/data


## AND RUN federation_live_demo_col.py first


## 8) Make sure you have copied the workspace archive (.zip) from the aggregator node to the collaborator node.

agg_machine_name = 'paul-openfl-test-imagecopy-agg'
afqdn =  agg_machine_name + '.us-central1-a.c.fdl-us-astronaut-health.internal'

src_user = 'pduck_test'  # sshkey username
agg_user = 'paul'  # agg username
col_user = 'paul' # col username

archive_name = 'workspace.zip'
folder_name = archive_name.split('.')[0]
workspace_dst = '.'

workspace_src = os.path.join('/home', agg_user, 'crisp', folder_name)
workspace_col = os.path.join('/home', col_user, 'crisp', folder_name)

logger = paramiko.util.logging.getLogger()
hdlr = logging.FileHandler('app.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

try:
    ssh_client = SSHClient()
    ssh_client.set_missing_host_key_policy(AutoAddPolicy())
    k = paramiko.RSAKey.from_private_key_file(os.path.join('/home', col_user, '.ssh/id_rsa'))
    ssh_client.connect(afqdn, username=src_user, pkey = k)
    print("authenticated & logged-in")

    scp = SCPClient(ssh_client.get_transport())
    scp.get(str(Path(workspace_src) / 'agg_to_col_bob_signed_cert.zip'), workspace_col)
    print("Received signed certificates back.")

except Exception as err:
    logging.debug(err)
    logging.info('Error connecting to Host')

# Import the signed certificate
os.chdir(workspace_col)
os.system('fx collaborator certify --import agg_to_col_bob_signed_cert.zip')

## Final Process: run the collaborator
print("Starting colaborator node: bob")
os.system('fx collaborator start -n bob')
