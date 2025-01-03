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
data_path = os.path.join('/home', col_user, 'data')

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
    os.makedirs(workspace_dst, exist_ok=True)
    print("created workspace dir")
    print(str(Path(workspace_src)))
    print(str(Path(workspace_src) / archive_name), workspace_dst)
    scp.get(str(Path(workspace_src) / archive_name), workspace_dst)
    print("Received workspace zip from aggregator")
except Exception as err:
    logging.debug(err)
    logging.info('Error connecting to Host')

# 10) Import the workspace archive.
os.system('cd %s' % workspace_dst)
os.system('rm -rf %s' % folder_name)
os.system('fx workspace import --archive %s' % archive_name)
#os.system('cd %s' % folder_name)

# 12) Create a collaborator certificate request.
os.chdir(workspace_col)
col_name = 'bob'
col_0_data = os.path.join(data_path, 'col_0')
os.system('fx collaborator generate-cert-request -d %s -n %s' % (os.path.join(data_path, 'col_0'), col_name))

os.chdir(workspace_col)
scp.put('col_bob_to_agg_cert_request.zip', workspace_src)
print("Sent the cert back to the agg")
