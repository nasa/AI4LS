#############################
# FDL US 2021 - Astronaut Health Team
# 2021 Docker Version:v07
# James Casaletto 
# Mar 2022 
#############################

#FROM nvidia/cuda:11.4.0-base-centos7
FROM nvidia/cudagl:11.4.0-devel-centos7 

LABEL authors="jcasalet@ucsc.edu"

# Install some basic utilities
RUN yum install -y  net-tools  bzip2  zip  unzip  git which openssh-server


# create fluid user
RUN groupadd -g 1002 fluid
RUN echo fluid:fluid > /tmp/pass.txt
RUN useradd -u 1002 -g 1002 -s /bin/bash -d /home/fluid -m fluid
RUN chpasswd < /tmp/pass.txt


# Create directories & set default directory
USER fluid
WORKDIR /home/fluid
RUN git clone -b live_demo https://gitlab.com/frontierdevelopmentlab/astronaut-health/crisp.git

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/fluid/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.7 

RUN conda create -y -n venv_3.7 python=3.7 anaconda
SHELL ["conda", "init", "bash"]
SHELL ["conda", "run", "-n", "venv_3.7", "/bin/bash", "-c"]
RUN source ~/miniconda/etc/profile.d/conda.sh
RUN echo "conda activate venv_3.7" > .bash_profile

# Installing neccessary libraries
RUN pip install paramiko
RUN pip install scp
RUN pip install pyflakes==2.3.0
RUN pip install pycodestyle==2.7.0
RUN pip install -r crisp/requirements.txt
RUN pip install torch
RUN pip install torchvision
RUN conda install tensorflow-gpu cudatoolkit=10.1

# openfl:
WORKDIR /home/fluid
#RUN git clone --single-branch --branch fdl2021 https://github.com/PDuckworth/openfl.git
#RUN git clone --branch fluid https://github.com/intel/openfl.git
RUN git clone --branch fluid_rebase https://github.com/intel/openfl.git
RUN pip install /home/fluid/openfl/.

# ssh keys
RUN mkdir .ssh
RUN chmod 700 .ssh
ADD ./FILES/id_rsa.pub .ssh
ADD ./FILES/id_rsa .ssh
ADD ./FILES/known_hosts .ssh
ADD ./FILES/authorized_keys .ssh

RUN sh -c 'echo -e IMAGE COMPLETED - READY TO RUN'

# Set the default command to python3
#CMD ["python3"]
#CMD tail -f /dev/null
