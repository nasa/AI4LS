#############################
# FDL - Astronaut Health Team
# 2020 Docker Version:v01
# Frank Soboczenski
# 2020, July
#############################

FROM nvidia/cuda:10.2-base-ubuntu18.04

MAINTAINER "Frank Soboczenski <frank.soboczenski@gmail.com>"

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    bash \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
COPY . /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory

ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.7 \
 && conda clean -ya

# && conda install -y python==3.8.1 \

# CUDA 10.2-specific steps
RUN conda install -y -c pytorch \
    cudatoolkit=10.2 \
    "pytorch=1.5.0=py3.7_cuda10.2.89_cudnn7.6.5_0" \
    "torchvision=0.6.0=py37_cu102" \
 && conda clean -ya

# Installing neccessary libraries
RUN pip install -r requirements.txt
RUN sh -c 'echo -e IMAGE COMPLETED - READY TO RUN'

# Set the default command to python3
CMD ["python3"]
CMD tail -f /dev/null
