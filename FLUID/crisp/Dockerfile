#############################
# FDL US 2021 - Astronaut Health Team
# 2021 Docker Version:v07
# Paul Duckworth, Linus Scheibenreif & Frank Soboczenski
# 2021, July
#############################

FROM nvidia/cuda:11.4.0-base-ubuntu20.04

LABEL authors="frank.soboczenski@gmail.com, pduckworth@oxfordrobotics.institute, scheibenreif.linus@gmail.com"

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    ca-certificates \
    sudo \
    bash \
    git \
    bzip2 \
    libx11-6 \
    nano \
    curl \
  && echo "deb http://packages.cloud.google.com/apt cloud-sdk-stretch main" | tee /etc/apt/sources.list.d/google-cloud.sdk.list \
  && echo "deb http://packages.cloud.google.com/apt gcsfuse-stretch main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list \
  && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
  && apt-get update \
  && apt-get install --yes google-cloud-sdk gcsfuse \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create directories & set default directory
RUN mkdir /home/user
RUN mkdir /home/user/data
RUN mkdir /home/user/crisp
COPY . /home/user/crisp

WORKDIR /home/user

# Create a non-root user and switch to it
#RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
# && chown -R user:user /app
#RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
#USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.8.3\
 && conda clean -ya\
 && conda update conda

# CUDA 11.1-specific steps
RUN conda install -y -c conda-forge cudatoolkit=11.1.1 \
 && conda install -y -c conda-forge tmux \
 && conda install -y -c pytorch \
     "pytorch=1.8.1=py3.8_cuda11.1_cudnn8.0.5_0" \
     "torchvision=0.9.1=py38_cu111" \
 && conda clean -ya

# Installing neccessary libraries
RUN pip install -r crisp/requirements.txt

# openfl:
RUN git clone --single-branch --branch fdl2021 https://github.com/PDuckworth/openfl.git
RUN pip install /home/user/openfl/.

# Mount the GCP Bucket via GCSFuse
# RUN sudo gcsfuse --implicit-dirs -only-dir validation ah21_data /app/data

RUN sh -c 'echo -e IMAGE COMPLETED - READY TO RUN'

# Set the default command to python3
CMD ["python3"]
CMD tail -f /dev/null
