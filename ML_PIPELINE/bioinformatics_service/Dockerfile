# bioinformatics_service/Dockerfile
FROM rocker/r-ver:4.3

WORKDIR /app

# Install Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libgit2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set R library path for rpy2
ENV R_HOME=/usr/local/lib/R
ENV LD_LIBRARY_PATH=/usr/local/lib/R/lib:${LD_LIBRARY_PATH}

# Install R packages
RUN R -e "install.packages('BiocManager', repos='http://cran.rstudio.com/')" && \
    R -e "BiocManager::install(c('DESeq2', 'clusterProfiler', 'enrichplot', 'DOSE', 'org.Mm.eg.db', 'org.Hs.eg.db', 'AnnotationDbi'), ask=FALSE)" && \
    R -e "install.packages('ggplot2', repos='http://cran.rstudio.com/')"

# Verify R packages
RUN R -e "library(DESeq2); library(clusterProfiler); library(enrichplot)"

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy proto files
COPY proto/ ./proto/

# Generate gRPC code
RUN mkdir -p generated && \
    touch generated/__init__.py && \
    python -m grpc_tools.protoc \
        -I./proto \
        --python_out=./generated \
        --grpc_python_out=./generated \
        ./proto/bioinformatics_service.proto && \
    sed -i 's/^import bioinformatics_service_pb2/from . import bioinformatics_service_pb2/' generated/bioinformatics_service_pb2_grpc.py && \
    python -m grpc_tools.protoc \
        -I./proto \
        --python_out=./generated \
        --grpc_python_out=./generated \
        ./proto/data_service.proto && \
    sed -i 's/^import data_service_pb2/from . import data_service_pb2/' generated/data_service_pb2_grpc.py

ENV PYTHONPATH=/app

COPY r_scripts/ ./r_scripts/
COPY src/ ./src/

EXPOSE 50054

CMD ["python", "-m", "src.server"]
