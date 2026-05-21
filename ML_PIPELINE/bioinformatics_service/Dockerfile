FROM rocker/r-ver:4.3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Set R environment variables
ENV R_HOME=/usr/local/lib/R
ENV LD_LIBRARY_PATH=/usr/local/lib/R/lib:$LD_LIBRARY_PATH

# Install R packages
RUN R -e "install.packages('BiocManager', repos='https://cloud.r-project.org')"
RUN R -e "BiocManager::install(c('DESeq2', 'clusterProfiler', 'enrichplot', 'DOSE', 'org.Mm.eg.db', 'org.Hs.eg.db', 'AnnotationDbi'), update=FALSE, ask=FALSE)"
RUN R -e "install.packages('ggplot2', repos='https://cloud.r-project.org')"

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy proto files
COPY proto/ ./proto/

# Generate gRPC code
RUN mkdir -p generated && \
    touch generated/__init__.py && \
    python3 -m grpc_tools.protoc \
        -I./proto \
        --python_out=./generated \
        --grpc_python_out=./generated \
        ./proto/bioinformatics_service.proto && \
    sed -i 's/^import bioinformatics_service_pb2/from . import bioinformatics_service_pb2/' generated/bioinformatics_service_pb2_grpc.py

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Copy source code
COPY src/ ./src/
COPY r_scripts/ ./r_scripts/

# Expose gRPC port
EXPOSE 50054

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import grpc; channel = grpc.insecure_channel('localhost:50054'); channel.close()" || exit 1

# Run server
CMD ["python3", "-m", "src.server"]
