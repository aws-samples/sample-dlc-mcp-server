FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.7.1-gpu-py312-cu128-ubuntu22.04-ec2-v1.2

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NEMO_HOME=/workspace/nemo
ENV PYTHONPATH=/workspace

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    libsndfile1 \
    ffmpeg \
    sox \
    libsox-fmt-all \
    vim \
    htop \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    nemo-toolkit[all] \
    pytorch-lightning \
    hydra-core \
    omegaconf \
    librosa \
    soundfile \
    datasets \
    transformers \
    accelerate \
    tensorboard

# Custom commands
RUN mkdir -p /workspace/nemo && \
    cd /workspace && git clone https://github.com/NVIDIA/NeMo.git nemo-repo && \
    pip install --no-cache-dir -e /workspace/nemo-repo/. && \
    mkdir -p /workspace/models /workspace/data /workspace/experiments && \
    echo 'NeMo toolkit installation completed'

WORKDIR /workspace

# Expose common ports
EXPOSE 8888 6006

# Default command
CMD ["bash"]
