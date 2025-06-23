FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-cpu-py312-ubuntu22.04-ec2-v1.6

# Set environment variables
ENV HF_HOME=/opt/ml/model
ENV MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-instruct
ENV TRANSFORMERS_CACHE=/opt/ml/model
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    transformers>=4.36.0 \
    torch \
    accelerate \
    sentencepiece \
    protobuf \
    tiktoken \
    flask \
    gunicorn \
    requests

# Create directories
RUN mkdir -p /opt/ml/model
RUN mkdir -p /opt/ml/code

# Copy inference script
COPY inference.py /opt/ml/code/

# Set working directory
WORKDIR /opt/ml/code

# Make inference script executable
RUN chmod +x /opt/ml/code/inference.py

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/ping || exit 1

# Default command
CMD ["python", "inference.py"]
