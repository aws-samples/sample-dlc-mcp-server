# DLC MCP Server Examples

This directory contains example use cases for AWS Deep Learning Containers.

## Examples

### [inference-deepseek](./inference-deepseek/)
DeepSeek model inference using vLLM container with custom configuration.

### [deepseek-enhanced](./deepseek-enhanced/)
Enhanced DeepSeek deployment with optimized settings.

### [text-recognition](./text-recognition/)
Text recognition model deployment using PyTorch DLC.

### [nemo](./nemo/)
NVIDIA NeMo framework example for speech and NLP models.

### [cifar10](./cifar10/)
CIFAR-10 image classification training example with PyTorch.

## Usage

Each example contains:
- `Dockerfile` - Custom container configuration
- `build.sh` - Build script (where applicable)
- Python scripts for training/inference

To build an example:
```bash
cd <example-directory>
./build.sh
```

## Prerequisites

- Docker installed
- AWS CLI configured
- ECR login for DLC base images:
  ```bash
  aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com
  ```
