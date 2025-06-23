#!/bin/bash

# Build script for Text Recognition DLC Image
set -e

IMAGE_NAME="pytorch-text-recognition"
TAG="latest"
DOCKERFILE="Dockerfile.text-recognition"

echo "Building custom DLC image with text recognition capabilities..."
echo "Base image: 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.7.1-gpu-py312-ec2"
echo "Target image: ${IMAGE_NAME}:${TAG}"

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE" ]; then
    echo "Error: $DOCKERFILE not found!"
    exit 1
fi

# Build the image
echo "Starting Docker build..."
docker build -t ${IMAGE_NAME}:${TAG} -f ${DOCKERFILE} .

if [ $? -eq 0 ]; then
    echo "✅ Build completed successfully!"
    echo "Image: ${IMAGE_NAME}:${TAG}"
    echo ""
    echo "To run the container:"
    echo "docker run -it --gpus all ${IMAGE_NAME}:${TAG} bash"
    echo ""
    echo "To test text recognition:"
    echo "docker run -it --gpus all -v /path/to/your/images:/opt/ml/data ${IMAGE_NAME}:${TAG} python /opt/ml/code/text_recognition.py --image /opt/ml/data/your_image.jpg --model trocr"
else
    echo "❌ Build failed!"
    exit 1
fi
