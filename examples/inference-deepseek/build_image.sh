#!/bin/bash

# Build script for DeepSeek custom DLC image

set -e

# Configuration
IMAGE_NAME="deepseek-pytorch-inference"
TAG="2.6.0-cpu-py312"
REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}"

echo "Building DeepSeek PyTorch Inference Image..."
echo "Image: ${IMAGE_NAME}:${TAG}"
echo "ECR Repository: ${ECR_REPO}"

# Authenticate with ECR
echo "Authenticating with ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_REPO}

# Create ECR repository if it doesn't exist
echo "Creating ECR repository if it doesn't exist..."
aws ecr describe-repositories --repository-names ${IMAGE_NAME} --region ${REGION} 2>/dev/null || \
aws ecr create-repository --repository-name ${IMAGE_NAME} --region ${REGION}

# Build the Docker image
echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:${TAG} .

# Tag for ECR
echo "Tagging image for ECR..."
docker tag ${IMAGE_NAME}:${TAG} ${ECR_REPO}:${TAG}
docker tag ${IMAGE_NAME}:${TAG} ${ECR_REPO}:latest

# Push to ECR
echo "Pushing image to ECR..."
docker push ${ECR_REPO}:${TAG}
docker push ${ECR_REPO}:latest

echo "✅ Build and push completed successfully!"
echo "Image URI: ${ECR_REPO}:${TAG}"
echo "Latest URI: ${ECR_REPO}:latest"

# Test the image locally (optional)
read -p "Do you want to test the image locally? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting container for local testing..."
    docker run -d -p 8080:8080 --name deepseek-test ${IMAGE_NAME}:${TAG}
    
    echo "Waiting for container to start..."
    sleep 10
    
    echo "Testing the container..."
    python test_inference.py --url http://localhost:8080
    
    echo "Stopping test container..."
    docker stop deepseek-test
    docker rm deepseek-test
fi

echo "Done!"
