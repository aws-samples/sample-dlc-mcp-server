#!/bin/bash

# Build script for Enhanced DLC Image with DeepSeek-V3 and Text Recognition (Fixed)
set -e

IMAGE_NAME="pytorch-deepseek-enhanced"
TAG="latest"
DOCKERFILE="Dockerfile.deepseek-enhanced-fixed"

echo "Building enhanced DLC image with DeepSeek-V3 and text recognition..."
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
    echo "🚀 Features included:"
    echo "- DeepSeek-V3 (latest LLM)"
    echo "- Text Recognition (EasyOCR, Tesseract)"
    echo "- Multimodal inference capabilities"
    echo "- Optimized inference support"
    echo ""
    echo "💾 Memory Requirements:"
    echo "- Minimum: 8GB GPU memory"
    echo "- Recommended: 24GB+ GPU memory for full DeepSeek-V3"
    echo ""
    echo "🎮 Usage Examples:"
    echo ""
    echo "# Run container:"
    echo "docker run -it --gpus all ${IMAGE_NAME}:${TAG} bash"
    echo ""
    echo "# DeepSeek chat:"
    echo "docker run -it --gpus all ${IMAGE_NAME}:${TAG} python /opt/ml/code/deepseek_inference.py --chat"
    echo ""
    echo "# Text recognition:"
    echo "docker run -it --gpus all -v /path/to/images:/data ${IMAGE_NAME}:${TAG} python /opt/ml/code/text_recognition.py --image /data/image.jpg --model easyocr"
    echo ""
    echo "# Multimodal analysis:"
    echo "docker run -it --gpus all -v /path/to/images:/data ${IMAGE_NAME}:${TAG} python /opt/ml/code/multimodal_inference.py --image /data/image.jpg --question 'What is this document about?'"
else
    echo "❌ Build failed!"
    exit 1
fi
