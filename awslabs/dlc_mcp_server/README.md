# DeepSeek PyTorch Inference Custom DLC Image

This project creates a custom Docker image based on AWS Deep Learning Container (DLC) for PyTorch inference, specifically configured for DeepSeek models.

## Base Image
- **Base**: `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-cpu-py312-ubuntu22.04-ec2-v1.6`
- **PyTorch Version**: 2.6.0
- **Python Version**: 3.12
- **Platform**: CPU optimized
- **OS**: Ubuntu 22.04

## Features
- Pre-configured for DeepSeek models (default: deepseek-coder-6.7b-instruct)
- Flask-based inference server
- Health check endpoints
- Optimized for CPU inference
- Easy to customize for different DeepSeek models

## Files Structure
```
.
├── Dockerfile              # Main Docker configuration
├── inference.py           # DeepSeek inference server
├── requirements.txt       # Python dependencies
├── build_image.sh        # Build and push script
├── test_inference.py     # Testing script
└── README.md            # This file
```

## Quick Start

### 1. Build the Image
```bash
./build_image.sh
```

This script will:
- Authenticate with ECR
- Create ECR repository if needed
- Build the Docker image
- Push to your ECR repository
- Optionally test locally

### 2. Run Locally
```bash
# Build locally
docker build -t deepseek-pytorch-inference:latest .

# Run container
docker run -d -p 8080:8080 --name deepseek-inference deepseek-pytorch-inference:latest

# Test the inference
python test_inference.py
```

### 3. Test the Server
```bash
# Health check
curl http://localhost:8080/ping

# Root endpoint
curl http://localhost:8080/

# Inference
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": "Write a Python function to calculate fibonacci numbers",
    "max_length": 512,
    "temperature": 0.7
  }'
```

## Configuration

### Environment Variables
- `MODEL_NAME`: DeepSeek model to use (default: deepseek-ai/deepseek-coder-6.7b-instruct)
- `HF_HOME`: Hugging Face cache directory (default: /opt/ml/model)
- `TRANSFORMERS_CACHE`: Transformers cache directory (default: /opt/ml/model)
- `PORT`: Server port (default: 8080)

### Supported DeepSeek Models
You can customize the model by changing the `MODEL_NAME` environment variable:
- `deepseek-ai/deepseek-coder-6.7b-instruct` (default)
- `deepseek-ai/deepseek-coder-1.3b-instruct`
- `deepseek-ai/deepseek-coder-33b-instruct`
- `deepseek-ai/deepseek-llm-7b-chat`

## API Endpoints

### Health Check
- **GET** `/ping` - Returns health status

### Root
- **GET** `/` - Returns server information

### Inference
- **POST** `/invocations` - Main inference endpoint

#### Request Format
```json
{
  "inputs": "Your prompt here",
  "max_length": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "do_sample": true
}
```

#### Response Format
```json
{
  "generated_text": "Generated response",
  "full_response": "Full response including prompt",
  "input_length": 50,
  "output_length": 150
}
```

## Deployment Options

### 1. Amazon SageMaker
```python
import boto3

sagemaker = boto3.client('sagemaker')

# Create model
model_name = 'deepseek-pytorch-model'
image_uri = 'YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/deepseek-pytorch-inference:latest'

sagemaker.create_model(
    ModelName=model_name,
    PrimaryContainer={
        'Image': image_uri,
        'Environment': {
            'MODEL_NAME': 'deepseek-ai/deepseek-coder-6.7b-instruct'
        }
    },
    ExecutionRoleArn='arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole'
)
```

### 2. Amazon ECS
```yaml
# ECS Task Definition
family: deepseek-inference
cpu: 2048
memory: 4096
containerDefinitions:
  - name: deepseek-container
    image: YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/deepseek-pytorch-inference:latest
    portMappings:
      - containerPort: 8080
        hostPort: 8080
    environment:
      - name: MODEL_NAME
        value: deepseek-ai/deepseek-coder-6.7b-instruct
```

### 3. Amazon EKS
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepseek-inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: deepseek-inference
  template:
    metadata:
      labels:
        app: deepseek-inference
    spec:
      containers:
      - name: deepseek-container
        image: YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/deepseek-pytorch-inference:latest
        ports:
        - containerPort: 8080
        env:
        - name: MODEL_NAME
          value: deepseek-ai/deepseek-coder-6.7b-instruct
```

## Performance Considerations

### CPU Optimization
- The image is optimized for CPU inference
- Consider using instances with high CPU count (c5.xlarge, c5.2xlarge, etc.)
- Model loading can take 2-5 minutes depending on model size

### Memory Requirements
- DeepSeek 1.3B: ~4GB RAM
- DeepSeek 6.7B: ~8GB RAM
- DeepSeek 33B: ~32GB RAM

### Scaling
- Use load balancers for multiple instances
- Consider model caching strategies
- Monitor memory usage and CPU utilization

## Troubleshooting

### Common Issues
1. **Model loading timeout**: Increase health check timeout
2. **Out of memory**: Use smaller model or increase instance memory
3. **Slow inference**: Consider GPU instances for larger models

### Logs
```bash
# Check container logs
docker logs deepseek-inference

# Follow logs
docker logs -f deepseek-inference
```

## Customization

### Different Models
Modify the `MODEL_NAME` environment variable in the Dockerfile or at runtime.

### Additional Dependencies
Add packages to `requirements.txt` and rebuild the image.

### Custom Inference Logic
Modify `inference.py` to implement custom preprocessing or postprocessing logic.

## Security Best Practices
- Use IAM roles for AWS service access
- Store sensitive data in AWS Secrets Manager
- Use VPC endpoints for private communication
- Enable CloudTrail logging
- Regularly update base images

## Support
For issues related to:
- DeepSeek models: Check Hugging Face model pages
- AWS DLC: AWS documentation
- Custom implementation: Check logs and error messages
