# AWS Deep Learning Containers MCP Server

A Model Context Protocol (MCP) server for AWS Deep Learning Containers (DLC) that assists with:

- Building custom DLC images
- Deploying DLC images to various AWS platforms (EC2, SageMaker, ECS, EKS)
- Upgrading existing images to newer framework versions
- Troubleshooting DLC-related issues
- Providing best practices guidance for DLC usage

## Prerequisites

Before using the AWS Deep Learning Containers MCP Server, ensure you have the following set up:

### 1. AWS Q CLI
Install and configure the AWS Q CLI 
- [Installing AWS Q CLI](https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/command-line-installing.html)

### 2. Model Context Protocol (MCP)
Set up MCP to enable seamless integration with AWS Q:
- [Understanding and Configuring MCP](https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/command-line-mcp-understanding-config.html)

### 3. Docker Environment
Choose one of the following options:
- **Local Development**: [Install Docker Desktop](https://docs.docker.com/get-started/introduction/get-docker-desktop/)
- **Cloud Development**: Set up an EC2 instance with Docker installed

### 4. AWS Credentials and Configuration
Configure your AWS credentials using one of these methods:
```bash
$ aws configure
AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
Default region name [None]: us-west-2
Default output format [None]: json

$ aws configure set aws_session_token fcZib3JpZ2luX2IQoJb3JpZ2luX2IQoJb3JpZ2luX2IQoJb3JpZ2luX2IQoJb3JpZVERYLONGSTRINGEXAMPLE

$ ada credentials update --once 
```



Other authentication methods: 
- **Configuration Files**: [AWS CLI Configuration and Credential Files](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-files.html)
- **Authentication Methods**: [AWS CLI User Authentication](https://docs.aws.amazon.com/cli/v1/userguide/cli-authentication-user.html)

## Installation

```bash
pip install awslabs.dlc-mcp-server
```

## Usage

Start the server:

```bash
dlc-mcp-server
```

By default, the server runs on port 8080. You can configure this with the `FASTMCP_PORT` environment variable.

## Example Prompts

Here are some example prompts you can use with the AWS DLC MCP Server:

### Building Custom Images
```
1. "Please add the latest version of DeepSeek model to my DLC"

2. "Use latest 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training as base image and can you build a custom image that has image recognition model"

3. "Create a custom pytorch image with text recognition model"

4. "Please create a container from the latest DLC version"
```

### Managing and Upgrading Containers
```
5. "Can you list available DLC images"

6. "Please update the container with the nightly version of PyTorch"

7. "Please update DLC with the latest version of CUDA"

8. "Please add Nemo toolkit (https://github.com/NVIDIA/NeMo) to my container"
```

### Advanced Customizations
```
9. "Add NVIDIA NeMo Framework to my existing PyTorch DLC image"

10. "Optimize my DLC image for inference workloads on GPU instances"

11. "Create a multi-stage build for my custom DLC with minimal runtime footprint"
```

## Features

### DLC Image Building

- Create custom DLC images based on AWS base images
- Add custom dependencies and packages
- Optimize images for specific use cases
- Support for popular ML frameworks and models

### DLC Deployment

- Deploy DLC images to Amazon EC2
- Deploy DLC images to Amazon SageMaker
- Deploy DLC images to Amazon ECS
- Deploy DLC images to Amazon EKS

### DLC Troubleshooting

- Diagnose common DLC issues
- Provide solutions for framework-specific problems
- Optimize performance for training and inference

### DLC Upgrades

- Upgrade existing DLC images to newer framework versions
- Migrate between framework versions with minimal disruption
- Apply security patches and updates

### Best Practices

- Recommendations for DLC image optimization
- Security best practices for DLC
- Performance tuning for ML workloads

## Configuration

The server can be configured using environment variables:

- `FASTMCP_PORT`: Port to run the server on (default: 8080)
- `FASTMCP_LOG_LEVEL`: Logging level (default: INFO)
- `FASTMCP_LOG_FILE`: Path to log file (optional)
- `ALLOW_WRITE`: Enable write operations (default: false)
- `ALLOW_SENSITIVE_DATA`: Enable access to sensitive data (default: false)

## Supported Frameworks and Tools

The MCP server supports building and managing containers with:

- **Deep Learning Frameworks**: PyTorch, TensorFlow, MXNet, Hugging Face Transformers
- **NVIDIA Tools**: CUDA, cuDNN, NeMo Toolkit, TensorRT
- **Popular Models**: DeepSeek, LLaMA, BERT, ResNet, and custom models
- **Specialized Libraries**: Computer Vision, NLP, Speech Recognition, and more

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for development instructions.

## License

Apache 2.0