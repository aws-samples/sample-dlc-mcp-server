# AWS Deep Learning Containers MCP Server

A comprehensive Model Context Protocol (MCP) server for AWS Deep Learning Containers (DLC) that provides end-to-end support for machine learning workflows. This server offers six core service modules to help you build, deploy, upgrade, and optimize your DLC-based ML infrastructure.

## Quick Start Guide

### Installation Steps

### 1. **Prerequisites**:
   - Create an AWS Instance Profile with the following policies. Use this profile while creating EC2 instance in the next step. 
     - AmazonECS_FullAccess Policy 
     - AmazonEC2ContainerRegistryFullAccess
   - EC2 with [DLC Image](https://docs.aws.amazon.com/dlami/latest/devguide/overview-base.html) recommended.
     - Launch an Amazon Elastic Compute Cloud instance (CPU or GPU), preferably a Deep Learning Base AMI. Other AMIs work but require relevant GPU drivers. If you prefer to work with local docker desktop setup on your machine, then you can skip to step. 
   - AWS CLI
   - Python 3.11 or later
   - Install uv (pip install uv) to run the mcp server locally
   - Docker (DLC Image contains Docker)
   - Connect to your instance by using SSH. For more information about connections, see [Troubleshooting Connecting to Your Instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/TroubleshootingInstancesConnecting.html) in the Amazon EC2 user guide..
   - Install and configure the AWS Q CLI with this [guide](https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/command-line-installing-ssh-setup-autocomplete.html)
	

### 2. Configure DLC MCP Server


```bash
#clone the repo
git clone https://github.com/aws-samples/sample-dlc-mcp-server.git
cd sample-dlc-mcp-server
# Build and install MCP server
python3 -m pip install -e .

# Verify the server start
dlc-mcp-server
```

```bash
# Update ~/.aws/amazonq/mcp.json
# Update the path to your local sample-dlc-mcp-server directory
echo "{ "mcpServers": { "dlc-mcp-server": { "command": "uv", "args": [ "--directory", "<<Update-directory-path>>/sample-dlc-mcp-server", "run", "dlc-mcp-server" ], "env": {}, "timeout": 120000 } } }” > ~/.aws/amazonq/mcp.json```
```
### 3. AWS Credentials and Configuration
   - AWS Credentials and Configuration 
     - Configure your AWS credentials using one of these methods:
      ```bash
         aws configure
         aws configure set aws_session_token <<token>>
         
      ```
   - Other authentication methods:
     - Configuration Files: [AWS CLI Configuration and Credential Files](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-files.html)
     - Authentication Methods: [AWS CLI User Authentication](https://docs.aws.amazon.com/cli/v1/userguide/cli-authentication-user.html)


### Basic Usage Examples

#### List Available DLC Images
```bash
# List all available DLC images
q chat "List available DLC images"

# Filter by framework
q chat "Show me PyTorch training images"

# Filter by specific criteria
q chat "List PyTorch images with CUDA 12.1 and Python 3.10"
```

#### Build Custom Images
```bash
# Create a custom PyTorch image
q chat "Create a custom PyTorch image with scikit-learn and pandas"

# Build image with specific packages
q chat "Build a TensorFlow image with OpenCV and matplotlib"
```

#### Deploy to AWS Services
```bash
# Deploy to SageMaker
q chat "Deploy my custom image to SageMaker for inference"

# Deploy to ECS
q chat "Deploy to ECS cluster with 2 CPU and 4GB memory"
```

#### Upgrade Images
```bash
# Upgrade framework version
q chat "Upgrade my PyTorch image from 1.13 to 2.0"

# Analyze upgrade path
q chat "What's needed to upgrade my TensorFlow 2.10 image to 2.13?"
```

### 6. Configuration

The server can be configured using environment variables:

```bash
# Enable write operations (required for building and deployment)
export ALLOW_WRITE=true

# Enable access to sensitive data (for detailed logs and resource info)
export ALLOW_SENSITIVE_DATA=true

# Configure server port
export FASTMCP_PORT=8080

# Configure logging
export FASTMCP_LOG_LEVEL=INFO
export FASTMCP_LOG_FILE=/path/to/logfile.log
```

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

### Deployment and Troubleshooting
```
12. "Deploy my custom image to SageMaker for real-time inference"

13. "Help me troubleshoot CUDA out of memory errors in my training job"

14. "What are the security best practices for deploying DLC images in production?"

15. "How can I optimize costs when running DLC containers on AWS?"
```

### Performance and Best Practices
```
16. "Get performance optimization tips for PyTorch training on GPU"

17. "What are the framework-specific best practices for TensorFlow inference?"

18. "Show me deployment best practices for EKS"

19. "How do I create maintainable custom DLC images?"

20. "Check compatibility between PyTorch 1.13 and 2.0"
```

## Getting Started Instructions

### Step 1: Environment Setup
1. **Install Prerequisites**: Ensure you have AWS Q CLI, Docker, and proper AWS credentials configured
2. **Install the MCP Server**: Run `pip install aws_samples.dlc-mcp-server`
3. **Configure Environment Variables**: Set `ALLOW_WRITE=true` for building/deployment operations

### Step 2: Basic Operations
1. **Check Configuration**: Start with `q chat "Check my AWS configuration"`
2. **Explore Available Images**: Use `q chat "List available DLC images"`
3. **Authenticate with ECR**: Run `q chat "Setup ECR authentication"`

### Step 3: Choose Your Workflow
- **For Custom Image Building**: Follow Workflow 1 above
- **For Existing Image Upgrades**: Follow Workflow 2 above
- **For Troubleshooting**: Follow Workflow 3 above
- **For Distributed Training**: Follow Workflow 4 above

### Step 4: Deploy and Monitor
1. **Deploy to Your Platform**: Choose from EC2, SageMaker, ECS, or EKS
2. **Monitor Status**: Check deployment status and endpoint health
3. **Optimize Performance**: Apply best practices and performance tips

## Workflows and Usage Patterns

The AWS DLC MCP Server supports several common ML workflows:

### Workflow 1: Building and Deploying Custom DLC Images

1. **Discover Base Images**
   ```bash
   q chat "List available PyTorch base images for training"
   ```

2. **Create Custom Dockerfile**
   ```bash
   q chat "Create a custom PyTorch image with transformers, datasets, and wandb"
   ```

3. **Build Custom Image**
   ```bash
   q chat "Build the custom image and push to ECR repository 'my-pytorch-training'"
   ```

4. **Deploy to AWS Service**
   ```bash
   q chat "Deploy my custom image to SageMaker for training with ml.p3.2xlarge instance"
   ```

### Workflow 2: Upgrading Existing DLC Images

1. **Analyze Current Image**
   ```bash
   q chat "Analyze upgrade path from my current PyTorch 1.13 image to PyTorch 2.0"
   ```

2. **Generate Upgrade Dockerfile**
   ```bash
   q chat "Generate upgrade Dockerfile preserving my custom packages"
   ```

3. **Perform Upgrade**
   ```bash
   q chat "Upgrade my DLC image to PyTorch 2.0 while keeping custom configurations"
   ```

### Workflow 3: Troubleshooting and Optimization

1. **Diagnose Issues**
   ```bash
   q chat "Help me troubleshoot 'CUDA out of memory' error in my training job"
   ```

2. **Get Performance Tips**
   ```bash
   q chat "Get performance optimization tips for PyTorch training on GPU"
   ```

3. **Apply Best Practices**
   ```bash
   q chat "What are the security best practices for my DLC deployment?"
   ```

### Workflow 4: Distributed Training Setup

1. **Configure Environment**
   ```bash
   q chat "Setup distributed training for 4 nodes with 8 GPUs each using PyTorch"
   ```

2. **Run Training Container**
   ```bash
   q chat "Run my custom training container with GPU support"
   ```

## Core Services

The AWS DLC MCP Server provides six comprehensive service modules:

### 1. Container Management Service (`containers.py`)
**Purpose**: Core container operations and DLC image management

**Key Features**:
- **Image Discovery**: List and filter available DLC images by framework, Python version, CUDA version, and repository type
- **Container Runtime**: Run DLC containers locally with GPU support
- **Distributed Training Setup**: Configure multi-node distributed training environments
- **AWS Integration**: Automatic ECR authentication and AWS configuration validation
- **Environment Setup**: Check GPU availability and Docker configuration

**Available Tools**:
- `check_aws_config` - Validate AWS CLI configuration
- `setup_ecr_prod` - Authenticate with ECR production account
- `list_dlc_repos` - List available DLC repositories
- `list_dlc_images` - List and filter DLC images with advanced filtering
- `run_dlc_container` - Run containers locally with GPU support
- `setup_distributed_training` - Configure distributed training setups

### 2. Image Building Service (`image_building.py`)
**Purpose**: Create and customize DLC images for specific ML workloads

**Key Features**:
- **Base Image Selection**: Browse available DLC base images by framework and use case
- **Custom Dockerfile Generation**: Create optimized Dockerfiles with custom packages and configurations
- **Image Building**: Build custom DLC images locally or push to ECR
- **Package Management**: Install system packages, Python packages, and custom dependencies
- **Environment Configuration**: Set environment variables and custom commands

**Available Tools**:
- `list_base_images` - Browse available DLC base images
- `create_custom_dockerfile` - Generate custom Dockerfiles
- `build_custom_dlc_image` - Build and optionally push custom images to ECR

### 3. Deployment Service (`deployment.py`)
**Purpose**: Deploy DLC images across AWS compute platforms

**Key Features**:
- **Multi-Platform Deployment**: Support for EC2, SageMaker, ECS, and EKS
- **SageMaker Integration**: Create models and endpoints for inference
- **Container Orchestration**: Deploy to ECS clusters and EKS clusters
- **EC2 Deployment**: Launch EC2 instances with DLC images
- **Status Monitoring**: Check deployment status and endpoint health

**Available Tools**:
- `deploy_to_sagemaker` - Deploy to SageMaker for training/inference
- `deploy_to_ecs` - Deploy to Amazon ECS clusters
- `deploy_to_ec2` - Launch EC2 instances with DLC images
- `deploy_to_eks` - Deploy to Amazon EKS clusters
- `get_sagemaker_endpoint_status` - Monitor SageMaker endpoint status

### 4. Upgrade Service (`upgrade.py`)
**Purpose**: Upgrade and migrate DLC images to newer framework versions

**Key Features**:
- **Upgrade Path Analysis**: Analyze compatibility between current and target framework versions
- **Migration Planning**: Generate upgrade strategies with compatibility warnings
- **Dockerfile Generation**: Create upgrade Dockerfiles that preserve customizations
- **Version Migration**: Upgrade PyTorch, TensorFlow, and other frameworks
- **Custom File Preservation**: Maintain custom files and configurations during upgrades

**Available Tools**:
- `analyze_upgrade_path` - Analyze upgrade compatibility and requirements
- `generate_upgrade_dockerfile` - Create Dockerfiles for version upgrades
- `upgrade_dlc_image` - Perform complete image upgrades with customization preservation

### 5. Troubleshooting Service (`troubleshooting.py`)
**Purpose**: Diagnose and resolve DLC-related issues

**Key Features**:
- **Error Diagnosis**: Analyze error messages and provide specific solutions
- **Framework Compatibility**: Check version compatibility and requirements
- **Performance Optimization**: Get framework-specific performance tuning tips
- **Common Issues**: Database of solutions for frequent DLC problems
- **Environment Validation**: Verify system requirements and configurations

**Available Tools**:
- `diagnose_common_issues` - Analyze errors and provide solutions
- `get_framework_compatibility_info` - Check framework version compatibility
- `get_performance_optimization_tips` - Get performance tuning recommendations

### 6. Best Practices Service (`best_practices.py`)
**Purpose**: Provide expert guidance for optimal DLC usage

**Key Features**:
- **Security Guidelines**: Comprehensive security best practices for DLC deployments
- **Cost Optimization**: Strategies to reduce costs while maintaining performance
- **Deployment Patterns**: Platform-specific deployment recommendations
- **Framework Guidance**: Framework-specific best practices and optimizations
- **Custom Image Guidelines**: Best practices for creating maintainable custom images

**Available Tools**:
- `get_security_best_practices` - Security recommendations and guidelines
- `get_cost_optimization_tips` - Cost reduction strategies
- `get_deployment_best_practices` - Platform-specific deployment guidance
- `get_framework_specific_best_practices` - Framework optimization recommendations
- `get_custom_image_guidelines` - Custom image creation best practices

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

## Complete Tool Reference

### Container Management Tools
- `check_aws_config` - Validate AWS CLI configuration
- `setup_ecr_prod` - Authenticate with ECR production account (763104351884)
- `list_dlc_repos` - List available DLC repositories with filtering
- `list_dlc_images` - List and filter DLC images by framework, Python version, CUDA version
- `run_dlc_container` - Run containers locally with GPU support
- `setup_distributed_training` - Configure multi-node distributed training

### Image Building Tools
- `list_base_images` - Browse available DLC base images by framework and use case
- `create_custom_dockerfile` - Generate custom Dockerfiles with packages and configurations
- `build_custom_dlc_image` - Build and optionally push custom images to ECR

### Deployment Tools
- `deploy_to_sagemaker` - Deploy to SageMaker for training/inference
- `deploy_to_ecs` - Deploy to Amazon ECS clusters
- `deploy_to_ec2` - Launch EC2 instances with DLC images
- `deploy_to_eks` - Deploy to Amazon EKS clusters
- `get_sagemaker_endpoint_status` - Monitor SageMaker endpoint status

### Upgrade Tools
- `analyze_upgrade_path` - Analyze upgrade compatibility and requirements
- `generate_upgrade_dockerfile` - Create Dockerfiles for version upgrades
- `upgrade_dlc_image` - Perform complete image upgrades with customization preservation

### Troubleshooting Tools
- `diagnose_common_issues` - Analyze errors and provide solutions
- `get_framework_compatibility_info` - Check framework version compatibility
- `get_performance_optimization_tips` - Get performance tuning recommendations

### Best Practices Tools
- `get_security_best_practices` - Security recommendations and guidelines
- `get_cost_optimization_tips` - Cost reduction strategies
- `get_deployment_best_practices` - Platform-specific deployment guidance
- `get_framework_specific_best_practices` - Framework optimization recommendations
- `get_custom_image_guidelines` - Custom image creation best practices

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for development instructions.

## Disclaimer 

Intent of this project is to share a flavor of DLC MCP Server to demonstrate the use of Amazon Q and MCP server to
optimize the DLC maintenance. This project is not suited for direct production usage.

## License

This library is licensed under the [MIT-0 License](https://github.com/aws/mit-0).


