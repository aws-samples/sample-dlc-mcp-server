# AWS Deep Learning Containers MCP Server

A Model Context Protocol (MCP) server for AWS Deep Learning Containers (DLC) that provides tools for discovering, building, deploying, and troubleshooting DLC images.

## Features

- **Dynamic DLC Image Discovery**: Automatically fetches latest images from [AWS DLC GitHub](https://aws.github.io/deep-learning-containers/reference/available_images/) - always up-to-date
- **Image Building**: Create custom Dockerfiles and build images based on DLC base images
- **Multi-Platform Deployment**: Deploy to SageMaker, EC2, ECS, and EKS
- **Instance Recommendations**: Get GPU instance recommendations based on model size and budget
- **Upgrade Support**: Analyze upgrade paths and generate migration Dockerfiles
- **Troubleshooting**: Diagnose common DLC issues with actionable solutions
- **Best Practices**: Security, cost optimization, and deployment guidance
- **No AWS Credentials Required**: Discovery tools work without AWS credentials

## Quick Start

### Option 1: Run with uv (Recommended)

```bash
# Clone the repo
git clone https://github.com/aws-samples/sample-dlc-mcp-server.git
cd sample-dlc-mcp-server

# Run directly with uv
uv run dlc-mcp-server
```

### Option 2: Run with Docker

```bash
# Build the image
docker build -t dlc-mcp-server .

# Run the container
docker run -it --rm \
  -v ~/.aws:/root/.aws:ro \
  dlc-mcp-server
```

### Option 3: Install locally

```bash
pip install -e .
dlc-mcp-server
```

## MCP Client Configuration

### For Amazon Q CLI

Add to `~/.aws/amazonq/mcp.json`:

```json
{
  "mcpServers": {
    "dlc-mcp-server": {
      "command": "uv",
      "args": ["--directory", "/path/to/sample-dlc-mcp-server", "run", "dlc-mcp-server"],
      "timeout": 120000
    }
  }
}
```

### For Kiro

Add to `.kiro/settings/mcp.json`:

```json
{
  "mcpServers": {
    "dlc-mcp-server": {
      "command": "uv",
      "args": ["--directory", "/path/to/sample-dlc-mcp-server", "run", "dlc-mcp-server"],
      "timeout": 120000
    }
  }
}
```

### Using Docker

```json
{
  "mcpServers": {
    "dlc-mcp-server": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "-v", "~/.aws:/root/.aws:ro", "dlc-mcp-server"],
      "timeout": 120000
    }
  }
}
```

## Available Tools

### DLC Discovery
| Tool | Description |
|------|-------------|
| `search_dlc_images` | Search DLC images by framework, version, accelerator, platform |
| `get_dlc_recommendation` | Get image recommendations based on model type and size |
| `list_dlc_frameworks` | List all available frameworks with versions |
| `get_llm_serving_options` | Compare vLLM, SGLang, DJL, NeuronX options |
| `compare_dlc_images` | Side-by-side image comparison |
| `refresh_dlc_catalog` | Force refresh image catalog from GitHub |

### Image Building
| Tool | Description |
|------|-------------|
| `create_custom_dockerfile` | Generate Dockerfile with custom packages |
| `build_custom_dlc_image` | Build and optionally push to ECR |

### Deployment
| Tool | Description |
|------|-------------|
| `deploy_to_sagemaker` | Deploy to SageMaker endpoint |
| `deploy_to_ec2` | Launch EC2 instance with DLC |
| `deploy_to_ecs` | Deploy to ECS cluster |
| `deploy_to_eks` | Deploy to EKS cluster |
| `get_sagemaker_endpoint_status` | Check endpoint status |

### Instance Advisor
| Tool | Description |
|------|-------------|
| `get_instance_recommendation` | GPU instance recommendations by model size |
| `list_gpu_instances` | List available GPU instances with pricing |
| `estimate_training_cost` | Estimate training job costs |

### Troubleshooting
| Tool | Description |
|------|-------------|
| `analyze_dlc_error` | Analyze error logs with root cause analysis |
| `diagnose_common_issues` | Diagnose common DLC problems |
| `get_framework_compatibility_info` | Check framework version compatibility |

### Best Practices
| Tool | Description |
|------|-------------|
| `get_security_best_practices` | Security guidelines |
| `get_cost_optimization_tips` | Cost reduction strategies |
| `get_deployment_best_practices` | Platform-specific guidance |
| `get_framework_specific_best_practices` | Framework optimization tips |

## Supported Frameworks

| Framework | Latest Version | Use Cases |
|-----------|---------------|-----------|
| PyTorch | 2.9.0 | Training, Inference |
| TensorFlow | 2.19.0 | Training, Inference |
| vLLM | 0.15.1 | LLM Inference |
| SGLang | 0.5.8 | LLM Inference |
| HuggingFace PyTorch | 2.6.0 | NLP Training/Inference |
| AutoGluon | 1.5.0 | AutoML |
| DJL | 0.36.0 | Large Model Inference |
| PyTorch NeuronX | 2.9.0 | Trainium/Inferentia |

## Example Usage

### Find vLLM images
```
Search for vLLM images for SageMaker inference
```

### Deploy LLM to SageMaker
```
Deploy Qwen2.5-32B using vLLM on SageMaker with the right instance type
```

### Get instance recommendations
```
What instance should I use for a 35GB model?
```

### Troubleshoot errors
```
Help me fix this CUDA out of memory error: [paste error]
```

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `ALLOW_WRITE` | Enable build/deploy operations | `false` |
| `ALLOW_SENSITIVE_DATA` | Enable detailed logs access | `false` |
| `FASTMCP_LOG_LEVEL` | Logging level | `ERROR` |
| `FASTMCP_LOG_FILE` | Log file path | None |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Run linting
ruff check .
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for more details.

## License

This library is licensed under the [MIT-0 License](https://github.com/aws/mit-0).
