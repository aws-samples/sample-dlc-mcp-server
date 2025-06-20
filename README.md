# AWS Deep Learning Containers MCP Server

A Model Context Protocol (MCP) server for AWS Deep Learning Containers (DLC) that assists with:

- Building custom DLC images
- Deploying DLC images to various AWS platforms (EC2, SageMaker, ECS, EKS)
- Upgrading existing images to newer framework versions
- Troubleshooting DLC-related issues
- Providing best practices guidance for DLC usage

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

## Features

### DLC Image Building

- Create custom DLC images based on AWS base images
- Add custom dependencies and packages
- Optimize images for specific use cases

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

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for development instructions.

## License

Apache 2.0
