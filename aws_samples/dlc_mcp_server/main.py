#!/usr/bin/env python3
"""
AWS Deep Learning Containers MCP Server - Main entry point
"""

import logging
import os
import sys

from mcp.server.fastmcp import FastMCP

from aws_samples.dlc_mcp_server.modules import (
    image_building,
    deployment,
    troubleshooting,
    upgrade,
    best_practices,
    containers,
)
from aws_samples.dlc_mcp_server.utils.config import get_config
from aws_samples.dlc_mcp_server.utils.security import (
    PERMISSION_WRITE,
    PERMISSION_SENSITIVE_DATA,
    secure_tool,
)

# Configure logging
log_level = os.environ.get("FASTMCP_LOG_LEVEL", "ERROR")
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
log_file = os.environ.get("FASTMCP_LOG_FILE")

# Set up basic configuration
logging.basicConfig(
    level=log_level,
    format=log_format,
)

# Add file handler if log file path is specified
if log_file:
    try:
        # Create directory for log file if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Add file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
        logging.info(f"Logging to file: {log_file}")
    except Exception as e:
        logging.error(f"Failed to set up log file {log_file}: {e}")

logger = logging.getLogger("dlc-mcp-server")

# Load configuration
config = get_config()

# Create the MCP server
mcp = FastMCP(
    name="AWS Deep Learning Containers MCP Server",
    description=("A server for building, customizing, and deploying AWS Deep Learning Containers"),
    version="0.1.0",
    instructions="""Use this server to build, customize, and deploy AWS Deep Learning Containers (DLC).

WORKFLOWS:

1. Building Custom DLC Images:
   - List available base DLC images
   - Create custom Dockerfiles based on DLC base images
   - Build and push custom DLC images to ECR

2. Deploying DLC Images:
   - Deploy to Amazon SageMaker for training or inference
   - Deploy to Amazon EC2 for development or production
   - Deploy to Amazon ECS for containerized deployments
   - Deploy to Amazon EKS for Kubernetes-based deployments

3. Upgrading DLC Images:
   - Analyze upgrade paths between framework versions
   - Generate upgrade Dockerfiles
   - Build upgraded images with preserved customizations

4. Troubleshooting DLC Issues:
   - Diagnose common DLC-related issues
   - Get framework compatibility information
   - Get performance optimization tips

5. Best Practices:
   - Get security best practices for DLC usage
   - Get cost optimization tips
   - Get deployment best practices
   - Get framework-specific best practices
   - Get guidelines for creating custom DLC images

IMPORTANT:
- AWS credentials must be properly configured with appropriate permissions
- Set ALLOW_WRITE=true to enable image building and deployment operations
- Set ALLOW_SENSITIVE_DATA=true to enable access to logs and detailed resource information
""",
)

# Apply security wrappers to API functions that require write permissions
image_building.build_custom_dlc_image = secure_tool(
    config, PERMISSION_WRITE, "build_custom_dlc_image"
)(image_building.build_custom_dlc_image)

deployment.deploy_to_sagemaker = secure_tool(config, PERMISSION_WRITE, "deploy_to_sagemaker")(
    deployment.deploy_to_sagemaker
)

deployment.deploy_to_ecs = secure_tool(config, PERMISSION_WRITE, "deploy_to_ecs")(
    deployment.deploy_to_ecs
)

deployment.deploy_to_ec2 = secure_tool(config, PERMISSION_WRITE, "deploy_to_ec2")(
    deployment.deploy_to_ec2
)

deployment.deploy_to_eks = secure_tool(config, PERMISSION_WRITE, "deploy_to_eks")(
    deployment.deploy_to_eks
)

upgrade.upgrade_dlc_image = secure_tool(config, PERMISSION_WRITE, "upgrade_dlc_image")(
    upgrade.upgrade_dlc_image
)

# Register all modules
image_building.register_module(mcp)
deployment.register_module(mcp)
troubleshooting.register_module(mcp)
upgrade.register_module(mcp)
best_practices.register_module(mcp)
containers.register_module(mcp)


def main() -> None:
    """Main entry point for the DLC MCP Server."""
    try:
        # Start the server
        logger.info("Server started")
        logger.info(f"Write operations enabled: {config.get('allow-write', False)}")
        logger.info(f"Sensitive data access enabled: {config.get('allow-sensitive-data', False)}")
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
