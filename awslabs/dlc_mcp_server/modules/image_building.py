"""Module for building custom DLC images."""

import logging
import os
import tempfile
from typing import Dict, Any, List, Optional

from mcp.server.fastmcp import FastMCP

from awslabs.dlc_mcp_server.utils.aws_utils import create_ecr_repository, get_ecr_login_command
from awslabs.dlc_mcp_server.utils.docker_utils import build_image, pull_image, push_image
from awslabs.dlc_mcp_server.utils.config import get_aws_region

logger = logging.getLogger(__name__)


def list_base_images(
    framework: Optional[str] = None,
    use_case: Optional[str] = None,
    device_type: Optional[str] = None,
    platform: Optional[str] = None
) -> Dict[str, Any]:
    """
    List available base images for Deep Learning Containers.
    
    Args:
        framework (Optional[str]): Framework filter (pytorch, tensorflow, etc.)
        use_case (Optional[str]): Use case filter (training, inference)
        device_type (Optional[str]): Device type filter (cpu, gpu)
        platform (Optional[str]): Platform filter (ec2, sagemaker)
        
    Returns:
        Dict[str, Any]: List of available base images
    """
    # This is a simplified implementation
    # In a real implementation, you would query the AWS DLC registry
    
    base_images = [
        {
            "framework": "pytorch",
            "version": "2.6.0",
            "use_case": "training",
            "device_type": "gpu",
            "python_version": "3.12",
            "cuda_version": "12.6",
            "os": "ubuntu22.04",
            "platform": "ec2",
            "uri": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-gpu-py312-cu126-ubuntu22.04-ec2"
        },
        {
            "framework": "pytorch",
            "version": "2.6.0",
            "use_case": "training",
            "device_type": "gpu",
            "python_version": "3.12",
            "cuda_version": "12.6",
            "os": "ubuntu22.04",
            "platform": "sagemaker",
            "uri": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-gpu-py312-cu126-ubuntu22.04-sagemaker"
        },
        {
            "framework": "pytorch",
            "version": "2.6.0",
            "use_case": "inference",
            "device_type": "cpu",
            "python_version": "3.12",
            "os": "ubuntu22.04",
            "platform": "ec2",
            "uri": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-cpu-py312-ubuntu22.04-ec2"
        },
        {
            "framework": "pytorch",
            "version": "2.6.0",
            "use_case": "inference",
            "device_type": "gpu",
            "python_version": "3.12",
            "cuda_version": "12.4",
            "os": "ubuntu22.04",
            "platform": "sagemaker",
            "uri": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker"
        },
        {
            "framework": "tensorflow",
            "version": "2.18.0",
            "use_case": "training",
            "device_type": "gpu",
            "python_version": "3.10",
            "cuda_version": "12.5",
            "os": "ubuntu22.04",
            "platform": "ec2",
            "uri": "763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.18.0-gpu-py310-cu125-ubuntu22.04-ec2"
        },
        {
            "framework": "tensorflow",
            "version": "2.18.0",
            "use_case": "inference",
            "device_type": "cpu",
            "python_version": "3.10",
            "os": "ubuntu20.04",
            "platform": "sagemaker",
            "uri": "763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.18.0-cpu-py310-ubuntu20.04-sagemaker"
        }
    ]
    
    # Apply filters
    filtered_images = base_images
    
    if framework:
        filtered_images = [img for img in filtered_images if img["framework"].lower() == framework.lower()]
    
    if use_case:
        filtered_images = [img for img in filtered_images if img["use_case"].lower() == use_case.lower()]
    
    if device_type:
        filtered_images = [img for img in filtered_images if img["device_type"].lower() == device_type.lower()]
    
    if platform:
        filtered_images = [img for img in filtered_images if img["platform"].lower() == platform.lower()]
    
    return {
        "images": filtered_images
    }


def create_custom_dockerfile(
    base_image: str,
    packages: List[str] = None,
    python_packages: List[str] = None,
    custom_commands: List[str] = None,
    environment_variables: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Create a custom Dockerfile based on a DLC base image.
    
    Args:
        base_image (str): Base image URI
        packages (List[str], optional): System packages to install
        python_packages (List[str], optional): Python packages to install
        custom_commands (List[str], optional): Custom commands to add to Dockerfile
        environment_variables (Dict[str, str], optional): Environment variables to set
        
    Returns:
        Dict[str, Any]: Dockerfile content
    """
    packages = packages or []
    python_packages = python_packages or []
    custom_commands = custom_commands or []
    environment_variables = environment_variables or {}
    
    # Create Dockerfile content
    dockerfile_lines = [
        f"FROM {base_image}",
        "",
        "# Set environment variables"
    ]
    
    # Add environment variables
    for key, value in environment_variables.items():
        dockerfile_lines.append(f"ENV {key}={value}")
    
    # Add system packages if any
    if packages:
        dockerfile_lines.extend([
            "",
            "# Install system packages",
            "RUN apt-get update && apt-get install -y --no-install-recommends \\",
            "    " + " \\\n    ".join(packages) + " \\",
            "    && apt-get clean \\",
            "    && rm -rf /var/lib/apt/lists/*"
        ])
    
    # Add Python packages if any
    if python_packages:
        dockerfile_lines.extend([
            "",
            "# Install Python packages",
            "RUN pip install --no-cache-dir \\",
            "    " + " \\\n    ".join(python_packages)
        ])
    
    # Add custom commands if any
    if custom_commands:
        dockerfile_lines.extend([
            "",
            "# Custom commands"
        ])
        dockerfile_lines.extend(custom_commands)
    
    # Add default workdir and cmd if not in custom commands
    if not any("WORKDIR" in cmd for cmd in custom_commands):
        dockerfile_lines.extend([
            "",
            "WORKDIR /opt/ml"
        ])
    
    dockerfile_content = "\n".join(dockerfile_lines)
    
    return {
        "dockerfile_content": dockerfile_content
    }


def build_custom_dlc_image(
    base_image: str,
    repository_name: str,
    tag: str,
    dockerfile_content: str,
    push_to_ecr: bool = False,
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build a custom DLC image.
    
    Args:
        base_image (str): Base image URI
        repository_name (str): ECR repository name
        tag (str): Image tag
        dockerfile_content (str): Dockerfile content
        push_to_ecr (bool): Whether to push the image to ECR
        region (Optional[str]): AWS region
        
    Returns:
        Dict[str, Any]: Build result
    """
    try:
        region = region or get_aws_region()
        
        # Create a temporary directory for the Dockerfile
        with tempfile.TemporaryDirectory() as temp_dir:
            dockerfile_path = os.path.join(temp_dir, "Dockerfile")
            
            # Write the Dockerfile
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile_content)
            
            # Pull the base image
            pull_result = pull_image(base_image)
            if not pull_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to pull base image: {pull_result['error']}"
                }
            
            # Build the image
            image_tag = f"{repository_name}:{tag}"
            build_result = build_image(
                dockerfile_path=dockerfile_path,
                tag=image_tag,
                context_path=temp_dir
            )
            
            if not build_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to build image: {build_result['error']}"
                }
            
            # Push to ECR if requested
            if push_to_ecr:
                # Create ECR repository
                repo_result = create_ecr_repository(repository_name, region)
                if not repo_result["success"]:
                    return {
                        "success": False,
                        "error": f"Failed to create ECR repository: {repo_result['error']}"
                    }
                
                # Get ECR login command
                login_result = get_ecr_login_command(region)
                if not login_result["success"]:
                    return {
                        "success": False,
                        "error": f"Failed to get ECR login: {login_result['error']}"
                    }
                
                # Tag the image for ECR
                ecr_uri = f"{repo_result['repository_uri']}:{tag}"
                
                # Push the image
                push_result = push_image(ecr_uri)
                if not push_result["success"]:
                    return {
                        "success": False,
                        "error": f"Failed to push image: {push_result['error']}"
                    }
                
                return {
                    "success": True,
                    "image_id": build_result["image_id"],
                    "local_tag": image_tag,
                    "ecr_uri": ecr_uri,
                    "repository_uri": repo_result["repository_uri"],
                    "build_logs": build_result["logs"],
                    "push_logs": push_result["logs"]
                }
            
            return {
                "success": True,
                "image_id": build_result["image_id"],
                "local_tag": image_tag,
                "build_logs": build_result["logs"]
            }
    
    except Exception as e:
        logger.error(f"Failed to build custom DLC image: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def register_module(mcp: FastMCP) -> None:
    """
    Register the image building module with the MCP server.
    
    Args:
        mcp (FastMCP): MCP server instance
    """
    mcp.add_tool(
        name="list_base_images",
        description="List available base images for Deep Learning Containers by dynamically fetching from AWS DLC repository.\n\nArgs:\n    framework: Optional framework filter (pytorch, tensorflow, etc.)\n    use_case: Optional use case filter (training, inference)\n    device_type: Optional device type filter (cpu, gpu)\n    platform: Optional platform filter (ec2, sagemaker)\n",
        function=list_base_images
    )
    
    mcp.add_tool(
        name="create_custom_dockerfile",
        description="Create a custom Dockerfile based on a DLC base image.\n\nArgs:\n    base_image: Base image URI\n    packages: Optional list of system packages to install\n    python_packages: Optional list of Python packages to install\n    custom_commands: Optional list of custom commands to add to Dockerfile\n    environment_variables: Optional dictionary of environment variables to set\n",
        function=create_custom_dockerfile
    )
    
    mcp.add_tool(
        name="build_custom_dlc_image",
        description="Build a custom DLC image.\n\nArgs:\n    base_image: Base image URI\n    repository_name: ECR repository name\n    tag: Image tag\n    dockerfile_content: Dockerfile content\n    push_to_ecr: Whether to push the image to ECR\n    region: Optional AWS region\n",
        function=build_custom_dlc_image
    )
