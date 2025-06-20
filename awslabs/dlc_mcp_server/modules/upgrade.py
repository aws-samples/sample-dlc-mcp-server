"""Module for upgrading DLC images to newer framework versions."""

import logging
import os
import tempfile
from typing import Dict, Any, List, Optional

from mcp.server.fastmcp import FastMCP

from awslabs.dlc_mcp_server.utils.docker_utils import build_image, pull_image, push_image
from awslabs.dlc_mcp_server.utils.aws_utils import create_ecr_repository
from awslabs.dlc_mcp_server.utils.config import get_aws_region

logger = logging.getLogger(__name__)


def analyze_upgrade_path(
    current_image: str,
    target_framework: str,
    target_version: str
) -> Dict[str, Any]:
    """
    Analyze the upgrade path from current image to target framework version.
    
    Args:
        current_image (str): Current image URI
        target_framework (str): Target framework (pytorch, tensorflow, etc.)
        target_version (str): Target framework version
        
    Returns:
        Dict[str, Any]: Upgrade path analysis
    """
    try:
        # Parse current image information
        # Example: 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.13.1-gpu-py39-cu117-ubuntu20.04-ec2
        image_parts = current_image.split(":")
        if len(image_parts) != 2:
            return {
                "success": False,
                "error": "Invalid image URI format"
            }
        
        repository = image_parts[0]
        tag = image_parts[1]
        
        # Parse tag components
        tag_parts = tag.split("-")
        if len(tag_parts) < 4:
            return {
                "success": False,
                "error": "Invalid image tag format"
            }
        
        current_version = tag_parts[0]
        device_type = tag_parts[1]  # gpu or cpu
        
        # Extract Python version
        python_version = None
        cuda_version = None
        os_version = None
        platform = None
        
        for part in tag_parts:
            if part.startswith("py"):
                python_version = part
            elif part.startswith("cu"):
                cuda_version = part
            elif part.startswith("ubuntu"):
                os_version = part
            elif part in ["ec2", "sagemaker"]:
                platform = part
        
        # Determine framework from repository
        current_framework = None
        use_case = None
        
        if "pytorch" in repository:
            current_framework = "pytorch"
        elif "tensorflow" in repository:
            current_framework = "tensorflow"
        elif "mxnet" in repository:
            current_framework = "mxnet"
        
        if "training" in repository:
            use_case = "training"
        elif "inference" in repository:
            use_case = "inference"
        
        # Check if framework change is requested
        framework_change = current_framework != target_framework.lower()
        
        # Determine compatibility and upgrade steps
        compatibility_issues = []
        upgrade_steps = []
        
        # Framework-specific upgrade considerations
        if current_framework == "pytorch" and target_framework.lower() == "pytorch":
            current_major, current_minor = map(int, current_version.split(".")[:2])
            target_major, target_minor = map(int, target_version.split(".")[:2])
            
            if target_major > current_major:
                compatibility_issues.append(f"Major version upgrade from {current_version} to {target_version} may have breaking API changes")
                upgrade_steps.append("Review PyTorch migration guides for API changes")
                upgrade_steps.append("Update model architecture and training code for compatibility")
                
                if current_major == 1 and target_major == 2:
                    upgrade_steps.append("Consider using torch.compile() for performance improvements in PyTorch 2.x")
                    upgrade_steps.append("Update deprecated nn.Module APIs")
            
            if cuda_version:
                upgrade_steps.append(f"Verify CUDA compatibility with target PyTorch version {target_version}")
        
        elif current_framework == "tensorflow" and target_framework.lower() == "tensorflow":
            current_major, current_minor = map(int, current_version.split(".")[:2])
            target_major, target_minor = map(int, target_version.split(".")[:2])
            
            if target_major > current_major:
                compatibility_issues.append(f"Major version upgrade from {current_version} to {target_version} may have breaking API changes")
                upgrade_steps.append("Review TensorFlow migration guides for API changes")
                
                if current_major == 1 and target_major == 2:
                    compatibility_issues.append("TensorFlow 1.x to 2.x has significant API changes")
                    upgrade_steps.append("Use the TensorFlow upgrade script: tf_upgrade_v2")
                    upgrade_steps.append("Update to Keras 2.x API")
            
            if target_minor > current_minor and target_major == 2:
                if current_minor < 4 and target_minor >= 4:
                    upgrade_steps.append("Update to the new tf.keras.Model.fit API")
                if current_minor < 8 and target_minor >= 8:
                    upgrade_steps.append("Review changes to tf.data API")
        
        # Framework change considerations
        if framework_change:
            compatibility_issues.append(f"Changing framework from {current_framework} to {target_framework} requires model conversion")
            
            if current_framework == "pytorch" and target_framework.lower() == "tensorflow":
                upgrade_steps.append("Convert PyTorch model to ONNX format")
                upgrade_steps.append("Import ONNX model into TensorFlow using tf2onnx")
            elif current_framework == "tensorflow" and target_framework.lower() == "pytorch":
                upgrade_steps.append("Convert TensorFlow model to ONNX format")
                upgrade_steps.append("Import ONNX model into PyTorch")
        
        # Construct target image URI
        region = current_image.split(".")[2]
        
        # Determine latest CUDA version if needed
        latest_cuda_version = cuda_version
        if device_type == "gpu":
            if target_framework.lower() == "pytorch" and target_version.startswith("2."):
                latest_cuda_version = "cu126" if target_version >= "2.6.0" else "cu124"
            elif target_framework.lower() == "tensorflow" and target_version.startswith("2."):
                latest_cuda_version = "cu125" if target_version >= "2.18.0" else "cu122"
        
        # Determine latest Python version
        latest_python_version = python_version
        if target_framework.lower() == "pytorch" and target_version >= "2.6.0":
            latest_python_version = "py312"
        elif target_framework.lower() == "tensorflow" and target_version >= "2.18.0":
            latest_python_version = "py310"
        
        # Determine latest OS version
        latest_os_version = os_version
        if target_framework.lower() == "pytorch" and target_version >= "2.0.0":
            latest_os_version = "ubuntu22.04"
        elif target_framework.lower() == "tensorflow" and target_version >= "2.12.0":
            latest_os_version = "ubuntu22.04"
        
        # Construct target tag components
        target_tag_components = [
            target_version,
            device_type,
            latest_python_version
        ]
        
        if device_type == "gpu" and latest_cuda_version:
            target_tag_components.append(latest_cuda_version)
        
        if latest_os_version:
            target_tag_components.append(latest_os_version)
        
        if platform:
            target_tag_components.append(platform)
        
        target_tag = "-".join(target_tag_components)
        
        # Construct target repository
        target_repo = repository.split("/")[0]
        if framework_change:
            target_repo += f"/{target_framework.lower()}-{use_case}"
        
        target_image = f"{target_repo}:{target_tag}"
        
        return {
            "success": True,
            "current_image": {
                "uri": current_image,
                "framework": current_framework,
                "version": current_version,
                "device_type": device_type,
                "python_version": python_version,
                "cuda_version": cuda_version,
                "os_version": os_version,
                "platform": platform,
                "use_case": use_case
            },
            "target_image": {
                "uri": target_image,
                "framework": target_framework.lower(),
                "version": target_version,
                "device_type": device_type,
                "python_version": latest_python_version,
                "cuda_version": latest_cuda_version,
                "os_version": latest_os_version,
                "platform": platform,
                "use_case": use_case
            },
            "framework_change": framework_change,
            "compatibility_issues": compatibility_issues,
            "upgrade_steps": upgrade_steps
        }
    except Exception as e:
        logger.error(f"Failed to analyze upgrade path: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def generate_upgrade_dockerfile(
    current_image: str,
    target_image: str,
    preserve_custom_files: List[str] = None,
    additional_packages: List[str] = None
) -> Dict[str, Any]:
    """
    Generate a Dockerfile for upgrading a DLC image.
    
    Args:
        current_image (str): Current image URI
        target_image (str): Target image URI
        preserve_custom_files (List[str], optional): Files to preserve from current image
        additional_packages (List[str], optional): Additional packages to install
        
    Returns:
        Dict[str, Any]: Dockerfile content
    """
    preserve_custom_files = preserve_custom_files or []
    additional_packages = additional_packages or []
    
    # Create Dockerfile content
    dockerfile_lines = [
        f"# Upgrade Dockerfile from {current_image} to {target_image}",
        f"FROM {target_image} as target",
        "",
        f"FROM {current_image} as source",
        ""
    ]
    
    # Add commands to copy files from source image
    if preserve_custom_files:
        dockerfile_lines.extend([
            "# Create temporary directory for files to preserve",
            "RUN mkdir -p /tmp/preserve",
            ""
        ])
        
        for file_path in preserve_custom_files:
            dockerfile_lines.append(f"# Copy {file_path} to preserve")
            dockerfile_lines.append(f"RUN if [ -e {file_path} ]; then cp -r {file_path} /tmp/preserve/; fi")
        
        dockerfile_lines.append("")
    
    # Start with the target image
    dockerfile_lines.extend([
        "# Start with the target image",
        f"FROM {target_image}",
        ""
    ])
    
    # Copy preserved files
    if preserve_custom_files:
        dockerfile_lines.extend([
            "# Copy preserved files from source image",
            "COPY --from=source /tmp/preserve /tmp/preserve",
            ""
        ])
        
        for file_path in preserve_custom_files:
            base_name = os.path.basename(file_path)
            dir_name = os.path.dirname(file_path)
            dockerfile_lines.append(f"# Restore {file_path}")
            dockerfile_lines.append(f"RUN if [ -e /tmp/preserve/{base_name} ]; then mkdir -p {dir_name} && cp -r /tmp/preserve/{base_name} {file_path}; fi")
        
        dockerfile_lines.extend([
            "",
            "# Clean up temporary directory",
            "RUN rm -rf /tmp/preserve",
            ""
        ])
    
    # Install additional packages
    if additional_packages:
        dockerfile_lines.extend([
            "# Install additional packages",
            "RUN pip install --no-cache-dir \\",
            "    " + " \\\n    ".join(additional_packages),
            ""
        ])
    
    # Add final comments
    dockerfile_lines.extend([
        "# Image upgraded successfully",
        "WORKDIR /opt/ml"
    ])
    
    dockerfile_content = "\n".join(dockerfile_lines)
    
    return {
        "dockerfile_content": dockerfile_content
    }


def upgrade_dlc_image(
    current_image: str,
    target_framework: str,
    target_version: str,
    repository_name: str,
    tag: str,
    preserve_custom_files: List[str] = None,
    additional_packages: List[str] = None,
    push_to_ecr: bool = False,
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Upgrade a DLC image to a newer framework version.
    
    Args:
        current_image (str): Current image URI
        target_framework (str): Target framework (pytorch, tensorflow, etc.)
        target_version (str): Target framework version
        repository_name (str): ECR repository name
        tag (str): Image tag
        preserve_custom_files (List[str], optional): Files to preserve from current image
        additional_packages (List[str], optional): Additional packages to install
        push_to_ecr (bool): Whether to push the image to ECR
        region (Optional[str]): AWS region
        
    Returns:
        Dict[str, Any]: Upgrade result
    """
    try:
        # Analyze upgrade path
        analysis = analyze_upgrade_path(current_image, target_framework, target_version)
        if not analysis["success"]:
            return analysis
        
        target_image = analysis["target_image"]["uri"]
        
        # Generate upgrade Dockerfile
        dockerfile_result = generate_upgrade_dockerfile(
            current_image=current_image,
            target_image=target_image,
            preserve_custom_files=preserve_custom_files,
            additional_packages=additional_packages
        )
        
        dockerfile_content = dockerfile_result["dockerfile_content"]
        
        # Create a temporary directory for the Dockerfile
        with tempfile.TemporaryDirectory() as temp_dir:
            dockerfile_path = os.path.join(temp_dir, "Dockerfile")
            
            # Write the Dockerfile
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile_content)
            
            # Pull the base images
            pull_result = pull_image(current_image)
            if not pull_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to pull current image: {pull_result['error']}"
                }
            
            pull_result = pull_image(target_image)
            if not pull_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to pull target image: {pull_result['error']}"
                }
            
            # Build the upgraded image
            image_tag = f"{repository_name}:{tag}"
            build_result = build_image(
                dockerfile_path=dockerfile_path,
                tag=image_tag,
                context_path=temp_dir
            )
            
            if not build_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to build upgraded image: {build_result['error']}"
                }
            
            # Push to ECR if requested
            if push_to_ecr:
                region = region or get_aws_region()
                
                # Create ECR repository
                repo_result = create_ecr_repository(repository_name, region)
                if not repo_result["success"]:
                    return {
                        "success": False,
                        "error": f"Failed to create ECR repository: {repo_result['error']}"
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
                    "push_logs": push_result["logs"],
                    "upgrade_analysis": analysis
                }
            
            return {
                "success": True,
                "image_id": build_result["image_id"],
                "local_tag": image_tag,
                "build_logs": build_result["logs"],
                "upgrade_analysis": analysis
            }
    
    except Exception as e:
        logger.error(f"Failed to upgrade DLC image: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def register_module(mcp: FastMCP) -> None:
    """
    Register the upgrade module with the MCP server.
    
    Args:
        mcp (FastMCP): MCP server instance
    """
    mcp.add_tool(
        name="analyze_upgrade_path",
        description="Analyze the upgrade path from current image to target framework version.\n\nArgs:\n    current_image: Current image URI\n    target_framework: Target framework (pytorch, tensorflow, etc.)\n    target_version: Target framework version\n",
        function=analyze_upgrade_path
    )
    
    mcp.add_tool(
        name="generate_upgrade_dockerfile",
        description="Generate a Dockerfile for upgrading a DLC image.\n\nArgs:\n    current_image: Current image URI\n    target_image: Target image URI\n    preserve_custom_files: Optional list of files to preserve from current image\n    additional_packages: Optional list of additional packages to install\n",
        function=generate_upgrade_dockerfile
    )
    
    mcp.add_tool(
        name="upgrade_dlc_image",
        description="Upgrade a DLC image to a newer framework version.\n\nArgs:\n    current_image: Current image URI\n    target_framework: Target framework (pytorch, tensorflow, etc.)\n    target_version: Target framework version\n    repository_name: ECR repository name\n    tag: Image tag\n    preserve_custom_files: Optional list of files to preserve from current image\n    additional_packages: Optional list of additional packages to install\n    push_to_ecr: Whether to push the image to ECR\n    region: Optional AWS region\n",
        function=upgrade_dlc_image
    )
