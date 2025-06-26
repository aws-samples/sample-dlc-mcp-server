"""Module for building custom DLC images."""

import logging
import os
import tempfile
from typing import Dict, Any, List, Optional, NamedTuple
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP

from aws_samples.dlc_mcp_server.utils.aws_utils import create_ecr_repository, get_ecr_login_command
from aws_samples.dlc_mcp_server.utils.docker_utils import build_image, pull_image, push_image
from aws_samples.dlc_mcp_server.utils.config import get_aws_region

# Constants
DEFAULT_WORKDIR = "/opt/ml"
DOCKERFILE_NAME = "Dockerfile"

# AWS DLC Account ID
AWS_DLC_ACCOUNT_ID = "763104351884"

logger = logging.getLogger(__name__)


@dataclass
class BaseImageInfo:
    """Data class for base image information."""

    framework: str
    version: str
    use_case: str
    device_type: str
    python_version: str
    os: str
    platform: str
    uri: str
    cuda_version: Optional[str] = None


@dataclass
class DockerfileConfig:
    """Configuration for Dockerfile creation."""

    base_image: str
    packages: List[str]
    python_packages: List[str]
    custom_commands: List[str]
    environment_variables: Dict[str, str]


@dataclass
class BuildConfig:
    """Configuration for image building."""

    base_image: str
    repository_name: str
    tag: str
    dockerfile_content: str
    push_to_ecr: bool
    region: Optional[str]


class BaseImageRegistry:
    """Registry for DLC base images."""

    @staticmethod
    def get_base_images() -> List[BaseImageInfo]:
        """Get list of available base images."""
        return [
            # PyTorch Images
            BaseImageInfo(
                framework="pytorch",
                version="2.6.0",
                use_case="training",
                device_type="gpu",
                python_version="3.12",
                cuda_version="12.6",
                os="ubuntu22.04",
                platform="ec2",
                uri=f"{AWS_DLC_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-gpu-py312-cu126-ubuntu22.04-ec2",
            ),
            BaseImageInfo(
                framework="pytorch",
                version="2.6.0",
                use_case="training",
                device_type="gpu",
                python_version="3.12",
                cuda_version="12.6",
                os="ubuntu22.04",
                platform="sagemaker",
                uri=f"{AWS_DLC_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-gpu-py312-cu126-ubuntu22.04-sagemaker",
            ),
            BaseImageInfo(
                framework="pytorch",
                version="2.6.0",
                use_case="inference",
                device_type="cpu",
                python_version="3.12",
                os="ubuntu22.04",
                platform="ec2",
                uri=f"{AWS_DLC_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-cpu-py312-ubuntu22.04-ec2",
            ),
            BaseImageInfo(
                framework="pytorch",
                version="2.6.0",
                use_case="inference",
                device_type="gpu",
                python_version="3.12",
                cuda_version="12.4",
                os="ubuntu22.04",
                platform="sagemaker",
                uri=f"{AWS_DLC_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker",
            ),
            # TensorFlow Images
            BaseImageInfo(
                framework="tensorflow",
                version="2.18.0",
                use_case="training",
                device_type="gpu",
                python_version="3.10",
                cuda_version="12.5",
                os="ubuntu22.04",
                platform="ec2",
                uri=f"{AWS_DLC_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.18.0-gpu-py310-cu125-ubuntu22.04-ec2",
            ),
            BaseImageInfo(
                framework="tensorflow",
                version="2.18.0",
                use_case="inference",
                device_type="cpu",
                python_version="3.10",
                os="ubuntu20.04",
                platform="sagemaker",
                uri=f"{AWS_DLC_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.18.0-cpu-py310-ubuntu20.04-sagemaker",
            ),
        ]


class ImageFilter:
    """Utility class for filtering base images."""

    @staticmethod
    def apply_filters(
        images: List[BaseImageInfo],
        framework: Optional[str] = None,
        use_case: Optional[str] = None,
        device_type: Optional[str] = None,
        platform: Optional[str] = None,
    ) -> List[BaseImageInfo]:
        """Apply filters to image list."""
        filtered_images = images

        if framework:
            filtered_images = [
                img for img in filtered_images if img.framework.lower() == framework.lower()
            ]

        if use_case:
            filtered_images = [
                img for img in filtered_images if img.use_case.lower() == use_case.lower()
            ]

        if device_type:
            filtered_images = [
                img for img in filtered_images if img.device_type.lower() == device_type.lower()
            ]

        if platform:
            filtered_images = [
                img for img in filtered_images if img.platform.lower() == platform.lower()
            ]

        return filtered_images


class DockerfileBuilder:
    """Builder class for creating Dockerfiles."""

    def __init__(self, config: DockerfileConfig):
        self.config = config
        self.dockerfile_lines: List[str] = []

    def build(self) -> str:
        """Build the complete Dockerfile content."""
        self._add_base_image()
        self._add_environment_variables()
        self._add_system_packages()
        self._add_python_packages()
        self._add_custom_commands()
        self._add_default_workdir()

        return "\n".join(self.dockerfile_lines)

    def _add_base_image(self) -> None:
        """Add FROM instruction."""
        self.dockerfile_lines.extend([f"FROM {self.config.base_image}", ""])

    def _add_environment_variables(self) -> None:
        """Add environment variables."""
        if not self.config.environment_variables:
            return

        self.dockerfile_lines.append("# Set environment variables")
        for key, value in self.config.environment_variables.items():
            self.dockerfile_lines.append(f"ENV {key}={value}")
        self.dockerfile_lines.append("")

    def _add_system_packages(self) -> None:
        """Add system package installation."""
        if not self.config.packages:
            return

        self.dockerfile_lines.extend(
            [
                "# Install system packages",
                "RUN apt-get update && apt-get install -y --no-install-recommends \\",
                "    " + " \\\n    ".join(self.config.packages) + " \\",
                "    && apt-get clean \\",
                "    && rm -rf /var/lib/apt/lists/*",
                "",
            ]
        )

    def _add_python_packages(self) -> None:
        """Add Python package installation."""
        if not self.config.python_packages:
            return

        self.dockerfile_lines.extend(
            [
                "# Install Python packages",
                "RUN pip install --no-cache-dir \\",
                "    " + " \\\n    ".join(self.config.python_packages),
                "",
            ]
        )

    def _add_custom_commands(self) -> None:
        """Add custom commands."""
        if not self.config.custom_commands:
            return

        self.dockerfile_lines.extend(["# Custom commands", *self.config.custom_commands, ""])

    def _add_default_workdir(self) -> None:
        """Add default workdir if not specified in custom commands."""
        has_workdir = any("WORKDIR" in cmd for cmd in self.config.custom_commands)
        if not has_workdir:
            self.dockerfile_lines.append(f"WORKDIR {DEFAULT_WORKDIR}")


class CustomImageBuilder:
    """Builder class for custom DLC images."""

    def __init__(self, config: BuildConfig):
        self.config = config
        self.region = config.region or get_aws_region()

    def build(self) -> Dict[str, Any]:
        """Build the custom image."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                dockerfile_path = self._create_dockerfile(temp_dir)

                # Pull base image
                pull_result = self._pull_base_image()
                if not pull_result["success"]:
                    return self._error_response(
                        f"Failed to pull base image: {pull_result['error']}"
                    )

                # Build image
                build_result = self._build_image(dockerfile_path, temp_dir)
                if not build_result["success"]:
                    return self._error_response(f"Failed to build image: {build_result['error']}")

                # Push to ECR if requested
                if self.config.push_to_ecr:
                    return self._build_and_push_to_ecr(build_result)

                return self._build_success_response(build_result)

        except Exception as e:
            logger.error(f"Failed to build custom DLC image: {e}")
            return self._error_response(str(e))

    def _create_dockerfile(self, temp_dir: str) -> str:
        """Create Dockerfile in temporary directory."""
        dockerfile_path = os.path.join(temp_dir, DOCKERFILE_NAME)
        with open(dockerfile_path, "w") as f:
            f.write(self.config.dockerfile_content)
        return dockerfile_path

    def _pull_base_image(self) -> Dict[str, Any]:
        """Pull the base image."""
        return pull_image(self.config.base_image)

    def _build_image(self, dockerfile_path: str, context_path: str) -> Dict[str, Any]:
        """Build the Docker image."""
        image_tag = f"{self.config.repository_name}:{self.config.tag}"
        return build_image(
            dockerfile_path=dockerfile_path, tag=image_tag, context_path=context_path
        )

    def _build_and_push_to_ecr(self, build_result: Dict[str, Any]) -> Dict[str, Any]:
        """Build and push image to ECR."""
        # Create ECR repository
        repo_result = create_ecr_repository(self.config.repository_name, self.region)
        if not repo_result["success"]:
            return self._error_response(f"Failed to create ECR repository: {repo_result['error']}")

        # Get ECR login
        login_result = get_ecr_login_command(self.region)
        if not login_result["success"]:
            return self._error_response(f"Failed to get ECR login: {login_result['error']}")

        # Push image
        ecr_uri = f"{repo_result['repository_uri']}:{self.config.tag}"
        push_result = push_image(ecr_uri)
        if not push_result["success"]:
            return self._error_response(f"Failed to push image: {push_result['error']}")

        return {
            "success": True,
            "image_id": build_result["image_id"],
            "local_tag": f"{self.config.repository_name}:{self.config.tag}",
            "ecr_uri": ecr_uri,
            "repository_uri": repo_result["repository_uri"],
            "build_logs": build_result["logs"],
            "push_logs": push_result["logs"],
        }

    def _build_success_response(self, build_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create success response for local build."""
        return {
            "success": True,
            "image_id": build_result["image_id"],
            "local_tag": f"{self.config.repository_name}:{self.config.tag}",
            "build_logs": build_result["logs"],
        }

    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response."""
        return {"success": False, "error": error_message}


# Public API Functions
def list_base_images(
    framework: Optional[str] = None,
    use_case: Optional[str] = None,
    device_type: Optional[str] = None,
    platform: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List available base images for Deep Learning Containers.

    Args:
        framework: Framework filter (pytorch, tensorflow, etc.)
        use_case: Use case filter (training, inference)
        device_type: Device type filter (cpu, gpu)
        platform: Platform filter (ec2, sagemaker)

    Returns:
        Dict containing list of available base images
    """
    try:
        base_images = BaseImageRegistry.get_base_images()
        filtered_images = ImageFilter.apply_filters(
            base_images, framework, use_case, device_type, platform
        )

        # Convert to dict format for JSON serialization
        images_dict = [
            {
                "framework": img.framework,
                "version": img.version,
                "use_case": img.use_case,
                "device_type": img.device_type,
                "python_version": img.python_version,
                "cuda_version": img.cuda_version,
                "os": img.os,
                "platform": img.platform,
                "uri": img.uri,
            }
            for img in filtered_images
        ]

        return {"images": images_dict}

    except Exception as e:
        logger.error(f"Failed to list base images: {e}")
        return {"success": False, "error": str(e)}


def create_custom_dockerfile(
    base_image: str,
    packages: List[str] = [],
    python_packages: List[str] = [],
    custom_commands: List[str] = [],
    environment_variables: Dict[str, str] = {},
) -> Dict[str, Any]:
    """
    Create a custom Dockerfile based on a DLC base image.

    Args:
        base_image: Base image URI
        packages: System packages to install
        python_packages: Python packages to install
        custom_commands: Custom commands to add to Dockerfile
        environment_variables: Environment variables to set

    Returns:
        Dict containing Dockerfile content
    """
    try:
        config = DockerfileConfig(
            base_image=base_image,
            packages=packages or [],
            python_packages=python_packages or [],
            custom_commands=custom_commands or [],
            environment_variables=environment_variables or {},
        )

        builder = DockerfileBuilder(config)
        dockerfile_content = builder.build()

        return {"dockerfile_content": dockerfile_content}

    except Exception as e:
        logger.error(f"Failed to create custom Dockerfile: {e}")
        return {"success": False, "error": str(e)}


def build_custom_dlc_image(
    base_image: str,
    repository_name: str,
    tag: str,
    dockerfile_content: str,
    push_to_ecr: bool = False,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a custom DLC image.

    Args:
        base_image: Base image URI
        repository_name: ECR repository name
        tag: Image tag
        dockerfile_content: Dockerfile content
        push_to_ecr: Whether to push the image to ECR
        region: AWS region

    Returns:
        Dict containing build results
    """
    config = BuildConfig(
        base_image=base_image,
        repository_name=repository_name,
        tag=tag,
        dockerfile_content=dockerfile_content,
        push_to_ecr=push_to_ecr,
        region=region,
    )

    builder = CustomImageBuilder(config)
    return builder.build()


def register_module(mcp: FastMCP) -> None:
    """Register the image building module with the MCP server."""

    @mcp.tool(
        name="list_base_images",
        description="List available base images for Deep Learning Containers",
    )
    async def mcp_list_base_images(
        framework: Optional[str] = None,
        use_case: Optional[str] = None,
        device_type: Optional[str] = None,
        platform: Optional[str] = None,
    ) -> Dict[str, Any]:
        return list_base_images(framework, use_case, device_type, platform)

    @mcp.tool(
        name="create_custom_dockerfile",
        description="Create a custom Dockerfile based on a DLC base image",
    )
    async def mcp_create_dockerfile(
        base_image: str,
        packages: List[str] = [],
        python_packages: List[str] = [],
        custom_commands: List[str] = [],
        environment_variables: Dict[str, str] = {},
    ) -> Dict[str, Any]:
        return create_custom_dockerfile(
            base_image, packages, python_packages, custom_commands, environment_variables
        )

    @mcp.tool(name="build_custom_dlc_image", description="Build a custom DLC image")
    async def mcp_build_image(
        base_image: str,
        repository_name: str,
        tag: str,
        dockerfile_content: str,
        push_to_ecr: bool = False,
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        return build_custom_dlc_image(
            base_image, repository_name, tag, dockerfile_content, push_to_ecr, region
        )
