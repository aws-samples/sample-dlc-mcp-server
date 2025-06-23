"""Module for AWS Deep Learning Containers operations."""

import logging
import os
import subprocess
from typing import Dict, Any, List, Optional

import docker
from docker import DockerClient
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from awslabs.dlc_mcp_server.utils.aws_utils import (
    get_ecr_login_command,
    filter_dlc_images,
    list_dlc_repositories,
)
from pydantic import Field
import re

FRAMEWORK_TYPES = {
    "pytorch": ["training", "inference"],
    "tensorflow": ["training", "inference"],
    "mxnet": ["training", "inference"],
    "huggingface": ["training", "inference"],
    "autogluon": ["training"],
    "stabilityai": ["inference"],
    "djl": ["inference"],
}

# Constants
DEFAULT_REGION = "us-west-2"
DEFAULT_GPU_PER_NODE = 1
DEFAULT_FRAMEWORK = "pytorch"

# Docker runtime configurations
NVIDIA_RUNTIME = "nvidia"

# Environment variables
NCCL_DEBUG_CONFIG = {"NCCL_DEBUG": "INFO", "PYTHONUNBUFFERED": "1"}
TENSORFLOW_CONFIG = {"TF_CONFIG": "auto", "PYTHONUNBUFFERED": "1"}

logger = logging.getLogger(__name__)


class DockerNotAvailableError(Exception):
    """Raised when Docker is not available or accessible."""

    pass


class DLCEnvironmentChecker:
    """Utility class for checking DLC environment prerequisites."""

    @staticmethod
    def check_gpu_availability() -> bool:
        """Check if GPU support is available via nvidia-smi."""
        try:
            subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    @staticmethod
    def check_docker_availability() -> tuple[bool, Optional[Dict[str, Any]]]:
        """Check if Docker is available and return version info."""
        try:
            client = docker.from_env()
            docker_version = client.version()
            return True, docker_version
        except Exception:
            return False, None


class DLCContainerManager:
    """Manages DLC container operations."""

    def __init__(self):
        self.client: Optional[DockerClient] = None
        self._initialize_docker_client()

    def _initialize_docker_client(self) -> None:
        """Initialize Docker client."""
        try:
            self.client = docker.from_env()
            logger.info("Docker client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            self.client = None

    def _ensure_docker_client(self) -> DockerClient:
        """Ensure Docker client is available, raise exception if not."""
        if self.client is None:
            raise DockerNotAvailableError(
                "Docker client is not available. Please ensure Docker is installed and running."
            )
        return self.client

    def run_container(
        self,
        image_uri: str,
        container_name: str,
        gpu: bool = False,
        command: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a DLC container with specified configuration.

        Args:
            image_uri: DLC image URI
            container_name: Name for the container
            gpu: Whether to enable GPU support
            command: Command to run in container

        Returns:
            Dict containing success status and container details
        """
        try:
            client = self._ensure_docker_client()

            container_config = self._build_container_config(container_name, gpu, command)

            # Pull image and run container
            self._pull_image(image_uri, client)
            container = client.containers.run(image_uri, **container_config)

            return {
                "success": True,
                "container_id": container.id,
                "status": container.status,
                "name": container_name,
            }

        except DockerNotAvailableError as e:
            logger.error(f"Docker not available: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Failed to run container {container_name}: {e}")
            return {"success": False, "error": str(e)}

    def _build_container_config(
        self, container_name: str, gpu: bool, command: Optional[str]
    ) -> Dict[str, Any]:
        """Build container configuration dictionary."""
        config = {"name": container_name, "detach": True}

        if command:
            config["command"] = command

        if gpu:
            config["runtime"] = NVIDIA_RUNTIME

        return config

    def _pull_image(self, image_uri: str, client: DockerClient) -> None:
        """Pull Docker image."""
        try:
            client.images.pull(image_uri)
            logger.info(f"Successfully pulled image: {image_uri}")
        except Exception as e:
            logger.error(f"Failed to pull image {image_uri}: {e}")
            raise


class DLCDistributedTrainingConfig:
    """Handles distributed training configuration for DLC."""

    @staticmethod
    def create_config(
        image_uri: str,
        num_nodes: int,
        gpu_per_node: int = DEFAULT_GPU_PER_NODE,
        framework: str = DEFAULT_FRAMEWORK,
    ) -> Dict[str, Any]:
        """
        Create distributed training configuration.

        Args:
            image_uri: DLC image URI
            num_nodes: Number of nodes for distributed training
            gpu_per_node: Number of GPUs per node
            framework: ML framework (pytorch/tensorflow)

        Returns:
            Configuration dictionary
        """
        try:
            base_config = {
                "image_uri": image_uri,
                "num_nodes": num_nodes,
                "gpu_per_node": gpu_per_node,
            }

            if framework.lower() == "pytorch":
                return DLCDistributedTrainingConfig._create_pytorch_config(
                    base_config, gpu_per_node
                )
            else:
                return DLCDistributedTrainingConfig._create_tensorflow_config(base_config)

        except Exception as e:
            logger.error(f"Failed to create distributed training config: {e}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def _create_pytorch_config(base_config: Dict[str, Any], gpu_per_node: int) -> Dict[str, Any]:
        """Create PyTorch distributed configuration."""
        return {
            "success": True,
            "config": {
                **base_config,
                "environment": NCCL_DEBUG_CONFIG,
                "launch_command": f"python -m torch.distributed.launch --nproc_per_node={gpu_per_node}",
            },
        }

    @staticmethod
    def _create_tensorflow_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create TensorFlow distributed configuration."""
        return {"success": True, "config": {**base_config, "environment": TENSORFLOW_CONFIG}}


# Constants
DEFAULT_REGION = "us-west-2"
DEFAULT_GPU_PER_NODE = 1
DEFAULT_FRAMEWORK = "pytorch"

logger = logging.getLogger(__name__)


def check_aws_configuration() -> Dict[str, Any]:
    """
    Check if AWS CLI is configured properly.

    Returns:
        Dict containing success status and configuration details or error message
    """
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, PartialCredentialsError

        # Try to create a session
        session = boto3.Session()

        # Try to get credentials
        credentials = session.get_credentials()
        if not credentials:
            return {
                "success": False,
                "error": "AWS credentials not found. Please run 'aws configure' to set up your credentials.",
                "setup_instructions": [
                    "1. Run 'aws configure' in your terminal",
                    "2. Enter your AWS Access Key ID",
                    "3. Enter your AWS Secret Access Key",
                    "4. Enter your default region (e.g., us-west-2)",
                    "5. Enter your default output format (json recommended)",
                ],
            }

        # Try to get region
        region = session.region_name
        if not region:
            return {
                "success": False,
                "error": "AWS region not configured. Please run 'aws configure' and set your default region.",
                "setup_instructions": [
                    "Run 'aws configure' and set your default region (e.g., us-west-2)"
                ],
            }

        # Test if credentials work by making a simple API call
        try:
            sts_client = session.client("sts")
            identity = sts_client.get_caller_identity()

            return {
                "success": True,
                "region": region,
                "account_id": identity.get("Account"),
                "user_arn": identity.get("Arn"),
                "message": "AWS configuration is valid and working",
            }
        except Exception as api_error:
            return {
                "success": False,
                "error": f"AWS credentials are configured but not working: {str(api_error)}",
                "setup_instructions": [
                    "Please check your AWS credentials and permissions",
                    "Run 'aws sts get-caller-identity' to test your configuration",
                ],
            }

    except NoCredentialsError:
        return {
            "success": False,
            "error": "AWS credentials not found. Please run 'aws configure' to set up your credentials.",
            "setup_instructions": [
                "1. Install AWS CLI: https://aws.amazon.com/cli/",
                "2. Run 'aws configure'",
                "3. Enter your AWS credentials and region",
            ],
        }
    except PartialCredentialsError:
        return {
            "success": False,
            "error": "Incomplete AWS credentials. Please run 'aws configure' to complete your setup.",
            "setup_instructions": ["Run 'aws configure' and ensure all fields are filled"],
        }
    except ImportError:
        return {
            "success": False,
            "error": "boto3 library not found. Please install it: pip install boto3",
            "setup_instructions": [
                "1. Install boto3: pip install boto3",
                "2. Install AWS CLI: https://aws.amazon.com/cli/",
                "3. Run 'aws configure'",
            ],
        }
    except Exception as e:
        return {"success": False, "error": f"AWS configuration check failed: {str(e)}"}


async def list_available_dlc_images(
    framework: Optional[str] = None,
    python_version: Optional[str] = None,
    cuda_version: Optional[str] = None,
    region: Optional[str] = None,
    repository_name: Optional[List[str]] = [],
) -> Dict[str, Any]:
    """
    List available DLC images with automatic ECR setup and AWS configuration check.

    Args:
        framework: Filter by ML framework (pytorch, tensorflow, etc.)
        python_version: Filter by Python version (3.8, 3.9, 3.10, etc.)
        cuda_version: Filter by CUDA version (11.8, 12.1, etc.)
        region: AWS region (defaults to configured region)

    Returns:
        Dict containing filtered DLC images or error information
    """
    try:
        # Step 1: Check AWS configuration
        logger.info("Checking AWS configuration...")
        aws_check = check_aws_configuration()
        if not aws_check["success"]:
            return {
                "success": False,
                "error": aws_check["error"],
                "setup_instructions": aws_check.get("setup_instructions", []),
                "action_required": "Please configure AWS CLI first",
            }

        # Step 2: Use region from AWS config if not provided
        if not region:
            region = aws_check.get("region", DEFAULT_REGION)

        logger.info(f"Using AWS region: {region}")

        # Step 3: Setup ECR authentication automatically
        logger.info("Setting up ECR authentication...")
        ecr_setup = get_ecr_login_command(prod=True, region=region)
        if not ecr_setup["success"]:
            return {
                "success": False,
                "error": f"Failed to setup ECR authentication: {ecr_setup['error']}",
                "troubleshooting": [
                    "Check your AWS permissions for ECR access",
                    "Ensure you have the correct AWS region configured",
                    "Verify Docker is running on your system",
                ],
            }

        # Step 4: List DLC repositories
        logger.info("Fetching DLC repositories...")
        repositories = list_dlc_repositories(region)
        if not repositories:
            return {
                "success": False,
                "error": "No DLC repositories found or failed to access repositories",
                "troubleshooting": [
                    "Check your internet connection",
                    "Verify ECR permissions in your AWS account",
                    f"Ensure region {region} has DLC repositories available",
                ],
            }

        # Step 5: Filter DLC images based on criteria
        logger.info(
            f"Filtering images with criteria - Framework: {framework}, Python: {python_version}, CUDA: {cuda_version}"
        )
        filtered_images = filter_dlc_images(repositories, framework, python_version, cuda_version)

        # Step 6: Provide helpful suggestions if no images found
        if not filtered_images:
            suggestions = []
            if framework:
                suggestions.append(f"Try without framework filter (currently: {framework})")
            if python_version:
                suggestions.append(
                    f"Try without Python version filter (currently: {python_version})"
                )
            if cuda_version:
                suggestions.append(f"Try without CUDA version filter (currently: {cuda_version})")

            return {
                "success": True,
                "images": [],
                "message": "No images found matching your criteria",
                "suggestions": suggestions or ["Try with broader search criteria"],
                "total_available_images": len(
                    [img for repo in repositories for img in repo.get("images", [])]
                ),
                "region": region,
            }

        return {
            "success": True,
            "images": filtered_images,
            "region": region,
            "total_images": len(filtered_images),
            "applied_filters": {
                "framework": framework,
                "python_version": python_version,
                "cuda_version": cuda_version,
            },
        }

    except Exception as e:
        logger.error(f"Failed to list available DLC images: {e}")
        return {
            "success": False,
            "error": str(e),
            "troubleshooting": [
                "Check your AWS configuration",
                "Verify your network connection",
                "Ensure Docker is running",
            ],
        }


# Main module functions
def setup_dlc_environment() -> Dict[str, Any]:
    """
    Setup environment for DLC usage including NVIDIA drivers check for GPU instances.

    Returns:
        Dict containing environment setup results
    """
    try:
        gpu_available = DLCEnvironmentChecker.check_gpu_availability()
        docker_available, docker_version = DLCEnvironmentChecker.check_docker_availability()

        return {
            "success": True,
            "environment": {
                "gpu_available": gpu_available,
                "docker_available": docker_available,
                "docker_version": docker_version,
            },
        }
    except Exception as e:
        logger.error(f"Failed to setup DLC environment: {e}")
        return {"success": False, "error": str(e)}


def run_dlc_container(
    image_uri: str,
    container_name: str,
    gpu: bool = False,
    command: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a DLC container with specified configuration.

    Args:
        image_uri: DLC image URI
        container_name: Name for the container
        gpu: Whether to enable GPU support
        command: Command to run in container

    Returns:
        Dict containing container run results
    """
    try:
        container_manager = DLCContainerManager()
        return container_manager.run_container(image_uri, container_name, gpu, command)
    except Exception as e:
        logger.error(f"Failed to create container manager: {e}")
        return {"success": False, "error": str(e)}


def get_repo_name(framework: str, repo_type: str) -> List[str]:
    """Construct repository name from framework and type."""
    return [f"{framework}-{repo_type}"]


def setup_distributed_training(
    image_uri: str,
    num_nodes: int,
    gpu_per_node: int = DEFAULT_GPU_PER_NODE,
    framework: str = DEFAULT_FRAMEWORK,
) -> Dict[str, Any]:
    """
    Setup distributed training configuration for DLC.

    Args:
        image_uri: DLC image URI
        num_nodes: Number of nodes for distributed training
        gpu_per_node: Number of GPUs per node
        framework: ML framework (pytorch/tensorflow)

    Returns:
        Dict containing configuration details
    """
    return DLCDistributedTrainingConfig.create_config(image_uri, num_nodes, gpu_per_node, framework)


def register_module(mcp: FastMCP) -> None:
    """Register DLC tools with the MCP server."""

    @mcp.tool(name="check_aws_config", description="Check if AWS CLI is properly configured")
    async def mcp_check_aws_config() -> Dict[str, Any]:
        return check_aws_configuration()

    @mcp.tool(
        name="setup_ecr_prod",
        description="Authenticate with ECR to access latest DLC images from production account (763104351884)",
    )
    async def mcp_setup_ecr_prod(region: str = DEFAULT_REGION) -> Dict[str, Any]:
        return get_ecr_login_command(prod=True, region=region)

    @mcp.tool(
        name="list_dlc_repos",
        description="List available DLC repositories (Requires ECR authentication)",
    )
    async def mcp_list_dlc_repos(
        framework: Optional[str] = Field(
            None,
            description="ML framework (pytorch, tensorflow, mxnet, huggingface, autogluon, stabilityai, djl)",
        ),
        repo_type: Optional[str] = Field(None, description="Repository type (training, inference)"),
        region: str = DEFAULT_REGION,
    ) -> List[Dict[str, Any]]:
        if framework:
            if framework not in FRAMEWORK_TYPES:
                raise ValueError(
                    f"Invalid framework. Choose from: {', '.join(FRAMEWORK_TYPES.keys())}"
                )

            if repo_type:
                if repo_type not in FRAMEWORK_TYPES[framework]:
                    raise ValueError(
                        f"Invalid type for {framework}. Choose from: {', '.join(FRAMEWORK_TYPES[framework])}"
                    )

                repository_name = get_repo_name(framework, repo_type)
                return list_dlc_repositories(region, repository_name)
            else:
                # If only framework is specified, list all repo types for that framework
                results = []
                for rt in FRAMEWORK_TYPES[framework]:
                    repository_name = get_repo_name(framework, rt)
                    results.extend(list_dlc_repositories(region, repository_name))
                return results
        else:
            # If neither framework nor type specified, prompt user
            raise ValueError(
                "Please specify framework and repository type.\n"
                f"Available frameworks: {', '.join(FRAMEWORK_TYPES.keys())}\n"
                "Repository types: training, inference (availability depends on framework)"
            )

    @mcp.tool(
        name="list_dlc_images",
        description="List and filter available DLC images with automatic AWS setup and ECR authentication",
    )
    async def mcp_list_dlc_images(
        framework: Optional[str] = Field(
            None,
            description="Filter by ML framework (pytorch, tensorflow, mxnet, huggingface, autogluon, stabilityai, djl)",
        ),
        repo_type: Optional[str] = Field(None, description="Repository type (training, inference)"),
        python_version: Optional[str] = Field(
            None, description="Filter by Python version (3.8, 3.9, 3.10, 3.11, 3.12)"
        ),
        cuda_version: Optional[str] = Field(
            None, description="Filter by CUDA version (11.8, 12.1, 12.4, 12.6)"
        ),
        region: Optional[str] = Field(
            None, description="AWS region (defaults to configured region)"
        ),
    ) -> Dict[str, Any]:
        if not framework or not repo_type:
            raise ValueError(
                "Please specify both framework and repository type.\n"
                f"Available frameworks: {', '.join(FRAMEWORK_TYPES.keys())}\n"
                "Repository types: training, inference (availability depends on framework)"
            )

        if framework not in FRAMEWORK_TYPES:
            raise ValueError(f"Invalid framework. Choose from: {', '.join(FRAMEWORK_TYPES.keys())}")

        if repo_type not in FRAMEWORK_TYPES[framework]:
            raise ValueError(
                f"Invalid type for {framework}. Choose from: {', '.join(FRAMEWORK_TYPES[framework])}"
            )

        repository_name = get_repo_name(framework, repo_type)
        return await list_available_dlc_images(
            framework, python_version, cuda_version, region, repository_name
        )

    @mcp.tool(name="run_dlc_container", description="Run a DLC container for training or inference")
    async def mcp_run_container(
        image_uri: str = Field(..., description="DLC image URI"),
        container_name: str = Field(..., description="Container name"),
        gpu: bool = Field(False, description="Enable GPU support"),
        command: Optional[str] = Field(None, description="Command to run"),
    ) -> Dict[str, Any]:
        return run_dlc_container(image_uri, container_name, gpu, command)

    @mcp.tool(
        name="setup_distributed_training", description="Configure distributed training for DLC"
    )
    async def mcp_setup_distributed(
        image_uri: str = Field(..., description="DLC image URI"),
        num_nodes: int = Field(..., description="Number of nodes"),
        gpu_per_node: int = Field(DEFAULT_GPU_PER_NODE, description="GPUs per node"),
        framework: str = Field(DEFAULT_FRAMEWORK, description="ML framework"),
    ) -> Dict[str, Any]:
        return setup_distributed_training(image_uri, num_nodes, gpu_per_node, framework)

    # Prompt patterns - Updated to use list_dlc_images directly
    @mcp.prompt("list available images")
    def list_images_prompt():
        """List available DLC images with automatic setup"""
        return ["list_dlc_images"]

    @mcp.prompt("show pytorch images")
    def pytorch_images_prompt():
        """Show available PyTorch DLC images"""
        return ["list_dlc_images"]

    @mcp.prompt("show tensorflow images")
    def tensorflow_images_prompt():
        """Show available TensorFlow DLC images"""
        return ["list_dlc_images"]

    @mcp.prompt("list gpu images")
    def gpu_images_prompt():
        """List GPU-enabled DLC images"""
        return ["list_dlc_images"]

    @mcp.prompt("check aws setup")
    def check_aws_prompt():
        """Check AWS configuration status"""
        return ["check_aws_config"]

    @mcp.prompt("run container")
    def run_container_prompt():
        """Run a DLC container"""
        return ["run_dlc_container"]

    @mcp.prompt("distributed training")
    def distributed_training_prompt():
        """Setup distributed training with DLC"""
        return ["setup_distributed_training"]

    # Workflow prompts - Simplified to use direct image listing
    @mcp.prompt("find images")
    def find_images_prompt():
        """Find and list DLC images"""
        return ["list_dlc_images"]

    @mcp.prompt("prepare training")
    def prepare_training_prompt():
        """Prepare for training by listing images and setting up container"""
        return ["list_dlc_images", "run_dlc_container"]

    @mcp.prompt("get ready for ml")
    def ml_ready_prompt():
        """Complete ML setup workflow"""
        return ["check_aws_config", "list_dlc_images"]
