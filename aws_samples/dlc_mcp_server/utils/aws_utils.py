###
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# Copyright Amazon.com, Inc. and its affiliates. All Rights Reserved.
#   SPDX-License-Identifier: MIT
######

"""AWS utilities for the DLC MCP Server."""

import logging
import boto3
import os
import subprocess
from typing import Dict, Any, List, Optional
from botocore.exceptions import ClientError, NoCredentialsError
import re
from aws_samples.dlc_mcp_server.utils.config import get_aws_region

logger = logging.getLogger(__name__)

# AWS DLC Production Account ID
AWS_DLC_ACCOUNT_ID = "763104351884"

SUPPORTED_FRAMEWORKS = {
    "pytorch": {
        "versions": {
            "2.7": {
                "cuda": "12.8",
                "eol": "2026-04-23",
                "status": "supported"
            },
            "2.6": {
                "cuda": {
                    "training": "12.6",
                    "inference": "12.4"
                },
                "eol": "2026-01-29",
                "status": "supported"
            },
            "2.5": {
                "cuda": "12.4",
                "eol": "2025-10-29",
                "status": "supported"
            },
            "2.4": {
                "cuda": "12.4",
                "eol": "2025-07-24",
                "status": "supported"
            }
        }
    },
    "tensorflow": {
        "versions": {
            "2.18": {
                "cuda": {
                    "training": "12.5",
                    "inference": "12.2"
                },
                "eol": "2026-01-24",
                "status": "supported"
            }
        }
    }
}

def get_ecr_client(region: Optional[str] = None) -> Any:
    """
    Get an ECR client.

    Args:
        region (Optional[str]): AWS region

    Returns:
        Any: ECR client
    """
    region = region or get_aws_region()
    return boto3.client("ecr", region_name=region)


def get_sagemaker_client(region: Optional[str] = None) -> Any:
    """
    Get a SageMaker client.

    Args:
        region (Optional[str]): AWS region

    Returns:
        Any: SageMaker client
    """
    region = region or get_aws_region()
    return boto3.client("sagemaker", region_name=region)


def get_ecs_client(region: Optional[str] = None) -> Any:
    """
    Get an ECS client.

    Args:
        region (Optional[str]): AWS region

    Returns:
        Any: ECS client
    """
    region = region or get_aws_region()
    return boto3.client("ecs", region_name=region)


def get_eks_client(region: Optional[str] = None) -> Any:
    """
    Get an EKS client.

    Args:
        region (Optional[str]): AWS region

    Returns:
        Any: EKS client
    """
    region = region or get_aws_region()
    return boto3.client("eks", region_name=region)


def get_ec2_client(region: Optional[str] = None) -> Any:
    """
    Get an EC2 client.

    Args:
        region (Optional[str]): AWS region

    Returns:
        Any: EC2 client
    """
    region = region or get_aws_region()
    return boto3.client("ec2", region_name=region)


def create_ecr_repository(repository_name: str, region: Optional[str] = None) -> Dict[str, Any]:
    """
    Create an ECR repository.

    Args:
        repository_name (str): Repository name
        region (Optional[str]): AWS region

    Returns:
        Dict[str, Any]: Repository details
    """
    try:
        ecr = get_ecr_client(region)
        response = ecr.create_repository(
            repositoryName=repository_name,
            imageScanningConfiguration={"scanOnPush": True},
            encryptionConfiguration={"encryptionType": "AES256"},
        )

        return {
            "success": True,
            "repository_uri": response["repository"]["repositoryUri"],
            "repository_name": repository_name,
        }
    except ClientError as e:
        if e.response["Error"]["Code"] == "RepositoryAlreadyExistsException":
            # Repository already exists, get its URI
            try:
                response = ecr.describe_repositories(repositoryNames=[repository_name])
                return {
                    "success": True,
                    "repository_uri": response["repositories"][0]["repositoryUri"],
                    "repository_name": repository_name,
                    "note": "Repository already exists",
                }
            except Exception as describe_error:
                logger.error(
                    f"Failed to describe existing repository {repository_name}: {describe_error}"
                )
                return {"success": False, "error": str(describe_error)}
        else:
            logger.error(f"Failed to create ECR repository {repository_name}: {e}")
            return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Failed to create ECR repository {repository_name}: {e}")
        return {"success": False, "error": str(e)}


def get_ecr_login_command(region: Optional[str] = None, prod: bool = False) -> Dict[str, Any]:
    """
    Get ECR login command and perform Docker login.

    Args:
        region (Optional[str]): AWS region
        prod (bool): Whether to authenticate with AWS DLC production account

    Returns:
        Dict[str, Any]: Login command details
    """
    try:
        region = region or get_aws_region()
        ecr = get_ecr_client(region)

        if prod:
            # Authenticate with AWS DLC production account
            account_id = AWS_DLC_ACCOUNT_ID

            # Get authorization token for the specific registry
            try:
                token_response = ecr.get_authorization_token(registryIds=[account_id])
            except ClientError as e:
                logger.error(f"Failed to get auth token for registry {account_id}: {e}")
                return {
                    "success": False,
                    "error": f"Failed to authenticate with AWS DLC registry: {str(e)}",
                }
        else:
            # Use default behavior for user's own account
            try:
                token_response = ecr.get_authorization_token()
            except ClientError as e:
                logger.error(f"Failed to get auth token: {e}")
                return {"success": False, "error": str(e)}

        auth_data = token_response["authorizationData"][0]
        endpoint = auth_data["proxyEndpoint"]
        auth_token = auth_data["authorizationToken"]

        # Perform Docker login
        try:
            import base64

            # Decode the token
            username, password = base64.b64decode(auth_token).decode().split(":")

            # Execute docker login command
            docker_login_cmd = [
                "docker",
                "login",
                "--username",
                username,
                "--password-stdin",
                endpoint,
            ]

            result = subprocess.run(
                docker_login_cmd, input=password, text=True, capture_output=True, check=True
            )

            logger.info(f"Successfully logged into ECR: {endpoint}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Docker login failed: {e}")
            return {
                "success": False,
                "error": f"Docker login failed: {e.stderr if e.stderr else str(e)}",
            }
        except Exception as e:
            logger.error(f"Failed to perform Docker login: {e}")
            return {"success": False, "error": f"Failed to perform Docker login: {str(e)}"}

        return {
            "success": True,
            "endpoint": endpoint,
            "auth_token": auth_token,
            "expires_at": auth_data["expiresAt"].isoformat(),
            "account_id": account_id if prod else endpoint.split(".")[0].split("//")[1],
            "region": region,
            "message": f"Successfully authenticated with ECR: {endpoint}",
        }

    except NoCredentialsError:
        return {
            "success": False,
            "error": "AWS credentials not found. Please run 'aws configure' first.",
        }
    except Exception as e:
        logger.error(f"Failed to get ECR login command: {e}")
        return {"success": False, "error": str(e)}

def list_dlc_repositories(
    region: Optional[str] = None, repository_name: List[str] = []
) -> List[Dict[str, Any]]:
    """
    List DLC repositories and their images from AWS DLC registry, filtering for supported frameworks.

    Args:
        region (Optional[str]): AWS region, defaults to current region

    Returns:
        List[Dict[str, Any]]: List of repositories and their images
    """
    try:
        region = region or get_aws_region()
        ecr = get_ecr_client(region)
        repositories = []

        try:
            response = ecr.describe_repositories(
                repositoryNames=repository_name,
                registryId=AWS_DLC_ACCOUNT_ID,
            )

            repo_list = response.get("repositories", [])

            for repo in repo_list:
                repo_name = repo["repositoryName"]

                # Check if the repository is for a supported framework
                if any(framework in repo_name.lower() for framework in SUPPORTED_FRAMEWORKS):
                    try:
                        image_response = ecr.describe_images(
                            registryId=AWS_DLC_ACCOUNT_ID,
                            repositoryName=repo_name,
                            maxResults=20,
                            filter={"tagStatus": "TAGGED"},
                        )

                        images = []
                        for image_detail in image_response.get("imageDetails", []):
                            tags = image_detail.get("imageTags", [])
                            for tag in tags:
                                images.append(
                                    {
                                        "tag": tag,
                                        "digest": image_detail.get("imageDigest", ""),
                                        "pushed_at": image_detail.get("imagePushedAt", ""),
                                        "size": image_detail.get("imageSizeInBytes", 0),
                                        "full_uri": f"{AWS_DLC_ACCOUNT_ID}.dkr.ecr.{region}.amazonaws.com/{repo_name}:{tag}",
                                    }
                                )

                        if images:
                            repositories.append(
                                {
                                    "repositoryName": repo_name,
                                    "repositoryUri": repo.get("repositoryUri", ""),
                                    "repositoryArn": repo.get("repositoryArn", ""),
                                    "images": images,
                                    "image_count": len(images),
                                }
                            )

                    except Exception as image_error:
                        logger.warning(f"Error getting images for {repo_name}: {image_error}")
                        continue

            next_token = response.get("nextToken")
            while next_token:
                next_response = ecr.describe_repositories(
                    registryId=AWS_DLC_ACCOUNT_ID, maxResults=100, nextToken=next_token
                )
                next_token = next_response.get("nextToken")

            logger.info(f"Found {len(repositories)} DLC repositories with images for supported frameworks")
            return repositories

        except Exception as repo_error:
            logger.error(f"Error listing repositories: {repo_error}")
            return []

    except Exception as e:
        logger.error(f"Error in list_dlc_repositories: {e}")
        return []

def filter_dlc_images(
    repositories: List[Dict[str, Any]],
    framework: Optional[str] = None,
    image_type: Optional[str] = None,
    python_version: Optional[str] = None,
    cuda_version: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Filter DLC images based on specified criteria and support policy.

    Args:
        repositories: List of repositories and their images
        framework: Framework name (e.g., 'pytorch', 'tensorflow')
        image_type: Type of image (e.g., 'training', 'inference')
        python_version: Python version (e.g., '3.9', '3.10')
        cuda_version: CUDA version (e.g., '11.8', '12.4')

    Returns:
        List[Dict[str, Any]]: Filtered list of repositories and images
    """
    if not repositories or framework not in SUPPORTED_FRAMEWORKS:
        return []

    filtered_repos = []

    for repo in repositories:
        repo_name = repo.get("repositoryName", "") or repo.get("name", "")
        repo_name_lower = repo_name.lower()

        # Filter by framework
        if framework and framework.lower() not in repo_name_lower:
            continue

        # Filter by image type
        if image_type and image_type.lower() not in repo_name_lower:
            continue

        images = repo.get("images", [])
        filtered_images = []

        for image in images:
            tag = image.get("tag", "") or image.get("imageTag", "")
            if not tag:
                continue

            # Extract framework version from tag
            version_match = re.search(r'(\d+\.\d+)', tag)
            if not version_match:
                continue
            
            fw_version = version_match.group(1)

            # Check if version is supported
            if not is_supported_version(framework, fw_version):
                continue

            # Filter by Python version
            if python_version:
                py_version = f"py{python_version.replace('.', '')}"
                if py_version not in tag.lower():
                    continue

            filtered_images.append(image)

        if filtered_images:
            filtered_repo = repo.copy()
            filtered_repo["images"] = filtered_images
            filtered_repo["filtered_image_count"] = len(filtered_images)
            filtered_repos.append(filtered_repo)

    return filtered_repos

def is_supported_version(framework: str, version: str) -> bool:
    """Check if the framework version is supported according to the support policy."""
    if framework not in SUPPORTED_FRAMEWORKS:
        return False
    
    framework_versions = SUPPORTED_FRAMEWORKS[framework]["versions"]
    if version not in framework_versions:
        return False
    
    version_info = framework_versions[version]
    return version_info["status"] == "supported"
