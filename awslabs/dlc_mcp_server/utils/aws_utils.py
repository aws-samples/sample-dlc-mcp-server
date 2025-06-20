"""AWS utilities for the DLC MCP Server."""

import logging
import boto3
from typing import Dict, Any, List, Optional

from awslabs.dlc_mcp_server.utils.config import get_aws_region

logger = logging.getLogger(__name__)


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
            encryptionConfiguration={"encryptionType": "AES256"}
        )
        
        return {
            "success": True,
            "repository_uri": response["repository"]["repositoryUri"],
            "repository_name": repository_name
        }
    except ecr.exceptions.RepositoryAlreadyExistsException:
        # Repository already exists, get its URI
        response = ecr.describe_repositories(repositoryNames=[repository_name])
        return {
            "success": True,
            "repository_uri": response["repositories"][0]["repositoryUri"],
            "repository_name": repository_name,
            "note": "Repository already exists"
        }
    except Exception as e:
        logger.error(f"Failed to create ECR repository {repository_name}: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def get_ecr_login_command(region: Optional[str] = None) -> Dict[str, Any]:
    """
    Get ECR login command.
    
    Args:
        region (Optional[str]): AWS region
        
    Returns:
        Dict[str, Any]: Login command details
    """
    try:
        ecr = get_ecr_client(region)
        token = ecr.get_authorization_token()
        
        auth_data = token["authorizationData"][0]
        endpoint = auth_data["proxyEndpoint"]
        auth_token = auth_data["authorizationToken"]
        
        return {
            "success": True,
            "endpoint": endpoint,
            "auth_token": auth_token,
            "expires_at": auth_data["expiresAt"].isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get ECR login command: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def list_dlc_repositories(region: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List DLC repositories in the AWS DLC registry.
    
    Args:
        region (Optional[str]): AWS region
        
    Returns:
        List[Dict[str, Any]]: List of repositories
    """
    try:
        # This is a simplified implementation since we don't have direct access to the AWS DLC registry
        # In a real implementation, you would query the public ECR registry
        frameworks = ["pytorch", "tensorflow", "mxnet", "autogluon"]
        use_cases = ["training", "inference"]
        
        repositories = []
        for framework in frameworks:
            for use_case in use_cases:
                repositories.append({
                    "name": f"{framework}-{use_case}",
                    "uri": f"763104351884.dkr.ecr.{region or get_aws_region()}.amazonaws.com/{framework}-{use_case}"
                })
        
        return repositories
    except Exception as e:
        logger.error(f"Failed to list DLC repositories: {e}")
        return []
