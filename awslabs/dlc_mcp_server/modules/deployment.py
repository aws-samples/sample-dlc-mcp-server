"""Module for deploying DLC images to various AWS platforms."""

import logging
import json
from typing import Dict, Any, Optional, List

from mcp.server.fastmcp import FastMCP

from awslabs.dlc_mcp_server.utils.aws_utils import (
    get_sagemaker_client,
    get_ecs_client,
    get_eks_client,
    get_ec2_client
)
from awslabs.dlc_mcp_server.utils.config import get_aws_region

logger = logging.getLogger(__name__)


def deploy_to_sagemaker(
    image_uri: str,
    model_name: str,
    instance_type: str,
    role_arn: str,
    model_data_url: Optional[str] = None,
    environment: Optional[Dict[str, str]] = None,
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Deploy a DLC image to Amazon SageMaker.
    
    Args:
        image_uri (str): DLC image URI
        model_name (str): Model name
        instance_type (str): Instance type
        role_arn (str): IAM role ARN
        model_data_url (Optional[str]): S3 URL to model data
        environment (Optional[Dict[str, str]]): Environment variables
        region (Optional[str]): AWS region
        
    Returns:
        Dict[str, Any]: Deployment result
    """
    try:
        region = region or get_aws_region()
        sm = get_sagemaker_client(region)
        
        # Create model
        model_params = {
            "ModelName": model_name,
            "PrimaryContainer": {
                "Image": image_uri,
                "Environment": environment or {}
            },
            "ExecutionRoleArn": role_arn
        }
        
        if model_data_url:
            model_params["PrimaryContainer"]["ModelDataUrl"] = model_data_url
        
        sm.create_model(**model_params)
        
        # Create endpoint configuration
        endpoint_config_name = f"{model_name}-config"
        sm.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "VariantName": "default",
                    "ModelName": model_name,
                    "InstanceType": instance_type,
                    "InitialInstanceCount": 1
                }
            ]
        )
        
        # Create endpoint
        sm.create_endpoint(
            EndpointName=model_name,
            EndpointConfigName=endpoint_config_name
        )
        
        return {
            "success": True,
            "endpoint_name": model_name,
            "status": "Creating",
            "message": f"SageMaker endpoint {model_name} is being created. Use get_sagemaker_endpoint_status to check status."
        }
    except Exception as e:
        logger.error(f"Failed to deploy to SageMaker: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def get_sagemaker_endpoint_status(
    endpoint_name: str,
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get the status of a SageMaker endpoint.
    
    Args:
        endpoint_name (str): Endpoint name
        region (Optional[str]): AWS region
        
    Returns:
        Dict[str, Any]: Endpoint status
    """
    try:
        region = region or get_aws_region()
        sm = get_sagemaker_client(region)
        
        response = sm.describe_endpoint(EndpointName=endpoint_name)
        
        return {
            "success": True,
            "endpoint_name": endpoint_name,
            "status": response["EndpointStatus"],
            "endpoint_url": f"https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint_name}/invocations",
            "created_at": response["CreationTime"].isoformat(),
            "last_modified": response["LastModifiedTime"].isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get SageMaker endpoint status: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def deploy_to_ecs(
    image_uri: str,
    task_name: str,
    cluster_name: str,
    cpu: str,
    memory: str,
    container_port: int = 8080,
    host_port: int = 8080,
    environment: Optional[Dict[str, str]] = None,
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Deploy a DLC image to Amazon ECS.
    
    Args:
        image_uri (str): DLC image URI
        task_name (str): Task name
        cluster_name (str): ECS cluster name
        cpu (str): CPU units
        memory (str): Memory limit
        container_port (int): Container port
        host_port (int): Host port
        environment (Optional[Dict[str, str]]): Environment variables
        region (Optional[str]): AWS region
        
    Returns:
        Dict[str, Any]: Deployment result
    """
    try:
        region = region or get_aws_region()
        ecs = get_ecs_client(region)
        
        # Convert environment dict to ECS format
        env_vars = []
        if environment:
            for key, value in environment.items():
                env_vars.append({
                    "name": key,
                    "value": value
                })
        
        # Register task definition
        response = ecs.register_task_definition(
            family=task_name,
            networkMode="awsvpc",
            requiresCompatibilities=["FARGATE"],
            cpu=cpu,
            memory=memory,
            containerDefinitions=[
                {
                    "name": task_name,
                    "image": image_uri,
                    "essential": True,
                    "portMappings": [
                        {
                            "containerPort": container_port,
                            "hostPort": host_port,
                            "protocol": "tcp"
                        }
                    ],
                    "environment": env_vars
                }
            ]
        )
        
        task_definition_arn = response["taskDefinition"]["taskDefinitionArn"]
        
        # Create service
        service_response = ecs.create_service(
            cluster=cluster_name,
            serviceName=task_name,
            taskDefinition=task_definition_arn,
            desiredCount=1,
            launchType="FARGATE",
            networkConfiguration={
                "awsvpcConfiguration": {
                    "assignPublicIp": "ENABLED",
                    "subnets": ["subnet-12345678"],  # This should be parameterized
                    "securityGroups": ["sg-12345678"]  # This should be parameterized
                }
            }
        )
        
        return {
            "success": True,
            "service_name": task_name,
            "task_definition_arn": task_definition_arn,
            "status": "Creating",
            "message": f"ECS service {task_name} is being created. Use get_ecs_service_status to check status."
        }
    except Exception as e:
        logger.error(f"Failed to deploy to ECS: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def deploy_to_ec2(
    image_uri: str,
    instance_type: str,
    key_name: Optional[str] = None,
    security_group_ids: Optional[List[str]] = None,
    subnet_id: Optional[str] = None,
    user_data: Optional[str] = None,
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Deploy a DLC image to Amazon EC2.
    
    Args:
        image_uri (str): DLC image URI
        instance_type (str): EC2 instance type
        key_name (Optional[str]): EC2 key pair name
        security_group_ids (Optional[List[str]]): Security group IDs
        subnet_id (Optional[str]): Subnet ID
        user_data (Optional[str]): User data script
        region (Optional[str]): AWS region
        
    Returns:
        Dict[str, Any]: Deployment result
    """
    try:
        region = region or get_aws_region()
        ec2 = get_ec2_client(region)
        
        # Generate user data script if not provided
        if not user_data:
            user_data = f"""#!/bin/bash
# Install Docker
apt-get update
apt-get install -y docker.io
systemctl start docker
systemctl enable docker

# Configure AWS CLI
aws configure set region {region}

# Login to ECR
aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin $(echo {image_uri} | cut -d/ -f1)

# Pull and run the DLC image
docker pull {image_uri}
docker run -d -p 8080:8080 {image_uri}
"""
        
        # Launch EC2 instance
        response = ec2.run_instances(
            ImageId="ami-12345678",  # This should be parameterized with a Deep Learning AMI
            InstanceType=instance_type,
            MinCount=1,
            MaxCount=1,
            KeyName=key_name,
            SecurityGroupIds=security_group_ids or [],
            SubnetId=subnet_id,
            UserData=user_data,
            TagSpecifications=[
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {
                            "Key": "Name",
                            "Value": "DLC-Instance"
                        },
                        {
                            "Key": "DLCImage",
                            "Value": image_uri
                        }
                    ]
                }
            ]
        )
        
        instance_id = response["Instances"][0]["InstanceId"]
        
        return {
            "success": True,
            "instance_id": instance_id,
            "status": "pending",
            "message": f"EC2 instance {instance_id} is being launched. Use get_ec2_instance_status to check status."
        }
    except Exception as e:
        logger.error(f"Failed to deploy to EC2: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def deploy_to_eks(
    image_uri: str,
    cluster_name: str,
    deployment_name: str,
    namespace: str = "default",
    replicas: int = 1,
    container_port: int = 8080,
    cpu_request: str = "1",
    memory_request: str = "2Gi",
    environment: Optional[Dict[str, str]] = None,
    region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Deploy a DLC image to Amazon EKS.
    
    Args:
        image_uri (str): DLC image URI
        cluster_name (str): EKS cluster name
        deployment_name (str): Kubernetes deployment name
        namespace (str): Kubernetes namespace
        replicas (int): Number of replicas
        container_port (int): Container port
        cpu_request (str): CPU request
        memory_request (str): Memory request
        environment (Optional[Dict[str, str]]): Environment variables
        region (Optional[str]): AWS region
        
    Returns:
        Dict[str, Any]: Deployment result
    """
    try:
        region = region or get_aws_region()
        eks = get_eks_client(region)
        
        # Get cluster info
        cluster = eks.describe_cluster(name=cluster_name)
        
        # Convert environment dict to Kubernetes format
        env_vars = []
        if environment:
            for key, value in environment.items():
                env_vars.append({
                    "name": key,
                    "value": value
                })
        
        # Create Kubernetes deployment manifest
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": deployment_name,
                "namespace": namespace
            },
            "spec": {
                "replicas": replicas,
                "selector": {
                    "matchLabels": {
                        "app": deployment_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": deployment_name
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": deployment_name,
                                "image": image_uri,
                                "ports": [
                                    {
                                        "containerPort": container_port
                                    }
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": cpu_request,
                                        "memory": memory_request
                                    }
                                },
                                "env": env_vars
                            }
                        ]
                    }
                }
            }
        }
        
        # Create service manifest
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": deployment_name,
                "namespace": namespace
            },
            "spec": {
                "selector": {
                    "app": deployment_name
                },
                "ports": [
                    {
                        "port": container_port,
                        "targetPort": container_port
                    }
                ],
                "type": "LoadBalancer"
            }
        }
        
        # In a real implementation, you would use the Kubernetes Python client
        # to apply these manifests. For this example, we'll just return them.
        
        return {
            "success": True,
            "cluster_name": cluster_name,
            "deployment_name": deployment_name,
            "namespace": namespace,
            "deployment_manifest": json.dumps(deployment, indent=2),
            "service_manifest": json.dumps(service, indent=2),
            "message": "Use kubectl to apply these manifests to your EKS cluster."
        }
    except Exception as e:
        logger.error(f"Failed to deploy to EKS: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def register_module(mcp: FastMCP) -> None:
    """
    Register the deployment module with the MCP server.
    
    Args:
        mcp (FastMCP): MCP server instance
    """
    mcp.add_tool(
        name="deploy_to_sagemaker",
        description="Deploy a DLC image to Amazon SageMaker.\n\nArgs:\n    image_uri: DLC image URI\n    model_name: Model name\n    instance_type: Instance type\n    role_arn: IAM role ARN\n    model_data_url: Optional S3 URL to model data\n    environment: Optional environment variables\n    region: Optional AWS region\n",
        function=deploy_to_sagemaker
    )
    
    mcp.add_tool(
        name="get_sagemaker_endpoint_status",
        description="Get the status of a SageMaker endpoint.\n\nArgs:\n    endpoint_name: Endpoint name\n    region: Optional AWS region\n",
        function=get_sagemaker_endpoint_status
    )
    
    mcp.add_tool(
        name="deploy_to_ecs",
        description="Deploy a DLC image to Amazon ECS.\n\nArgs:\n    image_uri: DLC image URI\n    task_name: Task name\n    cluster_name: ECS cluster name\n    cpu: CPU units\n    memory: Memory limit\n    container_port: Container port\n    host_port: Host port\n    environment: Optional environment variables\n    region: Optional AWS region\n",
        function=deploy_to_ecs
    )
    
    mcp.add_tool(
        name="deploy_to_ec2",
        description="Deploy a DLC image to Amazon EC2.\n\nArgs:\n    image_uri: DLC image URI\n    instance_type: EC2 instance type\n    key_name: Optional EC2 key pair name\n    security_group_ids: Optional security group IDs\n    subnet_id: Optional subnet ID\n    user_data: Optional user data script\n    region: Optional AWS region\n",
        function=deploy_to_ec2
    )
    
    mcp.add_tool(
        name="deploy_to_eks",
        description="Deploy a DLC image to Amazon EKS.\n\nArgs:\n    image_uri: DLC image URI\n    cluster_name: EKS cluster name\n    deployment_name: Kubernetes deployment name\n    namespace: Kubernetes namespace\n    replicas: Number of replicas\n    container_port: Container port\n    cpu_request: CPU request\n    memory_request: Memory request\n    environment: Optional environment variables\n    region: Optional AWS region\n",
        function=deploy_to_eks
    )
