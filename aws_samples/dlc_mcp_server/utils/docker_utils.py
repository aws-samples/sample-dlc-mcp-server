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

"""Docker utilities for the DLC MCP Server."""

import logging
import docker
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)


def get_docker_client() -> docker.DockerClient:
    """
    Get a Docker client.
    
    Returns:
        docker.DockerClient: Docker client
    """
    try:
        return docker.from_env()
    except Exception as e:
        logger.error(f"Failed to create Docker client: {e}")
        raise


def pull_image(image_uri: str) -> Dict[str, Any]:
    """
    Pull a Docker image.
    
    Args:
        image_uri (str): Image URI
        
    Returns:
        Dict[str, Any]: Result of the operation
    """
    try:
        client = get_docker_client()
        logger.info(f"Pulling image: {image_uri}")
        image = client.images.pull(image_uri)
        
        return {
            "success": True,
            "image_id": image.id,
            "tags": image.tags
        }
    except Exception as e:
        logger.error(f"Failed to pull image {image_uri}: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def build_image(
    dockerfile_path: str,
    tag: str,
    build_args: Optional[Dict[str, str]] = None,
    context_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build a Docker image.
    
    Args:
        dockerfile_path (str): Path to Dockerfile
        tag (str): Image tag
        build_args (Optional[Dict[str, str]]): Build arguments
        context_path (Optional[str]): Build context path
        
    Returns:
        Dict[str, Any]: Result of the operation
    """
    try:
        client = get_docker_client()
        context = context_path or "."
        logger.info(f"Building image {tag} from {dockerfile_path}")
        
        # Build the image
        image, logs = client.images.build(
            path=context,
            dockerfile=dockerfile_path,
            tag=tag,
            buildargs=build_args or {},
            rm=True
        )
        
        # Process build logs
        build_logs = []
        for log in logs:
            if "stream" in log:
                log_line = log["stream"].strip()
                if log_line:
                    build_logs.append(log_line)
        
        return {
            "success": True,
            "image_id": image.id,
            "tag": tag,
            "logs": build_logs
        }
    except Exception as e:
        logger.error(f"Failed to build image {tag}: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def push_image(image_uri: str) -> Dict[str, Any]:
    """
    Push a Docker image to a registry.
    
    Args:
        image_uri (str): Image URI
        
    Returns:
        Dict[str, Any]: Result of the operation
    """
    try:
        client = get_docker_client()
        logger.info(f"Pushing image: {image_uri}")
        
        # Push the image
        push_logs = []
        for line in client.images.push(image_uri, stream=True, decode=True):
            if "status" in line:
                push_logs.append(f"{line['status']}")
            if "error" in line:
                raise Exception(line["error"])
        
        return {
            "success": True,
            "image_uri": image_uri,
            "logs": push_logs
        }
    except Exception as e:
        logger.error(f"Failed to push image {image_uri}: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def list_local_dlc_images() -> List[Dict[str, Any]]:
    """
    List local DLC images.
    
    Returns:
        List[Dict[str, Any]]: List of DLC images
    """
    try:
        client = get_docker_client()
        images = client.images.list()
        
        dlc_images = []
        for image in images:
            for tag in image.tags:
                if "dkr.ecr" in tag and any(fw in tag for fw in ["pytorch", "tensorflow", "mxnet", "autogluon"]):
                    dlc_images.append({
                        "id": image.id,
                        "tag": tag,
                        "created": image.attrs["Created"],
                        "size": image.attrs["Size"]
                    })
        
        return dlc_images
    except Exception as e:
        logger.error(f"Failed to list DLC images: {e}")
        return []
