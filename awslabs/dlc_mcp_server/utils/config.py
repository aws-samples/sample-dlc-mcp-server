"""Configuration utilities for the DLC MCP Server."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def get_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables and config file.

    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    # Default configuration
    config = {
        "allow-write": False,
        "allow-sensitive-data": False,
    }

    # Load from environment variables
    if os.environ.get("ALLOW_WRITE", "").lower() == "true":
        config["allow-write"] = True

    if os.environ.get("ALLOW_SENSITIVE_DATA", "").lower() == "true":
        config["allow-sensitive-data"] = True

    # Load from config file if it exists
    config_file = os.environ.get("DLC_MCP_CONFIG", "~/.dlc-mcp/config.yaml")
    config_path = Path(os.path.expanduser(config_file))

    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f)
                if file_config and isinstance(file_config, dict):
                    config.update(file_config)
        except Exception as e:
            print(f"Error loading config file: {e}")

    return config


def get_aws_region() -> str:
    """
    Get the AWS region from environment variables or config.

    Returns:
        str: AWS region
    """
    return os.environ.get("AWS_REGION", "us-east-1")


def get_dlc_registry(region: Optional[str] = None) -> str:
    """
    Get the DLC registry URI for the specified region.

    Args:
        region (Optional[str]): AWS region

    Returns:
        str: DLC registry URI
    """
    if not region:
        region = get_aws_region()

    return f"763104351884.dkr.ecr.{region}.amazonaws.com"
