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

"""Security utilities for the DLC MCP Server."""

import functools
import logging
from typing import Dict, Any, Callable, TypeVar, cast

# Permission constants
PERMISSION_WRITE = "write"
PERMISSION_SENSITIVE_DATA = "sensitive-data"

# Type variables for function signatures
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def secure_tool(config: Dict[str, Any], permission: str, tool_name: str) -> Callable[[F], F]:
    """
    Decorator to secure tools based on permissions.
    
    Args:
        config (Dict[str, Any]): Server configuration
        permission (str): Required permission
        tool_name (str): Name of the tool being secured
        
    Returns:
        Callable: Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            permission_key = f"allow-{permission}"
            
            if permission_key not in config or not config[permission_key]:
                logger.warning(f"Access denied: {tool_name} requires {permission} permission")
                return {
                    "error": f"Access denied: This operation requires {permission} permission. "
                             f"Set ALLOW_{permission.upper().replace('-', '_')}=true to enable."
                }
            
            return func(*args, **kwargs)
        
        return cast(F, wrapper)
    
    return decorator


def validate_aws_credentials() -> Dict[str, Any]:
    """
    Validate AWS credentials.
    
    Returns:
        Dict[str, Any]: Validation result
    """
    import boto3
    
    try:
        # Try to get caller identity to validate credentials
        sts = boto3.client("sts")
        identity = sts.get_caller_identity()
        
        return {
            "valid": True,
            "account_id": identity["Account"],
            "user_id": identity["UserId"],
            "arn": identity["Arn"]
        }
    except Exception as e:
        logger.error(f"AWS credentials validation failed: {e}")
        return {
            "valid": False,
            "error": str(e)
        }
