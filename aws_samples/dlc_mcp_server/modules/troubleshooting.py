###
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
######

"""Module for troubleshooting DLC-related issues with intelligent error analysis."""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

DLC_RESOLVER_GROUPS = [
    "DLC Customer Issues",
    "DLC General",
    "Inference DLC",
    "InferenceDLC-AutoCut",
    "Asimov DLC Autocut",
    "Asimov DLAMI Autocut",
    "Marin",
    "deepengines-conda-dev",
]


class ErrorCategory(Enum):
    CUDA_OOM = "cuda out of memory"
    CUDA_VERSION = "cuda version mismatch"
    IMPORT_ERROR = "import error module not found"
    PERMISSION = "permission denied"
    DOCKER = "docker daemon issue"
    ECR_AUTH = "ecr authentication"
    SHAPE_MISMATCH = "tensor shape mismatch"
    DTYPE_MISMATCH = "data type mismatch"
    DISTRIBUTED = "distributed training nccl"
    MODEL_LOADING = "model loading failure"
    TIMEOUT = "inference timeout"
    MEMORY_OOM = "system memory exhaustion"
    UNKNOWN = "unknown error"


@dataclass
class ExtractedContext:
    gpu_memory_total: Optional[float] = None
    gpu_memory_used: Optional[float] = None
    batch_size: Optional[int] = None
    framework: Optional[str] = None
    cuda_version: Optional[str] = None
    instance_type: Optional[str] = None
    key_error_message: str = ""
    stack_trace_summary: List[str] = field(default_factory=list)


def _extract_context(error_log: str) -> ExtractedContext:
    context = ExtractedContext()
    gpu_match = re.search(
        r"(\d+\.?\d*)\s*GiB total.*?(\d+\.?\d*)\s*GiB already", error_log, re.I | re.DOTALL
    )
    if gpu_match:
        context.gpu_memory_total = float(gpu_match.group(1))
        context.gpu_memory_used = float(gpu_match.group(2))
    batch_match = re.search(r"batch[_\s]?size\s*[=:]\s*(\d+)", error_log, re.I)
    if batch_match:
        context.batch_size = int(batch_match.group(1))
    instance_match = re.search(r"ml\.[a-z0-9]+\.\d*x?large", error_log, re.I)
    if instance_match:
        context.instance_type = instance_match.group(0)
    if "torch" in error_log.lower():
        context.framework = "pytorch"
    elif "tensorflow" in error_log.lower():
        context.framework = "tensorflow"
    cuda_match = re.search(r"cuda[:\s]+(\d+\.\d+)", error_log, re.I)
    if cuda_match:
        context.cuda_version = cuda_match.group(1)
    error_lines = [l.strip() for l in error_log.split("\n") if "error" in l.lower()]
    if error_lines:
        context.key_error_message = error_lines[0][:200]
    return context


def _categorize_error(error_log: str) -> List[ErrorCategory]:
    categories = []
    el = error_log.lower()
    patterns = {
        ErrorCategory.CUDA_OOM: [r"cuda.*out of memory"],
        ErrorCategory.MEMORY_OOM: [r"out of memory", r"more memory", r"oom"],
        ErrorCategory.CUDA_VERSION: [r"cuda.*version", r"incompatible.*cuda"],
        ErrorCategory.IMPORT_ERROR: [r"modulenotfounderror", r"no module named"],
        ErrorCategory.PERMISSION: [r"permission denied"],
        ErrorCategory.DOCKER: [r"docker daemon"],
        ErrorCategory.ECR_AUTH: [r"failed to pull", r"ecr.*auth"],
        ErrorCategory.SHAPE_MISMATCH: [r"shape mismatch", r"size mismatch"],
        ErrorCategory.DISTRIBUTED: [r"nccl", r"rank.*failed"],
        ErrorCategory.MODEL_LOADING: [r"model.*not found"],
        ErrorCategory.TIMEOUT: [r"timeout", r"timed out"],
    }
    for cat, pats in patterns.items():
        for p in pats:
            if re.search(p, el):
                categories.append(cat)
                break
    return categories if categories else [ErrorCategory.UNKNOWN]


def _build_search_query(error_log: str, categories: List[ErrorCategory]) -> str:
    terms = [c.value for c in categories]
    codes = re.findall(r"error[:\s]+(\w+)", error_log, re.I)
    terms.extend(codes[:2])
    return " ".join(terms[:5])


def _generate_resolution(
    error_log: str, categories: List[ErrorCategory], context: ExtractedContext
) -> Dict[str, Any]:
    resolution = {
        "summary": "",
        "steps": [],
        "additional_resources": [],
        "confidence": "medium",
        "root_cause_analysis": "",
    }

    for cat in categories:
        if cat in (ErrorCategory.MEMORY_OOM, ErrorCategory.CUDA_OOM):
            resolution["summary"] = "Memory exhaustion detected"
            resolution[
                "root_cause_analysis"
            ] = """This error typically occurs when:
1. Data size has grown beyond what the instance can handle
2. Code changes introduced memory leaks or inefficient patterns
3. Concurrent processes consuming more memory
4. Framework/library updates changed memory allocation

For SageMaker Processing Jobs:
- Entire dataset may be loaded into memory at once
- Pandas operations can create multiple copies of data
- String operations and type conversions are memory-intensive"""
            inst = f" (current: {context.instance_type})" if context.instance_type else ""
            resolution["steps"] = [
                "IMMEDIATE: Process data in smaller chunks instead of loading all at once",
                "Check if recent data has grown in size compared to successful runs",
                "Review code for memory-inefficient patterns (df.copy(), string concat in loops)",
                "Use memory profiling: memory_profiler or tracemalloc",
                "For Pandas: use dtype optimization (category for strings, downcast numerics)",
                "Consider streaming/chunked processing: pd.read_csv(..., chunksize=N)",
                f"If data truly requires more memory, try memory-optimized instances{inst}",
                "Check CloudWatch metrics to compare memory usage between jobs",
            ]
            resolution["confidence"] = "high"
            resolution["additional_resources"] = [
                "https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html",
                "https://pandas.pydata.org/docs/user_guide/scale.html",
            ]
            break
        elif cat == ErrorCategory.IMPORT_ERROR:
            resolution["summary"] = "Missing Python package"
            resolution["steps"] = [
                "pip list | grep <package>",
                "pip install <package>",
                "Check Python environment",
            ]
            resolution["confidence"] = "high"
            break
        elif cat == ErrorCategory.CUDA_VERSION:
            resolution["summary"] = "CUDA version mismatch"
            resolution["steps"] = ["Check nvcc --version", "Use matching DLC image"]
            break
        elif cat == ErrorCategory.ECR_AUTH:
            resolution["summary"] = "ECR authentication failure"
            resolution["steps"] = [
                "aws ecr get-login-password | docker login",
                "Check IAM permissions",
            ]
            resolution["confidence"] = "high"
            break
        elif cat == ErrorCategory.UNKNOWN:
            resolution["summary"] = "Requires investigation"
            resolution["steps"] = [
                "Review full stack trace",
                "Check DLC docs",
                "Compare with successful runs",
            ]
            resolution["confidence"] = "low"

    resolution["additional_resources"].append("https://aws.github.io/deep-learning-containers/")
    return resolution


async def analyze_error(
    error_log: str, framework: Optional[str] = None, use_case: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze DLC/SageMaker error logs and provide intelligent resolution."""
    context = _extract_context(error_log)
    if framework:
        context.framework = framework
    categories = _categorize_error(error_log)
    search_query = _build_search_query(error_log, categories)
    resolution = _generate_resolution(error_log, categories, context)

    return {
        "success": True,
        "error_categories": [c.value for c in categories],
        "extracted_context": {
            "framework": context.framework,
            "cuda_version": context.cuda_version,
            "batch_size": context.batch_size,
            "instance_type": context.instance_type,
            "gpu_memory_total": context.gpu_memory_total,
            "gpu_memory_used": context.gpu_memory_used,
        },
        "search_query": search_query,
        "resolution": resolution,
        "resolver_groups": DLC_RESOLVER_GROUPS,
    }


def diagnose_common_issues(
    error_message: str, framework: Optional[str] = None, use_case: Optional[str] = None
) -> Dict[str, Any]:
    """Diagnose common DLC-related issues."""
    patterns = [
        {
            "pattern": r"(?i)cuda.+out of memory",
            "diagnosis": "CUDA OOM",
            "solution": ["Reduce batch size", "Use mixed precision"],
        },
        {
            "pattern": r"(?i)(out of memory|more memory|oom)",
            "diagnosis": "Memory exhaustion",
            "solution": ["Process in chunks", "Use larger instance", "Optimize dtypes"],
        },
        {
            "pattern": r"(?i)no module named",
            "diagnosis": "Missing package",
            "solution": ["pip install <package>", "Check DLC image"],
        },
        {
            "pattern": r"(?i)permission denied",
            "diagnosis": "Permission issue",
            "solution": ["Check permissions"],
        },
        {
            "pattern": r"(?i)failed to pull",
            "diagnosis": "Image pull failure",
            "solution": ["Authenticate with ECR"],
        },
    ]
    matches = [
        {"diagnosis": p["diagnosis"], "solution": p["solution"]}
        for p in patterns
        if re.search(p["pattern"], error_message)
    ]
    if not matches:
        return {
            "matched": False,
            "diagnosis": "Unknown",
            "general_recommendations": ["Check logs", "Verify config"],
        }
    return {"matched": True, "matches": matches}


def get_framework_compatibility_info(framework: str, version: str) -> Dict[str, Any]:
    """Get compatibility information for a specific framework version."""
    info = {
        "pytorch": {
            "2.6.0": {"python_versions": ["3.12"], "cuda_versions": ["12.6", "12.4"]},
            "2.0.0": {"python_versions": ["3.10"], "cuda_versions": ["11.8", "11.7"]},
        },
        "tensorflow": {
            "2.18.0": {"python_versions": ["3.10"], "cuda_versions": ["12.5", "12.2"]},
            "2.12.0": {"python_versions": ["3.9", "3.10"], "cuda_versions": ["11.8"]},
        },
    }
    fw = framework.lower()
    if fw not in info:
        return {"success": False, "error": f"Framework '{framework}' not found"}
    if version not in info[fw]:
        return {
            "success": False,
            "error": f"Version '{version}' not found",
            "available_versions": list(info[fw].keys()),
        }
    return {
        "success": True,
        "framework": fw,
        "version": version,
        "compatibility": info[fw][version],
    }


def register_module(mcp: FastMCP) -> None:
    """Register the troubleshooting module with the MCP server.

    Note: get_performance_optimization_tips removed - use get_framework_specific_best_practices
    from best_practices module instead (more comprehensive).
    """
    mcp.add_tool(
        name="analyze_dlc_error",
        description="Analyze DLC/SageMaker error logs and provide intelligent resolution with root cause analysis.",
        fn=analyze_error,
    )
    mcp.add_tool(
        name="diagnose_common_issues",
        description="Diagnose common DLC-related issues.",
        fn=diagnose_common_issues,
    )
    mcp.add_tool(
        name="get_framework_compatibility_info",
        description="Get compatibility information for a specific framework version.",
        fn=get_framework_compatibility_info,
    )
