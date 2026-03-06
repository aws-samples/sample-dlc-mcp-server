###
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
######

"""
DLC Discovery Module - Tools for discovering and selecting DLC images.
Provides intelligent recommendations and comprehensive image information.
"""

import logging
from typing import Dict, Any, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from aws_samples.dlc_mcp_server.utils.dlc_images import (
    filter_images,
    get_latest_image,
    get_available_frameworks,
    get_available_versions,
    get_recommended_image_for_model,
    get_ecr_account_for_region,
    is_neuron_supported_in_region,
    get_dlc_images,
    refresh_images,
)

logger = logging.getLogger(__name__)


def search_dlc_images(
    framework: Optional[str] = None,
    version: Optional[str] = None,
    python_version: Optional[str] = None,
    accelerator: Optional[str] = None,
    platform: Optional[str] = None,
    use_case: Optional[str] = None,
    architecture: Optional[str] = None,
    region: str = "us-west-2",
) -> Dict[str, Any]:
    """
    Search for DLC images with flexible filtering.

    Args:
        framework: Filter by framework (pytorch, tensorflow, vllm, etc.)
        version: Filter by version (e.g., "2.9" matches "2.9.0")
        python_version: Filter by Python version (e.g., "3.12")
        accelerator: Filter by accelerator (gpu, cpu, neuronx)
        platform: Filter by platform (sagemaker, ec2)
        use_case: Filter by use case (training, inference)
        architecture: Filter by architecture (x86_64, arm64)
        region: AWS region for generating URIs

    Returns:
        Dict with matching images and their full URIs
    """
    try:
        images = filter_images(
            framework=framework,
            version=version,
            python_version=python_version,
            accelerator=accelerator,
            platform=platform,
            use_case=use_case,
            architecture=architecture,
        )

        results = []
        for img in images:
            result = img.to_dict()
            result["image_uri"] = img.get_full_uri(region)
            results.append(result)

        return {
            "success": True,
            "total_matches": len(results),
            "region": region,
            "filters_applied": {
                "framework": framework,
                "version": version,
                "python_version": python_version,
                "accelerator": accelerator,
                "platform": platform,
                "use_case": use_case,
                "architecture": architecture,
            },
            "images": results,
        }
    except Exception as e:
        logger.error(f"Error searching DLC images: {e}")
        return {"success": False, "error": str(e)}


def get_image_recommendation(
    model_type: str,
    model_size: str = "medium",
    use_case: str = "inference",
    region: str = "us-west-2",
) -> Dict[str, Any]:
    """
    Get intelligent DLC image recommendation based on workload.

    Args:
        model_type: Type of ML model (llm, vision, nlp, tabular, diffusion)
        model_size: Size category (small, medium, large, xlarge)
        use_case: training or inference
        region: AWS region

    Returns:
        Recommended image with explanation
    """
    try:
        image = get_recommended_image_for_model(model_type, model_size, use_case)

        if not image:
            return {
                "success": False,
                "error": f"No recommendation found for model_type={model_type}, size={model_size}",
                "available_model_types": ["llm", "vision", "nlp", "tabular", "diffusion"],
                "available_sizes": ["small", "medium", "large", "xlarge"],
            }

        explanations = {
            "llm": {
                "small": "vLLM provides excellent performance for small-medium LLMs with optimized inference.",
                "medium": "vLLM with GPU acceleration handles medium-sized LLMs efficiently.",
                "large": "DJL with LMI engine is optimized for large model inference with tensor parallelism.",
                "xlarge": "NeuronX on AWS Trainium/Inferentia provides cost-effective inference for very large models.",
            },
            "vision": {
                "default": "PyTorch provides comprehensive support for vision models with CUDA optimization.",
            },
            "nlp": {
                "small": "HuggingFace containers include optimized transformers for NLP tasks.",
                "medium": "HuggingFace with GPU provides fast inference for transformer models.",
                "large": "vLLM handles large NLP models with efficient batching and memory management.",
            },
            "tabular": {
                "default": "AutoGluon provides automated ML for tabular data with minimal configuration.",
            },
            "diffusion": {
                "default": "StabilityAI containers are optimized for diffusion model inference.",
            },
        }

        explanation = explanations.get(model_type, {}).get(
            model_size,
            explanations.get(model_type, {}).get(
                "default", "Recommended based on workload characteristics."
            ),
        )

        return {
            "success": True,
            "recommendation": {
                "image": image.to_dict(),
                "image_uri": image.get_full_uri(region),
                "explanation": explanation,
            },
            "input": {
                "model_type": model_type,
                "model_size": model_size,
                "use_case": use_case,
                "region": region,
            },
        }
    except Exception as e:
        logger.error(f"Error getting recommendation: {e}")
        return {"success": False, "error": str(e)}


def list_frameworks() -> Dict[str, Any]:
    """
    List all available DLC frameworks with their latest versions.

    Returns:
        Dict with frameworks and version information
    """
    try:
        frameworks = get_available_frameworks()
        framework_info = []

        for fw in frameworks:
            versions = get_available_versions(fw)
            images = filter_images(framework=fw)
            use_cases = sorted(set(img.use_case for img in images))
            accelerators = sorted(set(img.accelerator for img in images))

            framework_info.append(
                {
                    "framework": fw,
                    "latest_version": versions[0] if versions else None,
                    "available_versions": versions[:5],  # Top 5 versions
                    "use_cases": use_cases,
                    "accelerators": accelerators,
                    "total_images": len(images),
                }
            )

        return {
            "success": True,
            "total_frameworks": len(frameworks),
            "frameworks": framework_info,
        }
    except Exception as e:
        logger.error(f"Error listing frameworks: {e}")
        return {"success": False, "error": str(e)}


def get_region_info(region: str) -> Dict[str, Any]:
    """
    Get DLC availability information for a specific AWS region.

    Args:
        region: AWS region code (e.g., us-west-2)

    Returns:
        Region-specific DLC information
    """
    try:
        account_id = get_ecr_account_for_region(region)
        neuron_supported = is_neuron_supported_in_region(region)

        is_china = region.startswith("cn-")
        ecr_suffix = ".amazonaws.com.cn" if is_china else ".amazonaws.com"

        return {
            "success": True,
            "region": region,
            "ecr_account_id": account_id,
            "ecr_registry": f"{account_id}.dkr.ecr.{region}{ecr_suffix}",
            "neuron_supported": neuron_supported,
            "is_china_region": is_china,
            "ecr_login_command": f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}{ecr_suffix}",
        }
    except Exception as e:
        logger.error(f"Error getting region info: {e}")
        return {"success": False, "error": str(e)}


def compare_images(image_uris: List[str]) -> Dict[str, Any]:
    """
    Compare multiple DLC images to help with selection.

    Args:
        image_uris: List of image URIs or framework:version strings to compare

    Returns:
        Comparison table of image features
    """
    try:
        comparisons = []
        all_images = get_dlc_images()

        for uri in image_uris:
            # Try to find matching image in catalog
            matching = None
            for img in all_images:
                if img.framework in uri and img.version in uri:
                    matching = img
                    break

            if matching:
                comparisons.append(
                    {
                        "input": uri,
                        "framework": matching.framework,
                        "version": matching.version,
                        "python_version": matching.python_version,
                        "accelerator": matching.accelerator,
                        "cuda_version": matching.cuda_version,
                        "platform": matching.platform,
                        "use_case": matching.use_case,
                        "architecture": matching.architecture,
                        "os_version": matching.os_version,
                    }
                )
            else:
                comparisons.append(
                    {
                        "input": uri,
                        "error": "Image not found in catalog",
                    }
                )

        return {
            "success": True,
            "comparison_count": len(comparisons),
            "comparisons": comparisons,
        }
    except Exception as e:
        logger.error(f"Error comparing images: {e}")
        return {"success": False, "error": str(e)}


def get_llm_serving_options(
    model_name: Optional[str] = None,
    max_model_size_gb: Optional[int] = None,
    target_latency: str = "balanced",
    region: str = "us-west-2",
) -> Dict[str, Any]:
    """
    Get LLM serving container options with recommendations.

    Args:
        model_name: Optional model name (e.g., "llama-3-70b", "mistral-7b")
        max_model_size_gb: Maximum model size in GB
        target_latency: Latency target (low, balanced, throughput)
        region: AWS region

    Returns:
        LLM serving options with pros/cons
    """
    try:
        options = []

        # vLLM option
        vllm_image = get_latest_image("vllm", "inference", "gpu", "sagemaker")
        if vllm_image:
            options.append(
                {
                    "container": "vLLM",
                    "image_uri": vllm_image.get_full_uri(region),
                    "version": vllm_image.version,
                    "best_for": "General LLM inference with PagedAttention",
                    "pros": [
                        "Excellent memory efficiency with PagedAttention",
                        "High throughput with continuous batching",
                        "Wide model support (Llama, Mistral, etc.)",
                        "OpenAI-compatible API",
                    ],
                    "cons": [
                        "GPU-only (no CPU support)",
                        "May require tuning for very large models",
                    ],
                    "recommended_for": ["llama", "mistral", "qwen", "phi"],
                }
            )

        # SGLang option
        sglang_image = get_latest_image("sglang", "inference", "gpu", "sagemaker")
        if sglang_image:
            options.append(
                {
                    "container": "SGLang",
                    "image_uri": sglang_image.get_full_uri(region),
                    "version": sglang_image.version,
                    "best_for": "Structured generation and complex prompting",
                    "pros": [
                        "RadixAttention for efficient prefix caching",
                        "Excellent for structured outputs (JSON, code)",
                        "Compressed FSM for constrained decoding",
                        "Multi-modal support",
                    ],
                    "cons": [
                        "Newer project, smaller community",
                        "SageMaker-only currently",
                    ],
                    "recommended_for": ["structured-output", "json-generation", "code-generation"],
                }
            )

        # DJL LMI option
        djl_image = get_latest_image("djl-inference", "inference", "gpu", "sagemaker")
        if djl_image:
            options.append(
                {
                    "container": "DJL LMI",
                    "image_uri": djl_image.get_full_uri(region),
                    "version": djl_image.version,
                    "best_for": "Large models with tensor parallelism",
                    "pros": [
                        "Built-in tensor parallelism for multi-GPU",
                        "Supports very large models (70B+)",
                        "Multiple backend options (vLLM, TensorRT-LLM)",
                        "SageMaker optimized",
                    ],
                    "cons": [
                        "More complex configuration",
                        "Higher resource requirements",
                    ],
                    "recommended_for": ["large-models", "multi-gpu", "production"],
                }
            )

        # NeuronX option
        neuron_image = get_latest_image("pytorch", "inference", "neuronx", "sagemaker")
        if neuron_image and is_neuron_supported_in_region(region):
            options.append(
                {
                    "container": "PyTorch NeuronX",
                    "image_uri": neuron_image.get_full_uri(region),
                    "version": neuron_image.version,
                    "best_for": "Cost-effective inference on AWS Inferentia",
                    "pros": [
                        "Lower cost than GPU instances",
                        "High throughput for compiled models",
                        "Good for consistent workloads",
                    ],
                    "cons": [
                        "Requires model compilation",
                        "Limited to supported model architectures",
                        "Less flexible than GPU options",
                    ],
                    "recommended_for": ["cost-optimization", "high-throughput", "production"],
                }
            )

        # Add recommendation based on criteria
        recommendation = None
        if target_latency == "low":
            recommendation = "vLLM or SGLang for lowest latency"
        elif target_latency == "throughput":
            recommendation = "DJL LMI for maximum throughput with large batches"
        else:
            recommendation = "vLLM for balanced performance and ease of use"

        if max_model_size_gb and max_model_size_gb > 50:
            recommendation = "DJL LMI recommended for models >50GB with tensor parallelism"

        return {
            "success": True,
            "options": options,
            "recommendation": recommendation,
            "input": {
                "model_name": model_name,
                "max_model_size_gb": max_model_size_gb,
                "target_latency": target_latency,
                "region": region,
            },
        }
    except Exception as e:
        logger.error(f"Error getting LLM serving options: {e}")
        return {"success": False, "error": str(e)}


def register_module(mcp: FastMCP) -> None:
    """Register DLC discovery tools with the MCP server."""

    @mcp.tool(
        name="search_dlc_images",
        description="Search for AWS Deep Learning Container images with flexible filtering by framework, version, accelerator, platform, and more. Returns matching images with full ECR URIs.",
    )
    async def mcp_search_dlc_images(
        framework: Optional[str] = Field(
            None,
            description="Framework filter (pytorch, tensorflow, vllm, sglang, huggingface-pytorch, autogluon, djl-inference)",
        ),
        version: Optional[str] = Field(
            None, description="Version filter (e.g., '2.9' matches '2.9.0')"
        ),
        python_version: Optional[str] = Field(None, description="Python version (e.g., '3.12')"),
        accelerator: Optional[str] = Field(
            None, description="Accelerator type: gpu, cpu, or neuronx"
        ),
        platform: Optional[str] = Field(None, description="Platform: sagemaker or ec2"),
        use_case: Optional[str] = Field(None, description="Use case: training or inference"),
        architecture: Optional[str] = Field(None, description="Architecture: x86_64 or arm64"),
        region: str = Field("us-west-2", description="AWS region for ECR URIs"),
    ) -> Dict[str, Any]:
        return search_dlc_images(
            framework,
            version,
            python_version,
            accelerator,
            platform,
            use_case,
            architecture,
            region,
        )

    @mcp.tool(
        name="get_dlc_recommendation",
        description="Get intelligent DLC image recommendation based on your ML workload type and size. Provides explanations for why each image is recommended.",
    )
    async def mcp_get_recommendation(
        model_type: str = Field(
            ..., description="Type of ML model: llm, vision, nlp, tabular, or diffusion"
        ),
        model_size: str = Field(
            "medium", description="Size category: small, medium, large, or xlarge"
        ),
        use_case: str = Field("inference", description="Use case: training or inference"),
        region: str = Field("us-west-2", description="AWS region"),
    ) -> Dict[str, Any]:
        return get_image_recommendation(model_type, model_size, use_case, region)

    @mcp.tool(
        name="list_dlc_frameworks",
        description="List all available DLC frameworks with their versions, use cases, and accelerator support. Useful for discovering what's available.",
    )
    async def mcp_list_frameworks() -> Dict[str, Any]:
        return list_frameworks()

    @mcp.tool(
        name="get_dlc_region_info",
        description="Get DLC availability information for a specific AWS region, including ECR account ID, registry URL, and Neuron support status.",
    )
    async def mcp_get_region_info(
        region: str = Field(..., description="AWS region code (e.g., us-west-2, eu-west-1)"),
    ) -> Dict[str, Any]:
        return get_region_info(region)

    @mcp.tool(
        name="compare_dlc_images",
        description="Compare multiple DLC images side-by-side to help with selection. Provide image URIs or framework:version strings.",
    )
    async def mcp_compare_images(
        image_uris: List[str] = Field(
            ..., description="List of image URIs or identifiers to compare"
        ),
    ) -> Dict[str, Any]:
        return compare_images(image_uris)

    @mcp.tool(
        name="get_llm_serving_options",
        description="Get comprehensive LLM serving container options (vLLM, SGLang, DJL, NeuronX) with pros/cons and recommendations based on your requirements.",
    )
    async def mcp_get_llm_options(
        model_name: Optional[str] = Field(
            None, description="Model name (e.g., llama-3-70b, mistral-7b)"
        ),
        max_model_size_gb: Optional[int] = Field(None, description="Maximum model size in GB"),
        target_latency: str = Field(
            "balanced", description="Latency target: low, balanced, or throughput"
        ),
        region: str = Field("us-west-2", description="AWS region"),
    ) -> Dict[str, Any]:
        return get_llm_serving_options(model_name, max_model_size_gb, target_latency, region)

    @mcp.tool(
        name="refresh_dlc_catalog",
        description="Force refresh the DLC image catalog from the AWS GitHub page. Use this to get the latest available images.",
    )
    async def mcp_refresh_catalog() -> Dict[str, Any]:
        return refresh_images()
