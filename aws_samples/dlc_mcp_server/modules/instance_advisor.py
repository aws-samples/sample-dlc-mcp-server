###
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
######

"""
Instance Advisor Module - Recommends EC2/SageMaker instance types for DLC workloads.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP
from pydantic import Field

logger = logging.getLogger(__name__)


@dataclass
class InstanceType:
    """Represents an EC2/SageMaker instance type."""

    name: str
    vcpus: int
    memory_gb: float
    gpu_count: int
    gpu_memory_gb: float
    gpu_type: str
    network_bandwidth: str
    price_per_hour_usd: float  # On-demand pricing estimate
    category: str  # training, inference, general
    accelerator: str  # gpu, cpu, neuron


# Common instance types for ML workloads
INSTANCE_TYPES: List[InstanceType] = [
    # GPU Instances - NVIDIA
    InstanceType("ml.g4dn.xlarge", 4, 16, 1, 16, "T4", "Up to 25 Gbps", 0.736, "inference", "gpu"),
    InstanceType("ml.g4dn.2xlarge", 8, 32, 1, 16, "T4", "Up to 25 Gbps", 1.12, "inference", "gpu"),
    InstanceType("ml.g4dn.12xlarge", 48, 192, 4, 64, "T4", "50 Gbps", 5.67, "inference", "gpu"),
    InstanceType("ml.g5.xlarge", 4, 16, 1, 24, "A10G", "Up to 10 Gbps", 1.41, "inference", "gpu"),
    InstanceType("ml.g5.2xlarge", 8, 32, 1, 24, "A10G", "Up to 10 Gbps", 1.69, "inference", "gpu"),
    InstanceType("ml.g5.4xlarge", 16, 64, 1, 24, "A10G", "Up to 25 Gbps", 2.27, "inference", "gpu"),
    InstanceType("ml.g5.8xlarge", 32, 128, 1, 24, "A10G", "25 Gbps", 3.40, "inference", "gpu"),
    InstanceType("ml.g5.12xlarge", 48, 192, 4, 96, "A10G", "40 Gbps", 8.14, "training", "gpu"),
    InstanceType("ml.g5.24xlarge", 96, 384, 4, 96, "A10G", "50 Gbps", 11.34, "training", "gpu"),
    InstanceType("ml.g5.48xlarge", 192, 768, 8, 192, "A10G", "100 Gbps", 22.68, "training", "gpu"),
    InstanceType("ml.p3.2xlarge", 8, 61, 1, 16, "V100", "Up to 10 Gbps", 3.825, "training", "gpu"),
    InstanceType("ml.p3.8xlarge", 32, 244, 4, 64, "V100", "10 Gbps", 14.688, "training", "gpu"),
    InstanceType("ml.p3.16xlarge", 64, 488, 8, 128, "V100", "25 Gbps", 28.152, "training", "gpu"),
    InstanceType(
        "ml.p4d.24xlarge", 96, 1152, 8, 320, "A100", "400 Gbps", 37.688, "training", "gpu"
    ),
    InstanceType(
        "ml.p4de.24xlarge", 96, 1152, 8, 640, "A100-80GB", "400 Gbps", 44.856, "training", "gpu"
    ),
    InstanceType(
        "ml.p5.48xlarge", 192, 2048, 8, 640, "H100", "3200 Gbps", 98.32, "training", "gpu"
    ),
    # CPU Instances
    InstanceType("ml.m5.large", 2, 8, 0, 0, "None", "Up to 10 Gbps", 0.115, "inference", "cpu"),
    InstanceType("ml.m5.xlarge", 4, 16, 0, 0, "None", "Up to 10 Gbps", 0.23, "inference", "cpu"),
    InstanceType("ml.m5.2xlarge", 8, 32, 0, 0, "None", "Up to 10 Gbps", 0.461, "inference", "cpu"),
    InstanceType("ml.m5.4xlarge", 16, 64, 0, 0, "None", "Up to 10 Gbps", 0.922, "inference", "cpu"),
    InstanceType("ml.c5.xlarge", 4, 8, 0, 0, "None", "Up to 10 Gbps", 0.204, "inference", "cpu"),
    InstanceType("ml.c5.2xlarge", 8, 16, 0, 0, "None", "Up to 10 Gbps", 0.408, "inference", "cpu"),
    InstanceType("ml.c5.4xlarge", 16, 32, 0, 0, "None", "Up to 10 Gbps", 0.816, "inference", "cpu"),
    InstanceType("ml.c5.9xlarge", 36, 72, 0, 0, "None", "10 Gbps", 1.836, "training", "cpu"),
    # Neuron Instances (Inferentia/Trainium)
    InstanceType(
        "ml.inf1.xlarge", 4, 8, 1, 8, "Inferentia", "Up to 25 Gbps", 0.297, "inference", "neuron"
    ),
    InstanceType(
        "ml.inf1.2xlarge", 8, 16, 1, 8, "Inferentia", "Up to 25 Gbps", 0.489, "inference", "neuron"
    ),
    InstanceType(
        "ml.inf1.6xlarge", 24, 48, 4, 32, "Inferentia", "25 Gbps", 1.467, "inference", "neuron"
    ),
    InstanceType(
        "ml.inf1.24xlarge", 96, 192, 16, 128, "Inferentia", "100 Gbps", 5.868, "inference", "neuron"
    ),
    InstanceType(
        "ml.inf2.xlarge", 4, 16, 1, 32, "Inferentia2", "Up to 15 Gbps", 0.758, "inference", "neuron"
    ),
    InstanceType(
        "ml.inf2.8xlarge",
        32,
        128,
        1,
        32,
        "Inferentia2",
        "Up to 25 Gbps",
        1.968,
        "inference",
        "neuron",
    ),
    InstanceType(
        "ml.inf2.24xlarge", 96, 384, 6, 192, "Inferentia2", "50 Gbps", 6.489, "inference", "neuron"
    ),
    InstanceType(
        "ml.inf2.48xlarge",
        192,
        768,
        12,
        384,
        "Inferentia2",
        "100 Gbps",
        12.978,
        "inference",
        "neuron",
    ),
    InstanceType(
        "ml.trn1.2xlarge", 8, 32, 1, 32, "Trainium", "Up to 12.5 Gbps", 1.343, "training", "neuron"
    ),
    InstanceType(
        "ml.trn1.32xlarge", 128, 512, 16, 512, "Trainium", "800 Gbps", 21.5, "training", "neuron"
    ),
    InstanceType(
        "ml.trn1n.32xlarge", 128, 512, 16, 512, "Trainium", "1600 Gbps", 24.78, "training", "neuron"
    ),
]


def get_instance_recommendation(
    model_size_gb: float,
    use_case: str = "inference",
    batch_size: int = 1,
    target_latency_ms: Optional[int] = None,
    budget_per_hour: Optional[float] = None,
    prefer_cost_optimization: bool = False,
) -> Dict[str, Any]:
    """
    Get instance type recommendation based on workload requirements.

    Args:
        model_size_gb: Model size in GB (weights + overhead)
        use_case: training or inference
        batch_size: Expected batch size
        target_latency_ms: Target latency in milliseconds
        budget_per_hour: Maximum hourly budget in USD
        prefer_cost_optimization: Prefer cheaper options
    """
    try:
        # Filter by use case
        candidates = [
            i for i in INSTANCE_TYPES if i.category == use_case or i.category == "general"
        ]

        # Filter by budget if specified
        if budget_per_hour:
            candidates = [i for i in candidates if i.price_per_hour_usd <= budget_per_hour]

        # Calculate required GPU memory (model + KV cache + overhead)
        required_gpu_memory = model_size_gb * 1.5  # 50% overhead for activations/KV cache

        # Filter by GPU memory
        gpu_candidates = [
            i for i in candidates if i.gpu_memory_gb >= required_gpu_memory and i.gpu_count > 0
        ]
        cpu_candidates = [i for i in candidates if i.gpu_count == 0]

        recommendations = []

        # GPU recommendations
        if gpu_candidates:
            # Sort by cost-effectiveness (memory per dollar)
            if prefer_cost_optimization:
                gpu_candidates.sort(key=lambda x: x.price_per_hour_usd)
            else:
                gpu_candidates.sort(key=lambda x: -x.gpu_memory_gb / x.price_per_hour_usd)

            best_gpu = gpu_candidates[0]
            recommendations.append(
                {
                    "instance_type": best_gpu.name,
                    "category": "GPU (Recommended)",
                    "gpu_type": best_gpu.gpu_type,
                    "gpu_count": best_gpu.gpu_count,
                    "gpu_memory_gb": best_gpu.gpu_memory_gb,
                    "vcpus": best_gpu.vcpus,
                    "memory_gb": best_gpu.memory_gb,
                    "price_per_hour": best_gpu.price_per_hour_usd,
                    "estimated_monthly_cost": best_gpu.price_per_hour_usd * 730,
                    "reason": f"Sufficient GPU memory ({best_gpu.gpu_memory_gb}GB) for {model_size_gb}GB model",
                }
            )

        # Neuron recommendations for cost optimization
        neuron_candidates = [
            i
            for i in candidates
            if i.accelerator == "neuron" and i.gpu_memory_gb >= required_gpu_memory
        ]
        if neuron_candidates:
            neuron_candidates.sort(key=lambda x: x.price_per_hour_usd)
            best_neuron = neuron_candidates[0]
            savings = 0
            if recommendations:
                savings = (
                    (recommendations[0]["price_per_hour"] - best_neuron.price_per_hour_usd)
                    / recommendations[0]["price_per_hour"]
                    * 100
                )
            recommendations.append(
                {
                    "instance_type": best_neuron.name,
                    "category": "Neuron (Cost-Optimized)",
                    "gpu_type": best_neuron.gpu_type,
                    "gpu_count": best_neuron.gpu_count,
                    "gpu_memory_gb": best_neuron.gpu_memory_gb,
                    "vcpus": best_neuron.vcpus,
                    "memory_gb": best_neuron.memory_gb,
                    "price_per_hour": best_neuron.price_per_hour_usd,
                    "estimated_monthly_cost": best_neuron.price_per_hour_usd * 730,
                    "potential_savings_percent": round(savings, 1),
                    "reason": "AWS Inferentia/Trainium offers lower cost for compatible models",
                    "note": "Requires model compilation with Neuron SDK",
                }
            )

        # Multi-GPU recommendation for large models
        if model_size_gb > 40:
            multi_gpu = [
                i for i in INSTANCE_TYPES if i.gpu_count >= 4 and i.gpu_memory_gb >= model_size_gb
            ]
            if multi_gpu:
                multi_gpu.sort(key=lambda x: x.price_per_hour_usd)
                best_multi = multi_gpu[0]
                recommendations.append(
                    {
                        "instance_type": best_multi.name,
                        "category": "Multi-GPU (Large Models)",
                        "gpu_type": best_multi.gpu_type,
                        "gpu_count": best_multi.gpu_count,
                        "gpu_memory_gb": best_multi.gpu_memory_gb,
                        "vcpus": best_multi.vcpus,
                        "memory_gb": best_multi.memory_gb,
                        "price_per_hour": best_multi.price_per_hour_usd,
                        "estimated_monthly_cost": best_multi.price_per_hour_usd * 730,
                        "reason": f"Tensor parallelism across {best_multi.gpu_count} GPUs for large model",
                    }
                )

        if not recommendations:
            return {
                "success": False,
                "error": "No suitable instances found for the given requirements",
                "suggestions": [
                    "Try increasing budget_per_hour",
                    "Consider model quantization to reduce memory requirements",
                    "Use model sharding across multiple instances",
                ],
            }

        return {
            "success": True,
            "input": {
                "model_size_gb": model_size_gb,
                "use_case": use_case,
                "batch_size": batch_size,
                "target_latency_ms": target_latency_ms,
                "budget_per_hour": budget_per_hour,
            },
            "recommendations": recommendations,
            "tips": [
                "Use Spot instances for training to save up to 90%",
                "Consider Reserved Instances for production inference",
                "Enable auto-scaling for variable workloads",
            ],
        }
    except Exception as e:
        logger.error(f"Error getting instance recommendation: {e}")
        return {"success": False, "error": str(e)}


def estimate_training_cost(
    model_size_gb: float,
    dataset_size_gb: float,
    epochs: int,
    batch_size: int,
    instance_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Estimate training cost for a deep learning job.
    """
    try:
        # Find instance or recommend one
        instance = None
        if instance_type:
            instance = next((i for i in INSTANCE_TYPES if i.name == instance_type), None)

        if not instance:
            # Auto-select based on model size
            candidates = [
                i
                for i in INSTANCE_TYPES
                if i.category == "training" and i.gpu_memory_gb >= model_size_gb * 1.5
            ]
            if candidates:
                candidates.sort(key=lambda x: x.price_per_hour_usd)
                instance = candidates[0]

        if not instance:
            return {"success": False, "error": "No suitable training instance found"}

        # Rough estimation: ~1GB/s throughput for modern GPUs
        throughput_gb_per_hour = 3600 * instance.gpu_count  # Simplified
        total_data_gb = dataset_size_gb * epochs
        estimated_hours = max(1, total_data_gb / throughput_gb_per_hour)

        on_demand_cost = estimated_hours * instance.price_per_hour_usd
        spot_cost = on_demand_cost * 0.3  # ~70% savings with Spot

        return {
            "success": True,
            "instance_type": instance.name,
            "gpu_type": instance.gpu_type,
            "gpu_count": instance.gpu_count,
            "estimated_hours": round(estimated_hours, 1),
            "cost_estimates": {
                "on_demand": round(on_demand_cost, 2),
                "spot_estimated": round(spot_cost, 2),
                "reserved_1yr": round(on_demand_cost * 0.6, 2),
            },
            "input": {
                "model_size_gb": model_size_gb,
                "dataset_size_gb": dataset_size_gb,
                "epochs": epochs,
                "batch_size": batch_size,
            },
            "note": "Estimates are approximate. Actual costs depend on model architecture and optimization.",
        }
    except Exception as e:
        logger.error(f"Error estimating training cost: {e}")
        return {"success": False, "error": str(e)}


def list_gpu_instances(
    min_gpu_memory: Optional[float] = None,
    max_price_per_hour: Optional[float] = None,
    gpu_type: Optional[str] = None,
) -> Dict[str, Any]:
    """List available GPU instances with optional filtering."""
    try:
        instances = [i for i in INSTANCE_TYPES if i.gpu_count > 0]

        if min_gpu_memory:
            instances = [i for i in instances if i.gpu_memory_gb >= min_gpu_memory]
        if max_price_per_hour:
            instances = [i for i in instances if i.price_per_hour_usd <= max_price_per_hour]
        if gpu_type:
            instances = [i for i in instances if gpu_type.lower() in i.gpu_type.lower()]

        instances.sort(key=lambda x: x.price_per_hour_usd)

        return {
            "success": True,
            "total_instances": len(instances),
            "instances": [
                {
                    "name": i.name,
                    "gpu_type": i.gpu_type,
                    "gpu_count": i.gpu_count,
                    "gpu_memory_gb": i.gpu_memory_gb,
                    "vcpus": i.vcpus,
                    "memory_gb": i.memory_gb,
                    "price_per_hour": i.price_per_hour_usd,
                    "category": i.category,
                    "accelerator": i.accelerator,
                }
                for i in instances
            ],
        }
    except Exception as e:
        logger.error(f"Error listing GPU instances: {e}")
        return {"success": False, "error": str(e)}


def register_module(mcp: FastMCP) -> None:
    """Register instance advisor tools with the MCP server."""

    @mcp.tool(
        name="get_instance_recommendation",
        description="Get EC2/SageMaker instance type recommendations based on model size and workload requirements. Considers GPU memory, cost, and performance.",
    )
    async def mcp_get_instance_recommendation(
        model_size_gb: float = Field(..., description="Model size in GB (weights)"),
        use_case: str = Field("inference", description="Use case: training or inference"),
        batch_size: int = Field(1, description="Expected batch size"),
        target_latency_ms: Optional[int] = Field(None, description="Target latency in ms"),
        budget_per_hour: Optional[float] = Field(None, description="Max hourly budget in USD"),
        prefer_cost_optimization: bool = Field(False, description="Prefer cheaper options"),
    ) -> Dict[str, Any]:
        return get_instance_recommendation(
            model_size_gb,
            use_case,
            batch_size,
            target_latency_ms,
            budget_per_hour,
            prefer_cost_optimization,
        )

    @mcp.tool(
        name="estimate_training_cost",
        description="Estimate the cost of a deep learning training job based on model size, dataset, and training parameters.",
    )
    async def mcp_estimate_training_cost(
        model_size_gb: float = Field(..., description="Model size in GB"),
        dataset_size_gb: float = Field(..., description="Dataset size in GB"),
        epochs: int = Field(..., description="Number of training epochs"),
        batch_size: int = Field(32, description="Training batch size"),
        instance_type: Optional[str] = Field(
            None, description="Specific instance type (auto-selected if not provided)"
        ),
    ) -> Dict[str, Any]:
        return estimate_training_cost(
            model_size_gb, dataset_size_gb, epochs, batch_size, instance_type
        )

    @mcp.tool(
        name="list_gpu_instances",
        description="List available GPU instances for ML workloads with optional filtering by memory, price, and GPU type.",
    )
    async def mcp_list_gpu_instances(
        min_gpu_memory: Optional[float] = Field(None, description="Minimum GPU memory in GB"),
        max_price_per_hour: Optional[float] = Field(
            None, description="Maximum price per hour in USD"
        ),
        gpu_type: Optional[str] = Field(
            None, description="GPU type filter (T4, A10G, A100, H100, V100, Inferentia, Trainium)"
        ),
    ) -> Dict[str, Any]:
        return list_gpu_instances(min_gpu_memory, max_price_per_hour, gpu_type)
