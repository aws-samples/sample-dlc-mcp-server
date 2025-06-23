"""Module for DLC best practices guidance."""

import logging
from typing import Dict, Any, List, Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


def get_security_best_practices() -> Dict[str, Any]:
    """
    Get security best practices for DLC usage.

    Returns:
        Dict[str, Any]: Security best practices
    """
    return {
        "image_security": [
            "Always use the latest DLC images with security patches",
            "Scan images for vulnerabilities using Amazon ECR image scanning",
            "Avoid running containers as root; use the least privileged user",
            "Remove unnecessary tools and packages from custom images",
            "Use multi-stage builds to minimize image size and attack surface",
        ],
        "runtime_security": [
            "Use IAM roles for service accounts (IRSA) in Kubernetes",
            "Implement network policies to restrict container communications",
            "Enable AWS CloudTrail for auditing container activities",
            "Use secrets management solutions (AWS Secrets Manager, SSM Parameter Store) instead of environment variables",
            "Implement resource quotas and limits to prevent DoS attacks",
        ],
        "data_security": [
            "Encrypt data at rest using AWS KMS",
            "Use TLS for data in transit",
            "Implement proper access controls for S3 buckets containing training data",
            "Sanitize and validate all input data",
            "Implement proper logging but avoid logging sensitive information",
        ],
        "model_security": [
            "Protect model artifacts with appropriate access controls",
            "Implement model monitoring for detecting anomalies",
            "Consider model encryption for sensitive models",
            "Implement input validation for inference endpoints",
            "Use VPC endpoints for private communication with AWS services",
        ],
    }


def get_cost_optimization_tips() -> Dict[str, Any]:
    """
    Get cost optimization tips for DLC usage.

    Returns:
        Dict[str, Any]: Cost optimization tips
    """
    return {
        "instance_selection": [
            "Use Spot Instances for training workloads that can handle interruptions",
            "Right-size instances based on model requirements",
            "Consider using Graviton-based instances for cost-effective inference",
            "Use auto-scaling for variable workloads",
            "Consider using SageMaker managed spot training for cost savings",
        ],
        "storage_optimization": [
            "Use Amazon S3 for storing datasets and model artifacts",
            "Implement lifecycle policies for S3 to transition or expire objects",
            "Use EFS or FSx for shared storage when needed",
            "Compress large datasets and model artifacts",
            "Clean up unused EBS volumes and snapshots",
        ],
        "container_optimization": [
            "Optimize container size by removing unnecessary dependencies",
            "Use multi-stage builds to create smaller images",
            "Implement proper caching strategies for Docker layers",
            "Share base images across multiple containers",
            "Use ECR lifecycle policies to clean up unused images",
        ],
        "workflow_optimization": [
            "Implement checkpointing to resume training from interruptions",
            "Use incremental training when possible",
            "Implement proper shutdown of resources after completion",
            "Use AWS Batch for cost-effective batch processing",
            "Implement automated resource cleanup using AWS Lambda",
        ],
    }


def get_deployment_best_practices(platform: str, use_case: str) -> Dict[str, Any]:
    """
    Get deployment best practices for DLC usage.

    Args:
        platform (str): Deployment platform (ec2, sagemaker, ecs, eks)
        use_case (str): Use case (training, inference)

    Returns:
        Dict[str, Any]: Deployment best practices
    """
    # Common best practices
    common_practices = {
        "training": [
            "Use checkpointing to save progress and enable resuming",
            "Implement proper error handling and logging",
            "Monitor resource utilization during training",
            "Use distributed training for large models",
            "Implement proper data validation before training",
        ],
        "inference": [
            "Optimize model for inference (pruning, quantization)",
            "Implement proper input validation",
            "Set appropriate timeouts for inference requests",
            "Monitor inference latency and throughput",
            "Implement proper error handling for inference failures",
        ],
    }

    # Platform-specific best practices
    platform_practices = {
        "ec2": {
            "training": [
                "Use instance storage for temporary datasets",
                "Implement automatic shutdown after training completion",
                "Use placement groups for high-performance networking",
                "Consider using EC2 Fleet for mixed instance types",
                "Use EFA for multi-node distributed training",
            ],
            "inference": [
                "Use Auto Scaling groups for variable workloads",
                "Implement health checks and automatic recovery",
                "Use Elastic Load Balancing for distributing requests",
                "Consider using Elastic Inference for cost-effective inference",
                "Use instance metadata for configuration",
            ],
        },
        "sagemaker": {
            "training": [
                "Use SageMaker managed spot training for cost savings",
                "Implement proper hyperparameter tuning",
                "Use SageMaker Experiments for tracking",
                "Optimize data loading with SageMaker Pipe mode",
                "Use SageMaker Debugger for monitoring training",
            ],
            "inference": [
                "Use SageMaker multi-model endpoints for serving multiple models",
                "Implement auto-scaling for SageMaker endpoints",
                "Consider using SageMaker Serverless Inference for variable workloads",
                "Use SageMaker Model Monitor for monitoring",
                "Implement proper endpoint configuration for cost-performance balance",
            ],
        },
        "ecs": {
            "training": [
                "Use ECS task placement strategies for optimal resource utilization",
                "Implement proper task definition with resource limits",
                "Use ECS service for managing training tasks",
                "Consider using Fargate for serverless training",
                "Implement proper logging with CloudWatch Logs",
            ],
            "inference": [
                "Use ECS service auto-scaling for variable workloads",
                "Implement health checks for containers",
                "Use Application Load Balancer for distributing requests",
                "Consider using Fargate for serverless inference",
                "Implement proper service discovery",
            ],
        },
        "eks": {
            "training": [
                "Use Kubernetes Job or CronJob for training workloads",
                "Implement proper resource requests and limits",
                "Use node selectors or taints/tolerations for GPU nodes",
                "Implement Horizontal Pod Autoscaler for distributed training",
                "Use Persistent Volumes for storing checkpoints",
            ],
            "inference": [
                "Use Kubernetes Deployments with proper replica count",
                "Implement Horizontal Pod Autoscaler for variable workloads",
                "Use Kubernetes Service for load balancing",
                "Implement proper readiness and liveness probes",
                "Consider using Kubernetes Vertical Pod Autoscaler",
            ],
        },
    }

    platform = platform.lower()
    use_case = use_case.lower()

    if platform not in platform_practices:
        return {
            "error": f"Platform '{platform}' not found in best practices database",
            "available_platforms": list(platform_practices.keys()),
        }

    if use_case not in common_practices:
        return {
            "error": f"Use case '{use_case}' not found in best practices database",
            "available_use_cases": list(common_practices.keys()),
        }

    return {
        "platform": platform,
        "use_case": use_case,
        "common_practices": common_practices[use_case],
        "platform_practices": platform_practices[platform][use_case],
    }


def get_framework_specific_best_practices(framework: str, use_case: str) -> Dict[str, Any]:
    """
    Get framework-specific best practices for DLC usage.

    Args:
        framework (str): ML framework (pytorch, tensorflow, etc.)
        use_case (str): Use case (training, inference)

    Returns:
        Dict[str, Any]: Framework-specific best practices
    """
    framework_practices = {
        "pytorch": {
            "training": [
                "Use torch.compile() for PyTorch 2.0+ to optimize execution",
                "Implement proper data loading with DataLoader and num_workers",
                "Use mixed precision training with torch.cuda.amp",
                "Implement gradient accumulation for large batch sizes",
                "Use DistributedDataParallel for multi-GPU training",
                "Implement proper checkpointing with torch.save()",
                "Use torch.jit for optimizing critical components",
                "Implement proper learning rate scheduling",
                "Use torch.profiler to identify bottlenecks",
            ],
            "inference": [
                "Use TorchScript or ONNX for deployment",
                "Implement batching for higher throughput",
                "Use torch.inference_mode() instead of torch.no_grad()",
                "Consider quantization for faster inference",
                "Use TensorRT for optimized GPU inference",
                "Implement proper model loading with torch.load()",
                "Use torch.jit.trace for optimized inference",
                "Consider using torch.fx for model transformations",
                "Optimize memory usage with torch.cuda.empty_cache()",
            ],
        },
        "tensorflow": {
            "training": [
                "Use tf.data pipeline with prefetching and parallel processing",
                "Enable XLA compilation for faster training",
                "Use mixed precision with tf.keras.mixed_precision",
                "Implement proper distribution strategy for multi-GPU training",
                "Use tf.keras.callbacks for monitoring and checkpointing",
                "Implement proper learning rate scheduling",
                "Use tf.function to optimize execution",
                "Implement gradient accumulation for large batch sizes",
                "Use TensorBoard for monitoring training",
            ],
            "inference": [
                "Use SavedModel format for deployment",
                "Consider TensorFlow Lite for edge deployment",
                "Use TensorFlow Serving for optimized inference",
                "Implement batching for higher throughput",
                "Use TensorRT integration for optimized GPU inference",
                "Implement proper input preprocessing",
                "Use tf.function for optimized inference",
                "Consider quantization for faster inference",
                "Optimize thread settings with inter_op and intra_op parallelism",
            ],
        },
        "mxnet": {
            "training": [
                "Use proper data loading with mx.gluon.data.DataLoader",
                "Implement Hybridization for faster training",
                "Use mixed precision training with amp",
                "Implement proper distribution for multi-GPU training",
                "Use proper checkpointing with model.save_parameters()",
                "Implement proper learning rate scheduling",
                "Use MXBoard for monitoring training",
                "Optimize memory usage with gradient compression",
                "Use proper initialization for faster convergence",
            ],
            "inference": [
                "Export models to Symbol format for deployment",
                "Use MXNet Model Server for serving",
                "Implement batching for higher throughput",
                "Consider quantization for faster inference",
                "Use proper context (CPU/GPU) for inference",
                "Implement proper input preprocessing",
                "Use mx.nd.waitall() for synchronization",
                "Optimize memory usage with mx.nd.zeros()",
                "Consider using ONNX for cross-framework deployment",
            ],
        },
    }

    framework = framework.lower()
    use_case = use_case.lower()

    if framework not in framework_practices:
        return {
            "error": f"Framework '{framework}' not found in best practices database",
            "available_frameworks": list(framework_practices.keys()),
        }

    if use_case not in framework_practices[framework]:
        return {
            "error": f"Use case '{use_case}' not found for framework '{framework}'",
            "available_use_cases": list(framework_practices[framework].keys()),
        }

    return {
        "framework": framework,
        "use_case": use_case,
        "best_practices": framework_practices[framework][use_case],
    }


def get_custom_image_guidelines() -> Dict[str, Any]:
    """
    Get guidelines for creating custom DLC images.

    Returns:
        Dict[str, Any]: Custom image guidelines
    """
    return {
        "dockerfile_best_practices": [
            "Use official AWS DLC images as base",
            "Use multi-stage builds to minimize image size",
            "Group related commands in a single RUN instruction",
            "Remove unnecessary files and packages",
            "Set appropriate environment variables",
            "Use specific versions for dependencies",
            "Include proper documentation in Dockerfile",
            "Use .dockerignore to exclude unnecessary files",
            "Set appropriate user permissions",
            "Include health checks",
        ],
        "dependency_management": [
            "Pin dependency versions for reproducibility",
            "Use requirements.txt or environment.yml for Python dependencies",
            "Consider using pip constraints for compatible dependencies",
            "Clean up package manager caches after installation",
            "Group related dependencies in a single installation command",
            "Consider using virtual environments",
            "Test dependencies for compatibility",
            "Document dependency versions and sources",
            "Consider using dependency scanning tools",
            "Regularly update dependencies for security patches",
        ],
        "optimization_techniques": [
            "Minimize image layers for faster builds and pulls",
            "Use appropriate base image for your use case",
            "Optimize for caching during builds",
            "Remove development dependencies from production images",
            "Use appropriate compression for included files",
            "Consider using squashed images",
            "Optimize startup time by preloading models",
            "Use appropriate entrypoint scripts",
            "Implement proper signal handling",
            "Optimize for specific hardware (CPU/GPU)",
        ],
        "testing_guidelines": [
            "Test images in isolation before deployment",
            "Implement automated testing for custom images",
            "Test with representative workloads",
            "Verify resource usage (CPU, memory, GPU)",
            "Test startup and shutdown behavior",
            "Verify proper error handling",
            "Test with different input data",
            "Verify compatibility with deployment platforms",
            "Implement performance benchmarking",
            "Test for security vulnerabilities",
        ],
    }


def register_module(mcp: FastMCP) -> None:
    """
    Register the best practices module with the MCP server.

    Args:
        mcp (FastMCP): MCP server instance
    """
    mcp.add_tool(
        name="get_security_best_practices",
        description="Get security best practices for DLC usage.",
        fn=get_security_best_practices,
    )

    mcp.add_tool(
        name="get_cost_optimization_tips",
        description="Get cost optimization tips for DLC usage.",
        fn=get_cost_optimization_tips,
    )

    mcp.add_tool(
        name="get_deployment_best_practices",
        description="Get deployment best practices for DLC usage.\n\nArgs:\n    platform: Deployment platform (ec2, sagemaker, ecs, eks)\n    use_case: Use case (training, inference)\n",
        fn=get_deployment_best_practices,
    )

    mcp.add_tool(
        name="get_framework_specific_best_practices",
        description="Get framework-specific best practices for DLC usage.\n\nArgs:\n    framework: ML framework (pytorch, tensorflow, etc.)\n    use_case: Use case (training, inference)\n",
        fn=get_framework_specific_best_practices,
    )

    mcp.add_tool(
        name="get_custom_image_guidelines",
        description="Get guidelines for creating custom DLC images.",
        fn=get_custom_image_guidelines,
    )
