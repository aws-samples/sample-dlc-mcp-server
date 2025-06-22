"""Module for troubleshooting DLC-related issues."""

import logging
import re
from typing import Dict, Any, List, Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


def diagnose_common_issues(
    error_message: str, framework: Optional[str] = None, use_case: Optional[str] = None
) -> Dict[str, Any]:
    """
    Diagnose common DLC-related issues.

    Args:
        error_message (str): Error message
        framework (Optional[str]): ML framework (pytorch, tensorflow, etc.)
        use_case (Optional[str]): Use case (training, inference)

    Returns:
        Dict[str, Any]: Diagnosis result
    """
    # Common error patterns and solutions
    error_patterns = [
        {
            "pattern": r"(?i)cuda.+out of memory",
            "diagnosis": "CUDA out of memory error",
            "solution": [
                "Reduce batch size",
                "Use gradient accumulation",
                "Use mixed precision training",
                "Use a smaller model or optimize model memory usage",
                "Use a GPU instance with more memory",
            ],
        },
        {
            "pattern": r"(?i)no module named ['\"](torch|tensorflow|mxnet)",
            "diagnosis": "Missing framework package",
            "solution": [
                "Ensure you're using the correct DLC image for your framework",
                "Check if the framework is installed in the correct Python environment",
                "Try installing the framework manually: pip install torch/tensorflow/mxnet",
            ],
        },
        {
            "pattern": r"(?i)permission denied",
            "diagnosis": "Permission issue",
            "solution": [
                "Check file and directory permissions",
                "Ensure the container has the necessary permissions",
                "Run the container with appropriate user permissions",
                "Check if you need to mount volumes with correct permissions",
            ],
        },
        {
            "pattern": r"(?i)cannot connect to the docker daemon",
            "diagnosis": "Docker daemon connection issue",
            "solution": [
                "Ensure Docker daemon is running: systemctl start docker",
                "Check if you have permissions to access the Docker socket",
                "Try running with sudo or add your user to the docker group",
            ],
        },
        {
            "pattern": r"(?i)failed to pull image",
            "diagnosis": "Image pull failure",
            "solution": [
                "Check your network connection",
                "Verify you have permissions to pull from the ECR repository",
                "Ensure you've authenticated with ECR: aws ecr get-login-password | docker login --username AWS --password-stdin <registry>",
                "Check if the image URI is correct and exists in the repository",
            ],
        },
        {
            "pattern": r"(?i)resource temporarily unavailable",
            "diagnosis": "Resource availability issue",
            "solution": [
                "Check if you have enough system resources (CPU, memory)",
                "Verify if there are any resource quotas or limits in place",
                "Try scaling down resource requirements or using a larger instance",
            ],
        },
        {
            "pattern": r"(?i)incompatible (cpu|gpu) version",
            "diagnosis": "Hardware compatibility issue",
            "solution": [
                "Ensure your hardware is compatible with the DLC image",
                "Check CUDA version compatibility for GPU images",
                "Use a DLC image that matches your hardware configuration",
                "For GPU instances, ensure NVIDIA drivers are properly installed",
            ],
        },
    ]

    # Framework-specific error patterns
    if framework and framework.lower() == "pytorch":
        error_patterns.extend(
            [
                {
                    "pattern": r"(?i)expected scalar type \w+ but found \w+",
                    "diagnosis": "PyTorch dtype mismatch",
                    "solution": [
                        "Check tensor data types in your model",
                        "Ensure consistent data types across operations",
                        "Convert tensors to the expected data type using .to() method",
                    ],
                },
                {
                    "pattern": r"(?i)size mismatch",
                    "diagnosis": "PyTorch tensor size mismatch",
                    "solution": [
                        "Check tensor dimensions in your model",
                        "Ensure input and output dimensions match between layers",
                        "Use tensor.view() or tensor.reshape() to adjust dimensions",
                    ],
                },
            ]
        )
    elif framework and framework.lower() == "tensorflow":
        error_patterns.extend(
            [
                {
                    "pattern": r"(?i)invalidargumenterror",
                    "diagnosis": "TensorFlow invalid argument",
                    "solution": [
                        "Check input shapes and data types",
                        "Ensure model inputs match expected formats",
                        "Verify tensor dimensions are compatible with operations",
                    ],
                },
                {
                    "pattern": r"(?i)failed to get convolution algorithm",
                    "diagnosis": "TensorFlow CUDA convolution issue",
                    "solution": [
                        "Try setting TF_FORCE_GPU_ALLOW_GROWTH=true",
                        "Reduce model complexity or batch size",
                        "Check CUDA and cuDNN compatibility with TensorFlow version",
                    ],
                },
            ]
        )

    # Use case specific patterns
    if use_case and use_case.lower() == "inference":
        error_patterns.extend(
            [
                {
                    "pattern": r"(?i)model.+not found",
                    "diagnosis": "Model loading issue",
                    "solution": [
                        "Verify the model file path is correct",
                        "Check if the model file exists and is accessible",
                        "Ensure the model format is compatible with your framework",
                        "Check if you need to download or copy the model to the container",
                    ],
                },
                {
                    "pattern": r"(?i)timeout",
                    "diagnosis": "Inference timeout",
                    "solution": [
                        "Increase the timeout setting for your endpoint",
                        "Optimize your model for faster inference",
                        "Consider using a more powerful instance type",
                        "Check for any preprocessing bottlenecks",
                    ],
                },
            ]
        )
    elif use_case and use_case.lower() == "training":
        error_patterns.extend(
            [
                {
                    "pattern": r"(?i)out of memory",
                    "diagnosis": "Training memory issue",
                    "solution": [
                        "Reduce batch size",
                        "Use gradient accumulation",
                        "Enable mixed precision training",
                        "Optimize dataset loading and preprocessing",
                        "Use a more memory-efficient model architecture",
                    ],
                },
                {
                    "pattern": r"(?i)data.+not found",
                    "diagnosis": "Training data access issue",
                    "solution": [
                        "Verify data paths and permissions",
                        "Check if data is correctly mounted or accessible",
                        "Ensure data format is compatible with your data loader",
                        "Check network connectivity to data sources",
                    ],
                },
            ]
        )

    # Match error message against patterns
    matches = []
    for pattern in error_patterns:
        if re.search(pattern["pattern"], error_message):
            matches.append({"diagnosis": pattern["diagnosis"], "solution": pattern["solution"]})

    if not matches:
        # No specific match found, provide general guidance
        return {
            "matched": False,
            "diagnosis": "Unknown issue",
            "general_recommendations": [
                "Check container logs for more detailed error messages",
                "Verify your DLC image is compatible with your use case",
                "Ensure you have the necessary permissions and resources",
                "Check network connectivity and access to required services",
                "Verify your code is compatible with the framework version in the DLC",
            ],
        }

    return {"matched": True, "matches": matches}


def get_framework_compatibility_info(framework: str, version: str) -> Dict[str, Any]:
    """
    Get compatibility information for a specific framework version.

    Args:
        framework (str): ML framework (pytorch, tensorflow, etc.)
        version (str): Framework version

    Returns:
        Dict[str, Any]: Compatibility information
    """
    # Framework compatibility information
    compatibility_info = {
        "pytorch": {
            "2.6.0": {
                "python_versions": ["3.12"],
                "cuda_versions": ["12.6", "12.4"],
                "compatible_libraries": [
                    {"name": "torchvision", "version": "0.17.0"},
                    {"name": "torchaudio", "version": "2.6.0"},
                    {"name": "torchtext", "version": "0.17.0"},
                ],
                "known_issues": [
                    "Some older CUDA operations may not be supported in CUDA 12.6",
                    "Certain custom CUDA extensions may need to be recompiled",
                ],
            },
            "2.0.0": {
                "python_versions": ["3.10"],
                "cuda_versions": ["11.8", "11.7"],
                "compatible_libraries": [
                    {"name": "torchvision", "version": "0.15.0"},
                    {"name": "torchaudio", "version": "2.0.0"},
                    {"name": "torchtext", "version": "0.15.0"},
                ],
                "known_issues": [
                    "Breaking changes in torch.nn module compared to 1.x versions",
                    "Some deprecated features from 1.x have been removed",
                ],
            },
        },
        "tensorflow": {
            "2.18.0": {
                "python_versions": ["3.10"],
                "cuda_versions": ["12.5", "12.2"],
                "compatible_libraries": [
                    {"name": "keras", "version": "3.0.0"},
                    {"name": "tensorflow-datasets", "version": "4.9.0"},
                    {"name": "tensorflow-hub", "version": "0.15.0"},
                ],
                "known_issues": [
                    "Some ops have been deprecated and will be removed in future versions",
                    "Keras 3.0 integration has breaking changes from previous versions",
                ],
            },
            "2.12.0": {
                "python_versions": ["3.9", "3.10"],
                "cuda_versions": ["11.8", "11.7"],
                "compatible_libraries": [
                    {"name": "keras", "version": "2.12.0"},
                    {"name": "tensorflow-datasets", "version": "4.8.0"},
                    {"name": "tensorflow-hub", "version": "0.13.0"},
                ],
                "known_issues": [
                    "Some GPU operations may be slower compared to previous versions",
                    "Memory usage may be higher for certain model architectures",
                ],
            },
        },
    }

    framework = framework.lower()

    if framework not in compatibility_info:
        return {
            "error": f"Framework '{framework}' not found in compatibility database",
            "available_frameworks": list(compatibility_info.keys()),
        }

    if version not in compatibility_info[framework]:
        available_versions = list(compatibility_info[framework].keys())
        return {
            "error": f"Version '{version}' not found for framework '{framework}'",
            "available_versions": available_versions,
        }

    return {
        "framework": framework,
        "version": version,
        "compatibility": compatibility_info[framework][version],
    }


def get_performance_optimization_tips(
    framework: str, use_case: str, device_type: str
) -> Dict[str, Any]:
    """
    Get performance optimization tips for DLC usage.

    Args:
        framework (str): ML framework (pytorch, tensorflow, etc.)
        use_case (str): Use case (training, inference)
        device_type (str): Device type (cpu, gpu)

    Returns:
        Dict[str, Any]: Performance optimization tips
    """
    # Common optimization tips
    common_tips = {
        "data_loading": [
            "Use efficient data formats (e.g., TFRecord, PyTorch DataLoader)",
            "Implement proper data prefetching and caching",
            "Consider using memory-mapped files for large datasets",
            "Use appropriate batch sizes for your hardware",
        ],
        "memory_management": [
            "Monitor memory usage during execution",
            "Release unused tensors and variables",
            "Use checkpointing for large models",
            "Consider gradient accumulation for large models",
        ],
    }

    # Framework-specific tips
    framework_tips = {
        "pytorch": {
            "training": {
                "cpu": [
                    "Use DataLoader with num_workers > 0",
                    "Enable TorchScript for performance-critical components",
                    "Use torch.compile() for PyTorch 2.0+ to optimize execution",
                    "Consider quantization for inference",
                ],
                "gpu": [
                    "Use mixed precision training with torch.cuda.amp",
                    "Enable cudnn benchmarking: torch.backends.cudnn.benchmark = True",
                    "Use DistributedDataParallel for multi-GPU training",
                    "Optimize memory usage with gradient checkpointing",
                ],
            },
            "inference": {
                "cpu": [
                    "Use TorchScript or ONNX for deployment",
                    "Apply quantization to reduce model size and improve speed",
                    "Set appropriate thread settings: torch.set_num_threads()",
                    "Use batch inference when possible",
                ],
                "gpu": [
                    "Use TensorRT for optimized inference",
                    "Apply half-precision (FP16) for faster inference",
                    "Optimize CUDA memory usage with torch.cuda.empty_cache()",
                    "Consider ONNX Runtime with CUDA acceleration",
                ],
            },
        },
        "tensorflow": {
            "training": {
                "cpu": [
                    "Use tf.data pipeline with prefetching and parallel processing",
                    "Enable XLA compilation: tf.config.optimizer.set_jit(True)",
                    "Use TensorFlow's mixed precision API",
                    "Consider using TF Lite for deployment",
                ],
                "gpu": [
                    "Use mixed precision with tf.keras.mixed_precision",
                    "Enable XLA: tf.config.optimizer.set_jit(True)",
                    "Use tf.distribute.MirroredStrategy for multi-GPU training",
                    "Optimize memory usage with gradient accumulation",
                ],
            },
            "inference": {
                "cpu": [
                    "Use TensorFlow Serving for optimized inference",
                    "Apply quantization with TF Lite",
                    "Use SavedModel format for deployment",
                    "Optimize thread settings with inter_op and intra_op parallelism",
                ],
                "gpu": [
                    "Use TensorRT integration for optimized inference",
                    "Apply FP16 precision for faster inference",
                    "Optimize batch size for throughput",
                    "Consider TF Serving with GPU support",
                ],
            },
        },
    }

    framework = framework.lower()
    use_case = use_case.lower()
    device_type = device_type.lower()

    if framework not in framework_tips:
        return {
            "error": f"Framework '{framework}' not found in optimization database",
            "available_frameworks": list(framework_tips.keys()),
        }

    if use_case not in framework_tips[framework]:
        return {
            "error": f"Use case '{use_case}' not found for framework '{framework}'",
            "available_use_cases": list(framework_tips[framework].keys()),
        }

    if device_type not in framework_tips[framework][use_case]:
        return {
            "error": f"Device type '{device_type}' not found for {framework}/{use_case}",
            "available_device_types": list(framework_tips[framework][use_case].keys()),
        }

    return {
        "framework": framework,
        "use_case": use_case,
        "device_type": device_type,
        "common_tips": common_tips,
        "specific_tips": framework_tips[framework][use_case][device_type],
    }


def register_module(mcp: FastMCP) -> None:
    """
    Register the troubleshooting module with the MCP server.

    Args:
        mcp (FastMCP): MCP server instance
    """
    mcp.add_tool(
        name="diagnose_common_issues",
        description="Diagnose common DLC-related issues.\n\nArgs:\n    error_message: Error message\n    framework: Optional ML framework (pytorch, tensorflow, etc.)\n    use_case: Optional use case (training, inference)\n",
        fn=diagnose_common_issues,
    )

    mcp.add_tool(
        name="get_framework_compatibility_info",
        description="Get compatibility information for a specific framework version.\n\nArgs:\n    framework: ML framework (pytorch, tensorflow, etc.)\n    version: Framework version\n",
        fn=get_framework_compatibility_info,
    )

    mcp.add_tool(
        name="get_performance_optimization_tips",
        description="Get performance optimization tips for DLC usage.\n\nArgs:\n    framework: ML framework (pytorch, tensorflow, etc.)\n    use_case: Use case (training, inference)\n    device_type: Device type (cpu, gpu)\n",
        fn=get_performance_optimization_tips,
    )
