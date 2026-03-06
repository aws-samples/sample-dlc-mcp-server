###
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
######

"""
DLC Image Registry - Comprehensive catalog of AWS Deep Learning Container images.
Based on: https://aws.github.io/deep-learning-containers/reference/available_images/
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

# AWS DLC Production Account ID (primary)
AWS_DLC_ACCOUNT_ID = "763104351884"

# Region-specific account IDs for DLC images
REGION_ACCOUNT_MAP: Dict[str, str] = {
    "us-east-1": "763104351884",
    "us-east-2": "763104351884",
    "us-west-1": "763104351884",
    "us-west-2": "763104351884",
    "ap-south-1": "763104351884",
    "ap-northeast-2": "763104351884",
    "ap-southeast-1": "763104351884",
    "ap-southeast-2": "763104351884",
    "ap-northeast-1": "763104351884",
    "ca-central-1": "763104351884",
    "eu-central-1": "763104351884",
    "eu-west-1": "763104351884",
    "eu-west-2": "763104351884",
    "eu-west-3": "763104351884",
    "eu-north-1": "763104351884",
    "sa-east-1": "763104351884",
    "af-south-1": "626614931356",
    "ap-east-1": "871362719292",
    "ap-south-2": "772153158452",
    "ap-southeast-3": "907027046896",
    "ap-southeast-4": "457447274322",
    "ap-southeast-5": "550225433462",
    "ap-southeast-7": "590183813437",
    "ap-east-2": "975050140332",
    "ap-northeast-3": "364406365360",
    "ca-west-1": "204538143572",
    "eu-south-1": "692866216735",
    "eu-south-2": "503227376785",
    "eu-central-2": "380420809688",
    "il-central-1": "780543022126",
    "mx-central-1": "637423239942",
    "me-south-1": "217643126080",
    "me-central-1": "914824155844",
    "cn-north-1": "727897471807",
    "cn-northwest-1": "727897471807",
}

NEURON_SUPPORTED_REGIONS = [
    "us-east-1",
    "us-east-2",
    "us-west-2",
    "ap-south-1",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-northeast-1",
    "ap-east-2",
    "eu-central-1",
    "eu-west-1",
    "eu-west-3",
    "sa-east-1",
]


@dataclass
class DLCImage:
    """Represents a DLC image with all its metadata."""

    framework: str
    version: str
    python_version: str
    accelerator: str  # gpu, cpu, neuronx
    platform: str  # ec2, sagemaker, ecs, eks
    use_case: str  # training, inference
    cuda_version: Optional[str] = None
    sdk_version: Optional[str] = None
    transformers_version: Optional[str] = None
    optimum_version: Optional[str] = None
    engine_version: Optional[str] = None
    os_version: str = "ubuntu22.04"
    architecture: str = "x86_64"

    def get_repository_name(self) -> str:
        """Get the ECR repository name for this image."""
        base = self.framework
        if self.use_case and self.framework not in ["base", "sglang", "vllm", "djl-inference"]:
            base = f"{self.framework}-{self.use_case}"
        if self.architecture == "arm64":
            base = f"{base}-arm64"
        if self.accelerator == "neuronx":
            base = f"{base.replace('-inference', '-inference-neuronx').replace('-training', '-training-neuronx')}"
            if "neuronx" not in base:
                base = f"{base}-neuronx"
        return base

    def get_image_tag(self) -> str:
        """Generate the image tag based on image properties."""
        parts = [self.version]
        if self.accelerator == "neuronx":
            parts.append("neuronx")
            parts.append(f"py{self.python_version.replace('.', '')}")
            if self.sdk_version:
                parts.append(f"sdk{self.sdk_version}")
        elif self.accelerator == "gpu":
            parts.append("gpu")
            parts.append(f"py{self.python_version.replace('.', '')}")
            if self.cuda_version:
                parts.append(f"cu{self.cuda_version.replace('.', '')}")
        else:
            parts.append("cpu")
            parts.append(f"py{self.python_version.replace('.', '')}")
        parts.append(self.os_version)
        parts.append(self.platform)
        return "-".join(parts)

    def get_full_uri(self, region: str = "us-west-2") -> str:
        """Get the full ECR URI for this image."""
        account_id = REGION_ACCOUNT_MAP.get(region, AWS_DLC_ACCOUNT_ID)
        suffix = ".amazonaws.com.cn" if region.startswith("cn-") else ".amazonaws.com"
        return f"{account_id}.dkr.ecr.{region}{suffix}/{self.get_repository_name()}:{self.get_image_tag()}"

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "framework": self.framework,
            "version": self.version,
            "python_version": self.python_version,
            "accelerator": self.accelerator,
            "platform": self.platform,
            "use_case": self.use_case,
            "cuda_version": self.cuda_version,
            "sdk_version": self.sdk_version,
            "transformers_version": self.transformers_version,
            "os_version": self.os_version,
            "architecture": self.architecture,
            "repository_name": self.get_repository_name(),
        }


# =============================================================================
# COMPREHENSIVE DLC IMAGE CATALOG
# Updated: March 2026 from https://aws.github.io/deep-learning-containers/reference/available_images/
# =============================================================================

DLC_IMAGES: List[DLCImage] = [
    # -------------------------------------------------------------------------
    # PyTorch Training Images
    # -------------------------------------------------------------------------
    DLCImage("pytorch", "2.9.0", "3.12", "gpu", "sagemaker", "training", cuda_version="13.0"),
    DLCImage("pytorch", "2.9.0", "3.12", "cpu", "sagemaker", "training"),
    DLCImage("pytorch", "2.9.0", "3.12", "gpu", "ec2", "training", cuda_version="13.0"),
    DLCImage("pytorch", "2.9.0", "3.12", "cpu", "ec2", "training"),
    DLCImage("pytorch", "2.8.0", "3.12", "gpu", "sagemaker", "training", cuda_version="12.9"),
    DLCImage("pytorch", "2.8.0", "3.12", "cpu", "sagemaker", "training"),
    DLCImage("pytorch", "2.8.0", "3.12", "gpu", "ec2", "training", cuda_version="12.9"),
    DLCImage("pytorch", "2.8.0", "3.12", "cpu", "ec2", "training"),
    DLCImage("pytorch", "2.7.1", "3.12", "gpu", "sagemaker", "training", cuda_version="12.8"),
    DLCImage("pytorch", "2.7.1", "3.12", "cpu", "sagemaker", "training"),
    DLCImage("pytorch", "2.7.1", "3.12", "gpu", "ec2", "training", cuda_version="12.8"),
    DLCImage("pytorch", "2.7.1", "3.12", "cpu", "ec2", "training"),
    # -------------------------------------------------------------------------
    # PyTorch Inference Images
    # -------------------------------------------------------------------------
    DLCImage("pytorch", "2.6.0", "3.12", "gpu", "sagemaker", "inference", cuda_version="12.4"),
    DLCImage("pytorch", "2.6.0", "3.12", "cpu", "sagemaker", "inference"),
    DLCImage("pytorch", "2.6.0", "3.12", "gpu", "ec2", "inference", cuda_version="12.4"),
    DLCImage("pytorch", "2.6.0", "3.12", "cpu", "ec2", "inference"),
    # -------------------------------------------------------------------------
    # PyTorch ARM64 Images
    # -------------------------------------------------------------------------
    DLCImage(
        "pytorch",
        "2.7.0",
        "3.12",
        "gpu",
        "ec2",
        "training",
        cuda_version="12.8",
        architecture="arm64",
    ),
    DLCImage(
        "pytorch",
        "2.6.0",
        "3.12",
        "gpu",
        "ec2",
        "inference",
        cuda_version="12.4",
        architecture="arm64",
    ),
    DLCImage("pytorch", "2.6.0", "3.12", "cpu", "sagemaker", "inference", architecture="arm64"),
    DLCImage("pytorch", "2.6.0", "3.12", "cpu", "ec2", "inference", architecture="arm64"),
    # -------------------------------------------------------------------------
    # TensorFlow Training Images
    # -------------------------------------------------------------------------
    DLCImage("tensorflow", "2.19.0", "3.12", "gpu", "sagemaker", "training", cuda_version="12.5"),
    DLCImage("tensorflow", "2.19.0", "3.12", "cpu", "sagemaker", "training"),
    # -------------------------------------------------------------------------
    # TensorFlow Inference Images
    # -------------------------------------------------------------------------
    DLCImage("tensorflow", "2.19.0", "3.12", "gpu", "sagemaker", "inference", cuda_version="12.2"),
    DLCImage("tensorflow", "2.19.0", "3.12", "cpu", "sagemaker", "inference"),
    DLCImage("tensorflow", "2.19.0", "3.12", "cpu", "sagemaker", "inference", architecture="arm64"),
    # -------------------------------------------------------------------------
    # vLLM Images
    # -------------------------------------------------------------------------
    DLCImage("vllm", "0.15.1", "3.12", "gpu", "sagemaker", "inference", cuda_version="12.9"),
    DLCImage("vllm", "0.15.1", "3.12", "gpu", "ec2", "inference", cuda_version="12.9"),
    DLCImage("vllm", "0.14.0", "3.12", "gpu", "sagemaker", "inference", cuda_version="12.9"),
    DLCImage("vllm", "0.14.0", "3.12", "gpu", "ec2", "inference", cuda_version="12.9"),
    DLCImage("vllm", "0.13.0", "3.12", "gpu", "sagemaker", "inference", cuda_version="12.9"),
    DLCImage("vllm", "0.13.0", "3.12", "gpu", "ec2", "inference", cuda_version="12.9"),
    DLCImage(
        "vllm",
        "0.10.2",
        "3.12",
        "gpu",
        "ec2",
        "inference",
        cuda_version="12.9",
        architecture="arm64",
    ),
    # -------------------------------------------------------------------------
    # SGLang Images
    # -------------------------------------------------------------------------
    DLCImage(
        "sglang",
        "0.5.8",
        "3.12",
        "gpu",
        "sagemaker",
        "inference",
        cuda_version="12.9",
        os_version="ubuntu24.04",
    ),
    DLCImage(
        "sglang",
        "0.5.7",
        "3.12",
        "gpu",
        "sagemaker",
        "inference",
        cuda_version="12.9",
        os_version="ubuntu24.04",
    ),
    DLCImage("sglang", "0.5.6", "3.12", "gpu", "sagemaker", "inference", cuda_version="12.9"),
    DLCImage("sglang", "0.5.5", "3.12", "gpu", "sagemaker", "inference", cuda_version="12.9"),
    # -------------------------------------------------------------------------
    # PyTorch NeuronX Training Images
    # -------------------------------------------------------------------------
    DLCImage(
        "pytorch",
        "2.9.0",
        "3.12",
        "neuronx",
        "sagemaker",
        "training",
        sdk_version="2.28.0",
        os_version="ubuntu24.04",
    ),
    DLCImage(
        "pytorch",
        "2.9.0",
        "3.12",
        "neuronx",
        "sagemaker",
        "training",
        sdk_version="2.27.1",
        os_version="ubuntu24.04",
    ),
    DLCImage("pytorch", "2.8.0", "3.11", "neuronx", "sagemaker", "training", sdk_version="2.26.1"),
    DLCImage("pytorch", "2.7.0", "3.10", "neuronx", "sagemaker", "training", sdk_version="2.25.0"),
    DLCImage("pytorch", "2.7.0", "3.10", "neuronx", "sagemaker", "training", sdk_version="2.24.1"),
    DLCImage("pytorch", "2.6.0", "3.10", "neuronx", "sagemaker", "training", sdk_version="2.23.0"),
    # -------------------------------------------------------------------------
    # PyTorch NeuronX Inference Images
    # -------------------------------------------------------------------------
    DLCImage(
        "pytorch",
        "2.9.0",
        "3.12",
        "neuronx",
        "sagemaker",
        "inference",
        sdk_version="2.28.0",
        os_version="ubuntu24.04",
    ),
    DLCImage(
        "pytorch",
        "2.9.0",
        "3.12",
        "neuronx",
        "sagemaker",
        "inference",
        sdk_version="2.27.1",
        os_version="ubuntu24.04",
    ),
    DLCImage("pytorch", "2.8.0", "3.11", "neuronx", "sagemaker", "inference", sdk_version="2.26.1"),
    DLCImage("pytorch", "2.7.0", "3.10", "neuronx", "sagemaker", "inference", sdk_version="2.25.0"),
    DLCImage("pytorch", "2.7.0", "3.10", "neuronx", "sagemaker", "inference", sdk_version="2.24.1"),
    DLCImage("pytorch", "2.6.0", "3.10", "neuronx", "sagemaker", "inference", sdk_version="2.23.0"),
    # -------------------------------------------------------------------------
    # HuggingFace PyTorch Training Images
    # -------------------------------------------------------------------------
    DLCImage(
        "huggingface-pytorch",
        "2.5.1",
        "3.11",
        "gpu",
        "sagemaker",
        "training",
        cuda_version="12.4",
        transformers_version="4.49.0",
        os_version="ubuntu22.04",
    ),
    DLCImage(
        "huggingface-pytorch",
        "2.1.0",
        "3.10",
        "gpu",
        "sagemaker",
        "training",
        cuda_version="12.1",
        transformers_version="4.36.0",
        os_version="ubuntu20.04",
    ),
    # -------------------------------------------------------------------------
    # HuggingFace PyTorch Inference Images
    # -------------------------------------------------------------------------
    DLCImage(
        "huggingface-pytorch",
        "2.6.0",
        "3.12",
        "gpu",
        "sagemaker",
        "inference",
        cuda_version="12.4",
        transformers_version="4.49.0",
    ),
    DLCImage(
        "huggingface-pytorch",
        "2.6.0",
        "3.12",
        "cpu",
        "sagemaker",
        "inference",
        transformers_version="4.49.0",
    ),
    DLCImage(
        "huggingface-pytorch",
        "2.1.0",
        "3.10",
        "gpu",
        "sagemaker",
        "inference",
        cuda_version="11.8",
        transformers_version="4.37.0",
        os_version="ubuntu20.04",
    ),
    DLCImage(
        "huggingface-pytorch",
        "2.1.0",
        "3.10",
        "cpu",
        "sagemaker",
        "inference",
        transformers_version="4.37.0",
    ),
    # -------------------------------------------------------------------------
    # HuggingFace NeuronX Images
    # -------------------------------------------------------------------------
    DLCImage(
        "huggingface-pytorch",
        "2.8.0",
        "3.10",
        "neuronx",
        "sagemaker",
        "training",
        sdk_version="2.26.0",
        transformers_version="4.55.4",
    ),
    DLCImage(
        "huggingface-pytorch",
        "2.7.0",
        "3.10",
        "neuronx",
        "sagemaker",
        "training",
        sdk_version="2.24.1",
        transformers_version="4.51.0",
    ),
    DLCImage(
        "huggingface-pytorch",
        "2.8.0",
        "3.10",
        "neuronx",
        "sagemaker",
        "inference",
        sdk_version="2.26.0",
        transformers_version="4.55.4",
    ),
    DLCImage(
        "huggingface-pytorch",
        "2.7.1",
        "3.10",
        "neuronx",
        "sagemaker",
        "inference",
        sdk_version="2.24.1",
        transformers_version="4.51.3",
    ),
    # -------------------------------------------------------------------------
    # HuggingFace vLLM NeuronX Inference
    # -------------------------------------------------------------------------
    DLCImage(
        "huggingface-vllm",
        "0.11.0",
        "3.10",
        "neuronx",
        "sagemaker",
        "inference",
        sdk_version="2.26.1",
        optimum_version="0.4.5",
    ),
    # -------------------------------------------------------------------------
    # AutoGluon Training Images
    # -------------------------------------------------------------------------
    DLCImage("autogluon", "1.5.0", "3.12", "gpu", "sagemaker", "training", cuda_version="12.6"),
    DLCImage("autogluon", "1.5.0", "3.12", "cpu", "sagemaker", "training"),
    DLCImage("autogluon", "1.4.0", "3.11", "gpu", "sagemaker", "training", cuda_version="12.4"),
    DLCImage("autogluon", "1.4.0", "3.11", "cpu", "sagemaker", "training"),
    # -------------------------------------------------------------------------
    # AutoGluon Inference Images
    # -------------------------------------------------------------------------
    DLCImage("autogluon", "1.5.0", "3.12", "gpu", "sagemaker", "inference", cuda_version="12.4"),
    DLCImage("autogluon", "1.5.0", "3.12", "cpu", "sagemaker", "inference"),
    DLCImage("autogluon", "1.4.0", "3.11", "gpu", "sagemaker", "inference", cuda_version="12.4"),
    DLCImage("autogluon", "1.4.0", "3.11", "cpu", "sagemaker", "inference"),
    # -------------------------------------------------------------------------
    # DJL Inference Images
    # -------------------------------------------------------------------------
    DLCImage(
        "djl-inference",
        "0.36.0",
        "3.10",
        "gpu",
        "sagemaker",
        "inference",
        cuda_version="12.8",
        engine_version="lmi20.0.0",
    ),
    DLCImage(
        "djl-inference",
        "0.36.0",
        "3.10",
        "gpu",
        "sagemaker",
        "inference",
        cuda_version="12.8",
        engine_version="lmi19.0.0",
    ),
    DLCImage(
        "djl-inference",
        "0.36.0",
        "3.10",
        "gpu",
        "sagemaker",
        "inference",
        cuda_version="12.8",
        engine_version="lmi18.0.0",
    ),
    DLCImage(
        "djl-inference",
        "0.36.0",
        "3.10",
        "cpu",
        "sagemaker",
        "inference",
        engine_version="cpu-full",
    ),
    DLCImage(
        "djl-inference",
        "0.35.0",
        "3.10",
        "gpu",
        "sagemaker",
        "inference",
        cuda_version="12.8",
        engine_version="lmi17.0.0",
    ),
    DLCImage(
        "djl-inference",
        "0.34.0",
        "3.10",
        "gpu",
        "sagemaker",
        "inference",
        cuda_version="12.8",
        engine_version="lmi16.0.0",
    ),
    DLCImage(
        "djl-inference",
        "0.33.0",
        "3.10",
        "gpu",
        "sagemaker",
        "inference",
        cuda_version="12.8",
        engine_version="lmi15.0.0",
    ),
    DLCImage(
        "djl-inference",
        "0.33.0",
        "3.10",
        "gpu",
        "sagemaker",
        "inference",
        cuda_version="12.8",
        engine_version="tensorrtllm0.21.0",
    ),
    # -------------------------------------------------------------------------
    # StabilityAI PyTorch Inference
    # -------------------------------------------------------------------------
    DLCImage(
        "stabilityai-pytorch",
        "2.0.1",
        "3.10",
        "gpu",
        "sagemaker",
        "inference",
        cuda_version="11.8",
        os_version="ubuntu20.04",
    ),
    # -------------------------------------------------------------------------
    # Base Images
    # -------------------------------------------------------------------------
    DLCImage("base", "13.0.2", "3.13", "gpu", "ec2", "training", cuda_version="13.0"),
    DLCImage("base", "13.0.0", "3.12", "gpu", "ec2", "training", cuda_version="13.0"),
    DLCImage("base", "12.9.1", "3.12", "gpu", "ec2", "training", cuda_version="12.9"),
    DLCImage(
        "base",
        "12.8.1",
        "3.12",
        "gpu",
        "ec2",
        "training",
        cuda_version="12.8",
        os_version="ubuntu24.04",
    ),
    DLCImage("base", "12.8.0", "3.12", "gpu", "ec2", "training", cuda_version="12.8"),
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_available_frameworks() -> List[str]:
    """Get list of all available frameworks."""
    return sorted(set(img.framework for img in DLC_IMAGES))


def get_available_versions(framework: str) -> List[str]:
    """Get available versions for a framework."""
    return sorted(
        set(img.version for img in DLC_IMAGES if img.framework == framework), reverse=True
    )


def get_available_platforms() -> List[str]:
    """Get list of all available platforms."""
    return sorted(set(img.platform for img in DLC_IMAGES))


def get_available_accelerators() -> List[str]:
    """Get list of all available accelerators."""
    return sorted(set(img.accelerator for img in DLC_IMAGES))


def filter_images(
    framework: Optional[str] = None,
    version: Optional[str] = None,
    python_version: Optional[str] = None,
    accelerator: Optional[str] = None,
    platform: Optional[str] = None,
    use_case: Optional[str] = None,
    architecture: Optional[str] = None,
) -> List[DLCImage]:
    """Filter DLC images based on criteria."""
    results = DLC_IMAGES
    if framework:
        results = [img for img in results if framework.lower() in img.framework.lower()]
    if version:
        results = [img for img in results if img.version.startswith(version)]
    if python_version:
        results = [img for img in results if img.python_version == python_version]
    if accelerator:
        results = [img for img in results if img.accelerator == accelerator.lower()]
    if platform:
        results = [img for img in results if img.platform == platform.lower()]
    if use_case:
        results = [img for img in results if img.use_case == use_case.lower()]
    if architecture:
        results = [img for img in results if img.architecture == architecture.lower()]
    return results


def get_latest_image(
    framework: str,
    use_case: str = "training",
    accelerator: str = "gpu",
    platform: str = "sagemaker",
) -> Optional[DLCImage]:
    """Get the latest image for a framework/use_case combination."""
    images = filter_images(
        framework=framework,
        use_case=use_case,
        accelerator=accelerator,
        platform=platform,
    )
    if not images:
        return None
    # Sort by version descending and return first
    return sorted(images, key=lambda x: x.version, reverse=True)[0]


def get_image_uri(
    framework: str,
    version: str,
    use_case: str,
    accelerator: str,
    platform: str,
    region: str = "us-west-2",
) -> Optional[str]:
    """Get the full URI for a specific image configuration."""
    images = filter_images(
        framework=framework,
        version=version,
        use_case=use_case,
        accelerator=accelerator,
        platform=platform,
    )
    if images:
        return images[0].get_full_uri(region)
    return None


def get_recommended_image_for_model(
    model_type: str,
    model_size: str = "medium",
    use_case: str = "inference",
) -> Optional[DLCImage]:
    """
    Get recommended DLC image based on model type and size.

    Args:
        model_type: Type of model (llm, vision, nlp, tabular, diffusion)
        model_size: Size category (small, medium, large, xlarge)
        use_case: training or inference
    """
    recommendations = {
        "llm": {
            "small": ("vllm", "gpu", "sagemaker"),
            "medium": ("vllm", "gpu", "sagemaker"),
            "large": ("djl-inference", "gpu", "sagemaker"),
            "xlarge": ("pytorch", "neuronx", "sagemaker"),
        },
        "vision": {
            "small": ("pytorch", "gpu", "sagemaker"),
            "medium": ("pytorch", "gpu", "sagemaker"),
            "large": ("pytorch", "gpu", "sagemaker"),
        },
        "nlp": {
            "small": ("huggingface-pytorch", "gpu", "sagemaker"),
            "medium": ("huggingface-pytorch", "gpu", "sagemaker"),
            "large": ("vllm", "gpu", "sagemaker"),
        },
        "tabular": {
            "small": ("autogluon", "cpu", "sagemaker"),
            "medium": ("autogluon", "gpu", "sagemaker"),
            "large": ("autogluon", "gpu", "sagemaker"),
        },
        "diffusion": {
            "small": ("stabilityai-pytorch", "gpu", "sagemaker"),
            "medium": ("stabilityai-pytorch", "gpu", "sagemaker"),
            "large": ("pytorch", "gpu", "sagemaker"),
        },
    }

    if model_type not in recommendations:
        return None

    size_config = recommendations[model_type].get(model_size, recommendations[model_type]["medium"])
    framework, accelerator, platform = size_config

    return get_latest_image(framework, use_case, accelerator, platform)


def get_ecr_account_for_region(region: str) -> str:
    """Get the ECR account ID for a specific region."""
    return REGION_ACCOUNT_MAP.get(region, AWS_DLC_ACCOUNT_ID)


def is_neuron_supported_in_region(region: str) -> bool:
    """Check if Neuron images are available in a region."""
    return region in NEURON_SUPPORTED_REGIONS
