###
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
######

"""
DLC Image Registry - Dynamic catalog of AWS Deep Learning Container images.
Fetches latest images from: https://aws.github.io/deep-learning-containers/reference/available_images/
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.request import urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)

# AWS DLC Production Account ID (primary)
AWS_DLC_ACCOUNT_ID = "763104351884"

# GitHub page URL for available images
DLC_IMAGES_URL = "https://aws.github.io/deep-learning-containers/reference/available_images/"

# Cache settings
_cache: Dict = {"images": [], "timestamp": 0, "regions": {}, "neuron_regions": []}
CACHE_TTL_SECONDS = 3600  # 1 hour cache

# Region-specific account IDs (fallback if fetch fails)
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
    image_uri_template: Optional[str] = None  # Store the original URI template

    def get_repository_name(self) -> str:
        """Get the ECR repository name for this image."""
        if self.image_uri_template:
            # Extract repo name from template
            match = re.search(r"\.amazonaws\.com(?:\.cn)?/([^:]+):", self.image_uri_template)
            if match:
                return match.group(1)

        base = self.framework
        if self.use_case and self.framework not in ["base", "sglang", "vllm", "djl-inference"]:
            base = f"{self.framework}-{self.use_case}"
        if self.architecture == "arm64":
            base = f"{base}-arm64"
        if self.accelerator == "neuronx":
            base = base.replace("-inference", "-inference-neuronx").replace(
                "-training", "-training-neuronx"
            )
            if "neuronx" not in base:
                base = f"{base}-neuronx"
        return base

    def get_image_tag(self) -> str:
        """Generate the image tag based on image properties."""
        if self.image_uri_template:
            # Extract tag from template
            match = re.search(
                r":([^<]+)$", self.image_uri_template.replace("<region>", "us-west-2")
            )
            if match:
                return match.group(1)

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
        if self.image_uri_template:
            account_id = get_ecr_account_for_region(region)
            suffix = ".amazonaws.com.cn" if region.startswith("cn-") else ".amazonaws.com"
            # Replace <region> placeholder and update account ID
            uri = self.image_uri_template.replace("<region>", region)
            uri = re.sub(r"\d+\.dkr\.ecr\.", f"{account_id}.dkr.ecr.", uri)
            return uri

        account_id = get_ecr_account_for_region(region)
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
# DYNAMIC FETCHING FROM GITHUB
# =============================================================================


def _fetch_dlc_page() -> str:
    """Fetch the DLC available images page from GitHub."""
    try:
        with urlopen(DLC_IMAGES_URL, timeout=30) as response:
            return response.read().decode("utf-8")
    except URLError as e:
        logger.warning(f"Failed to fetch DLC images page: {e}")
        return ""
    except Exception as e:
        logger.warning(f"Error fetching DLC images: {e}")
        return ""


def _parse_image_uri(uri_template: str) -> Optional[DLCImage]:
    """Parse an image URI template into a DLCImage object."""
    # Example: 763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker

    # Extract repository and tag
    match = re.search(r"/([^:]+):(.+)$", uri_template)
    if not match:
        return None

    repo_name = match.group(1)
    tag = match.group(2)

    # Parse repository name for framework and use_case
    framework = repo_name
    use_case = "training"
    architecture = "x86_64"

    if "-training" in repo_name:
        use_case = "training"
        framework = (
            repo_name.replace("-training-neuronx", "")
            .replace("-training-arm64", "")
            .replace("-training", "")
        )
    elif "-inference" in repo_name:
        use_case = "inference"
        framework = (
            repo_name.replace("-inference-neuronx", "")
            .replace("-inference-arm64", "")
            .replace("-inference", "")
        )

    if "-arm64" in repo_name:
        architecture = "arm64"

    # Parse tag for version, accelerator, python, cuda, etc.
    # Examples:
    # 2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker
    # 2.9.0-cpu-py312-ubuntu22.04-ec2
    # 2.9.0-neuronx-py312-sdk2.28.0-ubuntu24.04
    # 0.15.1-gpu-py312-cu129-ubuntu22.04-sagemaker

    version = ""
    accelerator = "gpu"
    python_version = "3.12"
    cuda_version = None
    sdk_version = None
    os_version = "ubuntu22.04"
    platform = "sagemaker"
    transformers_version = None
    optimum_version = None
    engine_version = None

    # Extract version (first part before -)
    version_match = re.match(r"^([\d.]+)", tag)
    if version_match:
        version = version_match.group(1)

    # Detect accelerator
    if "-neuronx-" in tag or "neuronx" in repo_name:
        accelerator = "neuronx"
    elif "-cpu-" in tag:
        accelerator = "cpu"
    else:
        accelerator = "gpu"

    # Extract Python version
    py_match = re.search(r"py(\d+)", tag)
    if py_match:
        py_num = py_match.group(1)
        if len(py_num) >= 2:
            python_version = f"{py_num[0]}.{py_num[1:]}"

    # Extract CUDA version
    cuda_match = re.search(r"cu(\d+)", tag)
    if cuda_match:
        cuda_num = cuda_match.group(1)
        if len(cuda_num) >= 2:
            cuda_version = f"{cuda_num[:-1]}.{cuda_num[-1]}"

    # Extract SDK version (for NeuronX)
    sdk_match = re.search(r"sdk([\d.]+)", tag)
    if sdk_match:
        sdk_version = sdk_match.group(1)

    # Extract OS version
    os_match = re.search(r"(ubuntu\d+\.\d+)", tag)
    if os_match:
        os_version = os_match.group(1)

    # Extract platform
    if "-sagemaker" in tag:
        platform = "sagemaker"
    elif "-ec2" in tag:
        platform = "ec2"

    # Extract transformers version (for HuggingFace)
    tf_match = re.search(r"transformers([\d.]+)", tag)
    if tf_match:
        transformers_version = tf_match.group(1)

    # Extract optimum version
    opt_match = re.search(r"optimum([\d.]+)", tag)
    if opt_match:
        optimum_version = opt_match.group(1)

    # Extract engine version (for DJL)
    engine_match = re.search(r"(lmi[\d.]+|tensorrtllm[\d.]+|cpu-full)", tag)
    if engine_match:
        engine_version = engine_match.group(1)

    # Handle special frameworks
    if "huggingface-pytorch" in repo_name:
        framework = "huggingface-pytorch"
    elif "huggingface-vllm" in repo_name:
        framework = "huggingface-vllm"
    elif "huggingface-tensorflow" in repo_name:
        framework = "huggingface-tensorflow"
    elif "stabilityai" in repo_name:
        framework = "stabilityai-pytorch"
    elif "autogluon" in repo_name:
        framework = "autogluon"
    elif "djl-inference" in repo_name:
        framework = "djl-inference"
        use_case = "inference"
    elif repo_name in ["vllm", "vllm-arm64"]:
        framework = "vllm"
        use_case = "inference"
    elif repo_name == "sglang":
        framework = "sglang"
        use_case = "inference"
    elif repo_name == "base":
        framework = "base"
        use_case = "training"

    return DLCImage(
        framework=framework,
        version=version,
        python_version=python_version,
        accelerator=accelerator,
        platform=platform,
        use_case=use_case,
        cuda_version=cuda_version,
        sdk_version=sdk_version,
        transformers_version=transformers_version,
        optimum_version=optimum_version,
        engine_version=engine_version,
        os_version=os_version,
        architecture=architecture,
        image_uri_template=uri_template,
    )


def _parse_dlc_page(html: str) -> List[DLCImage]:
    """Parse the DLC available images HTML page and extract image information."""
    images = []

    # The HTML has escaped characters: &lt;region&gt; instead of <region>
    # Pattern: account.dkr.ecr.&lt;region&gt;.amazonaws.com/repo:tag
    uri_pattern = r'(\d+\.dkr\.ecr\.&lt;region&gt;\.amazonaws\.com(?:\.cn)?/[^<\s"&]+:[^<\s"&]+)'

    matches = re.findall(uri_pattern, html)
    seen_uris = set()

    for uri in matches:
        # Convert escaped HTML back to normal
        uri = uri.replace("&lt;", "<").replace("&gt;", ">")

        if uri in seen_uris:
            continue
        seen_uris.add(uri)

        image = _parse_image_uri(uri)
        if image and image.version:
            images.append(image)

    return images


def _parse_region_info(html: str) -> tuple:
    """Parse region availability information from the HTML page."""
    regions = {}
    neuron_regions = []

    # Pattern to find region rows in the table
    # Looking for: region code, account ID, and neuron support
    region_pattern = r"(\w{2}-\w+-\d+)[^✅❌]*?(✅|❌)[^✅❌]*?(✅|❌)[^<]*?(\d+)\.dkr\.ecr\."

    for match in re.finditer(region_pattern, html):
        region = match.group(1)
        general_support = match.group(2) == "✅"
        neuron_support = match.group(3) == "✅"
        account_id = match.group(4)

        if general_support:
            regions[region] = account_id
            if neuron_support:
                neuron_regions.append(region)

    return regions, neuron_regions


def _refresh_cache() -> None:
    """Refresh the image cache from the GitHub page."""
    global _cache

    html = _fetch_dlc_page()
    if not html:
        logger.warning("Failed to fetch DLC page, using cached data")
        return

    images = _parse_dlc_page(html)
    regions, neuron_regions = _parse_region_info(html)

    if images:
        _cache["images"] = images
        _cache["timestamp"] = time.time()
        logger.info(f"Refreshed DLC cache with {len(images)} images")

    if regions:
        _cache["regions"] = regions
        REGION_ACCOUNT_MAP.update(regions)

    if neuron_regions:
        _cache["neuron_regions"] = neuron_regions


def _ensure_cache() -> None:
    """Ensure the cache is populated and fresh."""
    global _cache

    current_time = time.time()
    if not _cache["images"] or (current_time - _cache["timestamp"]) > CACHE_TTL_SECONDS:
        _refresh_cache()


# =============================================================================
# PUBLIC API FUNCTIONS
# =============================================================================


def get_dlc_images() -> List[DLCImage]:
    """Get all DLC images, fetching from GitHub if needed."""
    _ensure_cache()
    return _cache["images"]


def get_available_frameworks() -> List[str]:
    """Get list of all available frameworks."""
    images = get_dlc_images()
    return sorted(set(img.framework for img in images))


def get_available_versions(framework: str) -> List[str]:
    """Get available versions for a framework."""
    images = get_dlc_images()
    return sorted(set(img.version for img in images if img.framework == framework), reverse=True)


def get_available_platforms() -> List[str]:
    """Get list of all available platforms."""
    images = get_dlc_images()
    return sorted(set(img.platform for img in images))


def get_available_accelerators() -> List[str]:
    """Get list of all available accelerators."""
    images = get_dlc_images()
    return sorted(set(img.accelerator for img in images))


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
    results = get_dlc_images()

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
    _ensure_cache()
    if _cache["regions"]:
        return _cache["regions"].get(region, AWS_DLC_ACCOUNT_ID)
    return REGION_ACCOUNT_MAP.get(region, AWS_DLC_ACCOUNT_ID)


def is_neuron_supported_in_region(region: str) -> bool:
    """Check if Neuron images are available in a region."""
    _ensure_cache()
    if _cache["neuron_regions"]:
        return region in _cache["neuron_regions"]
    return region in NEURON_SUPPORTED_REGIONS


def refresh_images() -> Dict:
    """Force refresh the image cache and return status."""
    global _cache
    _cache["timestamp"] = 0  # Force refresh
    _ensure_cache()
    return {
        "success": True,
        "images_count": len(_cache["images"]),
        "regions_count": len(_cache.get("regions", {})),
        "neuron_regions_count": len(_cache.get("neuron_regions", [])),
    }
