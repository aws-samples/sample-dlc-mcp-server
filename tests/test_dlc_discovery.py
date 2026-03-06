###
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
######

"""Tests for the DLC discovery module."""

import unittest

from aws_samples.dlc_mcp_server.modules.dlc_discovery import (
    search_dlc_images,
    get_image_recommendation,
    list_frameworks,
    get_region_info,
    get_llm_serving_options,
)
from aws_samples.dlc_mcp_server.utils.dlc_images import (
    filter_images,
    get_latest_image,
    get_available_frameworks,
    get_dlc_images,
)


class TestDLCDiscovery(unittest.TestCase):
    """Test cases for the DLC discovery module."""

    def test_list_frameworks(self):
        """Test listing available frameworks."""
        result = list_frameworks()
        self.assertTrue(result["success"])
        self.assertGreater(result["total_frameworks"], 0)

        # Check that common frameworks are present
        framework_names = [f["framework"] for f in result["frameworks"]]
        self.assertIn("pytorch", framework_names)
        self.assertIn("tensorflow", framework_names)
        self.assertIn("vllm", framework_names)

    def test_search_dlc_images_no_filter(self):
        """Test searching images without filters."""
        result = search_dlc_images()
        self.assertTrue(result["success"])
        self.assertGreater(result["total_matches"], 0)

    def test_search_dlc_images_with_framework(self):
        """Test searching images by framework."""
        result = search_dlc_images(framework="pytorch")
        self.assertTrue(result["success"])
        for img in result["images"]:
            self.assertIn("pytorch", img["framework"].lower())

    def test_search_dlc_images_with_multiple_filters(self):
        """Test searching with multiple filters."""
        result = search_dlc_images(
            framework="pytorch",
            use_case="training",
            accelerator="gpu",
            platform="sagemaker",
        )
        self.assertTrue(result["success"])
        for img in result["images"]:
            self.assertEqual(img["use_case"], "training")
            self.assertEqual(img["accelerator"], "gpu")
            self.assertEqual(img["platform"], "sagemaker")

    def test_get_image_recommendation_llm(self):
        """Test getting recommendation for LLM workload."""
        result = get_image_recommendation(
            model_type="llm",
            model_size="medium",
            use_case="inference",
        )
        self.assertTrue(result["success"])
        self.assertIn("recommendation", result)
        self.assertIn("image_uri", result["recommendation"])
        self.assertIn("explanation", result["recommendation"])

    def test_get_image_recommendation_invalid_type(self):
        """Test recommendation with invalid model type."""
        result = get_image_recommendation(
            model_type="invalid_type",
            model_size="medium",
        )
        self.assertFalse(result["success"])
        self.assertIn("available_model_types", result)

    def test_get_region_info(self):
        """Test getting region information."""
        result = get_region_info("us-west-2")
        self.assertTrue(result["success"])
        self.assertEqual(result["region"], "us-west-2")
        self.assertEqual(result["ecr_account_id"], "763104351884")
        self.assertTrue(result["neuron_supported"])
        self.assertIn("ecr_login_command", result)

    def test_get_region_info_secondary_region(self):
        """Test region info for secondary region with different account."""
        result = get_region_info("af-south-1")
        self.assertTrue(result["success"])
        self.assertEqual(result["ecr_account_id"], "626614931356")
        self.assertFalse(result["neuron_supported"])

    def test_get_llm_serving_options(self):
        """Test getting LLM serving options."""
        result = get_llm_serving_options()
        self.assertTrue(result["success"])
        self.assertGreater(len(result["options"]), 0)

        # Check that vLLM is in options
        container_names = [opt["container"] for opt in result["options"]]
        self.assertIn("vLLM", container_names)

    def test_filter_images_helper(self):
        """Test the filter_images helper function."""
        images = filter_images(framework="vllm")
        self.assertGreater(len(images), 0)
        for img in images:
            self.assertIn("vllm", img.framework.lower())

    def test_get_latest_image(self):
        """Test getting latest image for a framework."""
        image = get_latest_image("pytorch", "training", "gpu", "sagemaker")
        self.assertIsNotNone(image)
        self.assertEqual(image.framework, "pytorch")
        self.assertEqual(image.use_case, "training")

    def test_dlc_images_catalog_not_empty(self):
        """Test that DLC images catalog is populated."""
        images = get_dlc_images()
        self.assertGreater(len(images), 10)

    def test_available_frameworks(self):
        """Test getting available frameworks."""
        frameworks = get_available_frameworks()
        self.assertIn("pytorch", frameworks)
        self.assertIn("tensorflow", frameworks)
        self.assertIn("vllm", frameworks)
        self.assertIn("sglang", frameworks)


class TestDLCImageURIs(unittest.TestCase):
    """Test cases for DLC image URI generation."""

    def test_image_uri_format(self):
        """Test that image URIs are properly formatted."""
        result = search_dlc_images(framework="pytorch", use_case="training", region="us-east-1")
        self.assertTrue(result["success"])

        for img in result["images"]:
            uri = img["image_uri"]
            self.assertIn("763104351884.dkr.ecr.us-east-1.amazonaws.com", uri)
            self.assertIn("pytorch", uri)

    def test_china_region_uri(self):
        """Test URI format for China regions."""
        result = get_region_info("cn-north-1")
        self.assertTrue(result["success"])
        self.assertIn(".amazonaws.com.cn", result["ecr_registry"])


if __name__ == "__main__":
    unittest.main()
