###
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
######

"""Tests for the troubleshooting module."""

import unittest
import asyncio

from aws_samples.dlc_mcp_server.modules.troubleshooting import (
    analyze_error,
    get_framework_compatibility_info,
    _extract_context,
    _categorize_error,
    _build_search_query,
    ErrorCategory,
    DLC_RESOLVER_GROUPS,
)

# Note: get_performance_optimization_tips removed - use get_framework_specific_best_practices from best_practices module


class TestErrorAnalysis(unittest.TestCase):
    """Test cases for error analysis functions."""

    def test_extract_context_cuda_oom(self):
        """Test context extraction from CUDA OOM error."""
        error_log = """
        RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB 
        (GPU 0; 15.78 GiB total capacity; 14.23 GiB already allocated)
        batch_size=32
        """
        context = _extract_context(error_log)
        self.assertIsNotNone(context.gpu_memory_used)
        self.assertEqual(context.batch_size, 32)

    def test_extract_context_import_error(self):
        """Test context extraction from import error."""
        error_log = "ModuleNotFoundError: No module named 'transformers'"
        context = _extract_context(error_log)
        self.assertIn("ModuleNotFoundError", context.key_error_message)

    def test_categorize_error_oom(self):
        """Test error categorization for OOM."""
        categories = _categorize_error("CUDA out of memory. Tried to allocate 2GB")
        self.assertIn(ErrorCategory.CUDA_OOM, categories)

    def test_categorize_error_import(self):
        """Test error categorization for import error."""
        categories = _categorize_error("ModuleNotFoundError: No module named 'torch'")
        self.assertIn(ErrorCategory.IMPORT_ERROR, categories)

    def test_categorize_error_shape(self):
        """Test error categorization for shape mismatch."""
        categories = _categorize_error("RuntimeError: shape mismatch, expected [32, 768]")
        self.assertIn(ErrorCategory.SHAPE_MISMATCH, categories)

    def test_categorize_error_distributed(self):
        """Test error categorization for distributed training."""
        categories = _categorize_error("NCCL error: unhandled system error, rank 0")
        self.assertIn(ErrorCategory.DISTRIBUTED, categories)

    def test_build_search_query(self):
        """Test search query building."""
        query = _build_search_query("CUDA out of memory", [ErrorCategory.CUDA_OOM])
        self.assertIn("cuda out of memory", query.lower())


class TestAnalyzeError(unittest.TestCase):
    """Test cases for the analyze_error function."""

    def test_analyze_cuda_oom(self):
        """Test analysis of CUDA OOM error."""
        error_log = """
        torch.cuda.OutOfMemoryError: CUDA out of memory. 
        Tried to allocate 4.00 GiB (GPU 0; 24.00 GiB total capacity)
        """
        result = asyncio.run(analyze_error(error_log, framework="pytorch"))

        self.assertTrue(result["success"])
        self.assertIn("cuda out of memory", result["error_categories"])
        self.assertIn("resolution", result)

    def test_analyze_import_error(self):
        """Test analysis of import error."""
        error_log = "ModuleNotFoundError: No module named 'bitsandbytes'"
        result = asyncio.run(analyze_error(error_log))

        self.assertTrue(result["success"])
        self.assertIn("import error module not found", result["error_categories"])
        self.assertTrue(any("pip" in step.lower() for step in result["resolution"]["steps"]))

    def test_analyze_with_framework(self):
        """Test analysis with framework specified."""
        result = asyncio.run(analyze_error("CUDA error", framework="pytorch"))
        self.assertEqual(result["extracted_context"]["framework"], "pytorch")


class TestFrameworkCompatibility(unittest.TestCase):
    """Test cases for framework compatibility info."""

    def test_pytorch_compatibility(self):
        """Test PyTorch compatibility info."""
        result = get_framework_compatibility_info("pytorch", "2.6.0")
        self.assertTrue(result["success"])
        self.assertIn("cuda_versions", result["compatibility"])

    def test_tensorflow_compatibility(self):
        """Test TensorFlow compatibility info."""
        result = get_framework_compatibility_info("tensorflow", "2.18.0")
        self.assertTrue(result["success"])

    def test_unknown_framework(self):
        """Test unknown framework handling."""
        result = get_framework_compatibility_info("unknown", "1.0")
        self.assertFalse(result["success"])

    def test_unknown_version(self):
        """Test unknown version handling."""
        result = get_framework_compatibility_info("pytorch", "0.0.1")
        self.assertFalse(result["success"])


# TestPerformanceTips removed - functionality moved to best_practices module
# Use get_framework_specific_best_practices instead


class TestResolverGroups(unittest.TestCase):
    """Test resolver groups configuration."""

    def test_resolver_groups_defined(self):
        """Test that DLC resolver groups are defined."""
        self.assertGreater(len(DLC_RESOLVER_GROUPS), 0)
        self.assertIn("DLC Customer Issues", DLC_RESOLVER_GROUPS)
        self.assertIn("DLC General", DLC_RESOLVER_GROUPS)


if __name__ == "__main__":
    unittest.main()
