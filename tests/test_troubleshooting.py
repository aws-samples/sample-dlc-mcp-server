"""Tests for the troubleshooting module."""

import unittest

from awslabs.dlc_mcp_server.modules.troubleshooting import (
    diagnose_common_issues,
    get_framework_compatibility_info,
    get_performance_optimization_tips
)


class TestTroubleshooting(unittest.TestCase):
    """Test cases for the troubleshooting module."""

    def test_diagnose_common_issues(self):
        """Test diagnosing common issues."""
        # Test CUDA out of memory error
        result = diagnose_common_issues("CUDA out of memory")
        self.assertTrue(result["matched"])
        self.assertIn("matches", result)
        self.assertTrue(any("CUDA out of memory" in match["diagnosis"] for match in result["matches"]))
        
        # Test with framework-specific error
        result = diagnose_common_issues(
            "InvalidArgumentError: Invalid argument",
            framework="tensorflow",
            use_case="training"
        )
        self.assertTrue(result["matched"])
        self.assertIn("matches", result)
        
        # Test with unknown error
        result = diagnose_common_issues("This is a completely unknown error")
        self.assertFalse(result["matched"])
        self.assertIn("general_recommendations", result)

    def test_get_framework_compatibility_info(self):
        """Test getting framework compatibility information."""
        # Test valid framework and version
        result = get_framework_compatibility_info("pytorch", "2.6.0")
        self.assertEqual(result["framework"], "pytorch")
        self.assertEqual(result["version"], "2.6.0")
        self.assertIn("compatibility", result)
        self.assertIn("python_versions", result["compatibility"])
        self.assertIn("cuda_versions", result["compatibility"])
        
        # Test invalid framework
        result = get_framework_compatibility_info("invalid_framework", "1.0.0")
        self.assertIn("error", result)
        self.assertIn("available_frameworks", result)
        
        # Test invalid version
        result = get_framework_compatibility_info("pytorch", "999.0.0")
        self.assertIn("error", result)
        self.assertIn("available_versions", result)

    def test_get_performance_optimization_tips(self):
        """Test getting performance optimization tips."""
        # Test valid parameters
        result = get_performance_optimization_tips("pytorch", "training", "gpu")
        self.assertEqual(result["framework"], "pytorch")
        self.assertEqual(result["use_case"], "training")
        self.assertEqual(result["device_type"], "gpu")
        self.assertIn("common_tips", result)
        self.assertIn("specific_tips", result)
        
        # Test invalid framework
        result = get_performance_optimization_tips("invalid_framework", "training", "gpu")
        self.assertIn("error", result)
        self.assertIn("available_frameworks", result)
        
        # Test invalid use case
        result = get_performance_optimization_tips("pytorch", "invalid_use_case", "gpu")
        self.assertIn("error", result)
        self.assertIn("available_use_cases", result)
        
        # Test invalid device type
        result = get_performance_optimization_tips("pytorch", "training", "invalid_device")
        self.assertIn("error", result)
        self.assertIn("available_device_types", result)


if __name__ == "__main__":
    unittest.main()
