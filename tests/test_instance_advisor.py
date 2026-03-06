###
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
######

"""Tests for the instance advisor module."""

import unittest

from aws_samples.dlc_mcp_server.modules.instance_advisor import (
    get_instance_recommendation,
    estimate_training_cost,
    list_gpu_instances,
    INSTANCE_TYPES,
)


class TestInstanceAdvisor(unittest.TestCase):
    """Test cases for the instance advisor module."""

    def test_get_instance_recommendation_small_model(self):
        """Test recommendation for small model."""
        result = get_instance_recommendation(
            model_size_gb=7,  # ~7B parameter model
            use_case="inference",
        )
        self.assertTrue(result["success"])
        self.assertGreater(len(result["recommendations"]), 0)

        # Should recommend a GPU instance
        gpu_rec = result["recommendations"][0]
        self.assertIn("gpu_memory_gb", gpu_rec)
        self.assertGreaterEqual(gpu_rec["gpu_memory_gb"], 7 * 1.5)

    def test_get_instance_recommendation_large_model(self):
        """Test recommendation for large model requiring multi-GPU."""
        result = get_instance_recommendation(
            model_size_gb=70,  # ~70B parameter model
            use_case="inference",
        )
        self.assertTrue(result["success"])

        # Should include multi-GPU recommendation
        categories = [r["category"] for r in result["recommendations"]]
        self.assertTrue(any("Multi-GPU" in c for c in categories))

    def test_get_instance_recommendation_with_budget(self):
        """Test recommendation with budget constraint."""
        result = get_instance_recommendation(
            model_size_gb=14,
            use_case="inference",
            budget_per_hour=5.0,
        )
        self.assertTrue(result["success"])

        for rec in result["recommendations"]:
            self.assertLessEqual(rec["price_per_hour"], 5.0)

    def test_get_instance_recommendation_training(self):
        """Test recommendation for training workload."""
        result = get_instance_recommendation(
            model_size_gb=14,
            use_case="training",
        )
        self.assertTrue(result["success"])
        self.assertGreater(len(result["recommendations"]), 0)

    def test_estimate_training_cost(self):
        """Test training cost estimation."""
        result = estimate_training_cost(
            model_size_gb=7,
            dataset_size_gb=100,
            epochs=3,
            batch_size=32,
        )
        self.assertTrue(result["success"])
        self.assertIn("cost_estimates", result)
        self.assertIn("on_demand", result["cost_estimates"])
        self.assertIn("spot_estimated", result["cost_estimates"])

        # Spot should be cheaper than on-demand
        self.assertLess(
            result["cost_estimates"]["spot_estimated"], result["cost_estimates"]["on_demand"]
        )

    def test_estimate_training_cost_with_instance(self):
        """Test training cost with specific instance type."""
        result = estimate_training_cost(
            model_size_gb=7,
            dataset_size_gb=50,
            epochs=5,
            batch_size=16,
            instance_type="ml.g5.xlarge",
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["instance_type"], "ml.g5.xlarge")

    def test_list_gpu_instances_no_filter(self):
        """Test listing all GPU instances."""
        result = list_gpu_instances()
        self.assertTrue(result["success"])
        self.assertGreater(result["total_instances"], 0)

        # All should have GPUs
        for inst in result["instances"]:
            self.assertGreater(inst["gpu_count"], 0)

    def test_list_gpu_instances_with_memory_filter(self):
        """Test listing GPU instances with memory filter."""
        result = list_gpu_instances(min_gpu_memory=80)
        self.assertTrue(result["success"])

        for inst in result["instances"]:
            self.assertGreaterEqual(inst["gpu_memory_gb"], 80)

    def test_list_gpu_instances_with_price_filter(self):
        """Test listing GPU instances with price filter."""
        result = list_gpu_instances(max_price_per_hour=2.0)
        self.assertTrue(result["success"])

        for inst in result["instances"]:
            self.assertLessEqual(inst["price_per_hour"], 2.0)

    def test_list_gpu_instances_with_gpu_type(self):
        """Test listing GPU instances by GPU type."""
        result = list_gpu_instances(gpu_type="A100")
        self.assertTrue(result["success"])

        for inst in result["instances"]:
            self.assertIn("A100", inst["gpu_type"])

    def test_instance_types_catalog(self):
        """Test that instance types catalog is populated."""
        self.assertGreater(len(INSTANCE_TYPES), 20)

        # Check for different accelerator types
        accelerators = set(i.accelerator for i in INSTANCE_TYPES)
        self.assertIn("gpu", accelerators)
        self.assertIn("cpu", accelerators)
        self.assertIn("neuron", accelerators)

    def test_neuron_instances_available(self):
        """Test that Neuron instances are in the catalog."""
        result = list_gpu_instances(gpu_type="Inferentia")
        self.assertTrue(result["success"])
        self.assertGreater(result["total_instances"], 0)


if __name__ == "__main__":
    unittest.main()
