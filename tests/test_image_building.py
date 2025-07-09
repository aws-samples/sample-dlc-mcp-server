###
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# Copyright Amazon.com, Inc. and its affiliates. All Rights Reserved.
#   SPDX-License-Identifier: MIT
######

"""Tests for the image building module."""

import unittest
from unittest.mock import patch, MagicMock

from aws_samples.dlc_mcp_server.modules.image_building import (
    list_base_images,
    create_custom_dockerfile,
    build_custom_dlc_image,
)

from aws_samples.dlc_mcp_server.modules.containers import list_available_dlc_images


class TestImageBuilding(unittest.TestCase):
    """Test cases for the image building module."""

    def test_list_base_images(self):
        """Test listing base images."""
        result = list_base_images()
        self.assertIn("images", result)
        self.assertIsInstance(result["images"], list)

        # Test with filters
        result = list_base_images(framework="pytorch", use_case="training")
        self.assertIn("images", result)
        for image in result["images"]:
            self.assertEqual(image["framework"], "pytorch")
            self.assertEqual(image["use_case"], "training")

    def test_dlc_images(self):
        result = list_available_dlc_images()
        print(result)

    def test_create_custom_dockerfile(self):
        """Test creating a custom Dockerfile."""
        base_image = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-gpu-py312-cu126-ubuntu22.04-ec2"
        packages = ["git", "wget"]
        python_packages = ["transformers", "datasets"]
        custom_commands = ["RUN echo 'Custom command'"]
        env_vars = {"PYTHONPATH": "/opt/ml/code"}

        result = create_custom_dockerfile(
            base_image=base_image,
            packages=packages,
            python_packages=python_packages,
            custom_commands=custom_commands,
            environment_variables=env_vars,
        )

        self.assertIn("dockerfile_content", result)
        content = result["dockerfile_content"]

        # Check if base image is included
        self.assertIn(f"FROM {base_image}", content)

        # Check if packages are included
        for package in packages:
            self.assertIn(package, content)

        # Check if Python packages are included
        for package in python_packages:
            self.assertIn(package, content)

        # Check if custom commands are included
        for command in custom_commands:
            self.assertIn(command, content)

        # Check if environment variables are included
        for key, value in env_vars.items():
            self.assertIn(f"ENV {key}={value}", content)

    @patch("aws_samples.dlc_mcp_server.modules.image_building.pull_image")
    @patch("aws_samples.dlc_mcp_server.modules.image_building.build_image")
    @patch("aws_samples.dlc_mcp_server.modules.image_building.create_ecr_repository")
    @patch("aws_samples.dlc_mcp_server.modules.image_building.push_image")
    def test_build_custom_dlc_image(self, mock_push, mock_create_repo, mock_build, mock_pull):
        """Test building a custom DLC image."""
        # Mock successful responses
        mock_pull.return_value = {"success": True, "image_id": "sha256:123"}
        mock_build.return_value = {
            "success": True,
            "image_id": "sha256:456",
            "logs": ["Building..."],
        }
        mock_create_repo.return_value = {
            "success": True,
            "repository_uri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/test-repo",
        }
        mock_push.return_value = {"success": True, "logs": ["Pushing..."]}

        # Test building without pushing to ECR
        result = build_custom_dlc_image(
            base_image="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-gpu-py312-cu126-ubuntu22.04-ec2",
            repository_name="test-repo",
            tag="latest",
            dockerfile_content="FROM pytorch:latest\nRUN echo 'test'",
            push_to_ecr=False,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["local_tag"], "test-repo:latest")
        mock_pull.assert_called_once()
        mock_build.assert_called_once()
        mock_create_repo.assert_not_called()
        mock_push.assert_not_called()

        # Reset mocks
        mock_pull.reset_mock()
        mock_build.reset_mock()

        # Test building with pushing to ECR
        result = build_custom_dlc_image(
            base_image="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-gpu-py312-cu126-ubuntu22.04-ec2",
            repository_name="test-repo",
            tag="latest",
            dockerfile_content="FROM pytorch:latest\nRUN echo 'test'",
            push_to_ecr=True,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["local_tag"], "test-repo:latest")
        self.assertIn("ecr_uri", result)
        mock_pull.assert_called_once()
        mock_build.assert_called_once()
        mock_create_repo.assert_called_once()
        mock_push.assert_called_once()


if __name__ == "__main__":
    unittest.main()
