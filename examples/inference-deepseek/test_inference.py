#!/usr/bin/env python3
"""
Test script for DeepSeek inference server
"""

import requests
import json
import time

def test_inference_server(base_url="http://localhost:8080"):
    """Test the DeepSeek inference server"""
    
    print("Testing DeepSeek Inference Server...")
    print(f"Base URL: {base_url}")
    
    # Test health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/ping")
        print(f"Health check status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print("❌ Health check failed")
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return
    
    # Test root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(base_url)
        print(f"Root endpoint status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint response: {data}")
        else:
            print("❌ Root endpoint failed")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    # Test inference
    print("\n3. Testing inference...")
    test_prompts = [
        {
            "inputs": "Write a Python function to calculate fibonacci numbers",
            "max_length": 512,
            "temperature": 0.7
        },
        {
            "inputs": "Explain what is machine learning in simple terms",
            "max_length": 256,
            "temperature": 0.5
        },
        {
            "inputs": "Create a simple REST API using Flask",
            "max_length": 1024,
            "temperature": 0.8
        }
    ]
    
    for i, prompt_data in enumerate(test_prompts, 1):
        print(f"\n3.{i} Testing prompt: '{prompt_data['inputs'][:50]}...'")
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/invocations",
                json=prompt_data,
                headers={"Content-Type": "application/json"},
                timeout=120  # 2 minutes timeout
            )
            end_time = time.time()
            
            print(f"Response status: {response.status_code}")
            print(f"Response time: {end_time - start_time:.2f} seconds")
            
            if response.status_code == 200:
                result = response.json()
                if "generated_text" in result:
                    print("✅ Inference successful")
                    print(f"Generated text preview: {result['generated_text'][:200]}...")
                    print(f"Input length: {result.get('input_length', 'N/A')}")
                    print(f"Output length: {result.get('output_length', 'N/A')}")
                else:
                    print(f"❌ Unexpected response format: {result}")
            else:
                print(f"❌ Inference failed: {response.text}")
                
        except requests.exceptions.Timeout:
            print("❌ Request timed out")
        except Exception as e:
            print(f"❌ Inference error: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test DeepSeek inference server")
    parser.add_argument("--url", default="http://localhost:8080", 
                       help="Base URL of the inference server")
    
    args = parser.parse_args()
    test_inference_server(args.url)
