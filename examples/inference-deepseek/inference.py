#!/usr/bin/env python3
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

"""
DeepSeek Model Inference Script for PyTorch DLC
"""

import json
import logging
import os
import sys
from typing import Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, jsonify
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepSeekInferenceHandler:
    """
    Handler for DeepSeek model inference
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cpu"  # Using CPU for this DLC
        self.model_name = os.environ.get("MODEL_NAME", "deepseek-ai/deepseek-coder-6.7b-instruct")
        self.max_length = 2048
        self.temperature = 0.7
        self.top_p = 0.9
        
    def initialize(self):
        """Initialize the model and tokenizer"""
        try:
            logger.info(f"Loading DeepSeek model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=os.environ.get("TRANSFORMERS_CACHE", "/opt/ml/model")
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map="cpu",
                trust_remote_code=True,
                cache_dir=os.environ.get("TRANSFORMERS_CACHE", "/opt/ml/model")
            )
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            return False
    
    def preprocess(self, input_data: Dict[str, Any]) -> str:
        """Preprocess input data"""
        if isinstance(input_data, dict):
            prompt = input_data.get("inputs", input_data.get("prompt", ""))
        else:
            prompt = str(input_data)
        
        # Format prompt for DeepSeek
        if not prompt.startswith("### Instruction:"):
            formatted_prompt = f"### Instruction:\\n{prompt}\\n\\n### Response:\\n"
        else:
            formatted_prompt = prompt
            
        return formatted_prompt
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prediction"""
        try:
            # Preprocess input
            prompt = self.preprocess(input_data)
            
            # Get generation parameters
            max_length = input_data.get("max_length", self.max_length)
            temperature = input_data.get("temperature", self.temperature)
            top_p = input_data.get("top_p", self.top_p)
            do_sample = input_data.get("do_sample", True)
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract generated text (remove input prompt)
            generated_text = response[len(prompt):].strip()
            
            return {
                "generated_text": generated_text,
                "full_response": response,
                "input_length": len(inputs[0]),
                "output_length": len(outputs[0])
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {"error": str(e)}

# Global handler instance
handler = DeepSeekInferenceHandler()

# Flask app for serving
app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

@app.route("/invocations", methods=["POST"])
def invocations():
    """Main inference endpoint"""
    try:
        # Get input data
        input_data = request.get_json()
        
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Generate prediction
        result = handler.predict(input_data)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Invocation failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def root():
    """Root endpoint"""
    return jsonify({
        "message": "DeepSeek Model Inference Server",
        "model": handler.model_name,
        "status": "ready" if handler.model is not None else "initializing"
    })

def initialize_model():
    """Initialize model in background"""
    logger.info("Starting model initialization...")
    success = handler.initialize()
    if success:
        logger.info("Model initialization completed successfully")
    else:
        logger.error("Model initialization failed")
        sys.exit(1)

if __name__ == "__main__":
    # Initialize model in background thread
    init_thread = threading.Thread(target=initialize_model)
    init_thread.start()
    
    # Start Flask server
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
