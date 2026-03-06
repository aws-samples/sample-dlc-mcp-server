FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.7.1-gpu-py312-ec2

# Set environment variables
ENV HF_HOME=/opt/ml/models/huggingface
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
ENV TRANSFORMERS_CACHE=/opt/ml/models/transformers

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages (excluding problematic ones for now)
RUN pip install --no-cache-dir \
    easyocr \
    pytesseract \
    pillow \
    opencv-python \
    transformers \
    datasets \
    accelerate \
    matplotlib \
    numpy \
    scipy \
    tqdm \
    sentencepiece \
    protobuf \
    bitsandbytes

# Try to install flash-attn and vllm separately (they might fail on some systems)
RUN pip install flash-attn --no-build-isolation || echo "Flash attention installation failed, continuing without it"
RUN pip install vllm || echo "VLLM installation failed, continuing without it"

# Create directories for models and data
RUN mkdir -p /opt/ml/models/huggingface /opt/ml/models/transformers /opt/ml/data /opt/ml/code

# Download DeepSeek-V3 tokenizer (model weights downloaded on first use)
RUN python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-V3', trust_remote_code=True); tokenizer.save_pretrained('/opt/ml/models/deepseek-v3'); print('DeepSeek-V3 tokenizer downloaded')"

# Create DeepSeek inference script
COPY <<EOF /opt/ml/code/deepseek_inference.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import argparse
import os
from typing import Optional, List, Dict, Any

class DeepSeekModel:
    def __init__(self, model_path: str = "deepseek-ai/DeepSeek-V3", device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the DeepSeek model and tokenizer"""
        print(f"Loading DeepSeek model from {self.model_path}...")
        
        # Load tokenizer
        local_tokenizer_path = "/opt/ml/models/deepseek-v3"
        if os.path.exists(local_tokenizer_path):
            self.tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
        # Load model with appropriate settings
        try:
            if self.device == "auto":
                device_map = "auto"
            else:
                device_map = None
            
            # Try with flash attention first
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map=device_map,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2"
                )
            except:
                # Fallback without flash attention
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map=device_map,
                    trust_remote_code=True
                )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to CPU or reduced precision...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
    
    def generate_response(
        self, 
        prompt: str, 
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """Generate response using DeepSeek model"""
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available() and hasattr(self.model, 'device') and self.model.device.type == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        max_length: int = 2048,
        temperature: float = 0.7
    ) -> str:
        """Chat interface for DeepSeek model"""
        
        # Format messages for DeepSeek
        formatted_prompt = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                formatted_prompt += f"System: {content}\\n"
            elif role == "user":
                formatted_prompt += f"User: {content}\\n"
            elif role == "assistant":
                formatted_prompt += f"Assistant: {content}\\n"
        
        formatted_prompt += "Assistant: "
        
        return self.generate_response(
            formatted_prompt, 
            max_length=max_length, 
            temperature=temperature
        )

def main():
    parser = argparse.ArgumentParser(description="DeepSeek Model Inference")
    parser.add_argument("--prompt", type=str, help="Input prompt for generation")
    parser.add_argument("--chat", action="store_true", help="Enable chat mode")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--model-path", type=str, default="deepseek-ai/DeepSeek-V3", help="Model path")
    
    args = parser.parse_args()
    
    # Initialize model
    model = DeepSeekModel(model_path=args.model_path)
    
    if args.chat:
        print("DeepSeek Chat Mode - Type 'quit' to exit")
        messages = []
        
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            
            messages.append({"role": "user", "content": user_input})
            
            try:
                response = model.chat(messages, max_length=args.max_length, temperature=args.temperature)
                print(f"DeepSeek: {response}")
                messages.append({"role": "assistant", "content": response})
            except Exception as e:
                print(f"Error: {e}")
    
    elif args.prompt:
        try:
            response = model.generate_response(
                args.prompt, 
                max_length=args.max_length, 
                temperature=args.temperature
            )
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        print("Please provide --prompt or use --chat mode")

if __name__ == "__main__":
    main()
EOF

# Create text recognition script
COPY <<EOF /opt/ml/code/text_recognition.py
import torch
import easyocr
import pytesseract
from PIL import Image
import cv2
import numpy as np
import json
import argparse
import os

class TextRecognitionModel:
    def __init__(self, model_type='easyocr'):
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_type == 'easyocr':
            # Initialize EasyOCR
            self.reader = easyocr.Reader(['en'])
        elif model_type == 'tesseract':
            # Tesseract is ready to use
            pass
    
    def preprocess_image(self, image_path):
        """Preprocess image for better OCR results"""
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply thresholding
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def recognize_text_easyocr(self, image_path):
        """Recognize text using EasyOCR"""
        results = self.reader.readtext(image_path)
        text_results = []
        
        for (bbox, text, confidence) in results:
            text_results.append({
                'text': text,
                'confidence': float(confidence),
                'bbox': bbox
            })
        
        return text_results
    
    def recognize_text_tesseract(self, image_path):
        """Recognize text using Tesseract OCR"""
        # Preprocess image
        processed_image = self.preprocess_image(image_path)
        
        # Extract text
        text = pytesseract.image_to_string(processed_image)
        
        # Get detailed data
        data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)
        
        text_results = []
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 30:  # Filter low confidence
                text_results.append({
                    'text': data['text'][i],
                    'confidence': int(data['conf'][i]) / 100.0,
                    'bbox': [
                        data['left'][i],
                        data['top'][i],
                        data['left'][i] + data['width'][i],
                        data['top'][i] + data['height'][i]
                    ]
                })
        
        return {'full_text': text.strip(), 'detailed': text_results}
    
    def predict(self, image_path):
        """Main prediction method"""
        if self.model_type == 'easyocr':
            results = self.recognize_text_easyocr(image_path)
            return {'results': results, 'model': 'EasyOCR'}
        elif self.model_type == 'tesseract':
            results = self.recognize_text_tesseract(image_path)
            return {'results': results, 'model': 'Tesseract'}

def main():
    parser = argparse.ArgumentParser(description='Text Recognition')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--model', choices=['easyocr', 'tesseract'], 
                       default='easyocr', help='Model to use for text recognition')
    parser.add_argument('--output', help='Output file path (optional)')
    
    args = parser.parse_args()
    
    # Initialize model
    model = TextRecognitionModel(model_type=args.model)
    
    # Perform text recognition
    result = model.predict(args.image)
    
    # Output results
    output_json = json.dumps(result, indent=2)
    print(output_json)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_json)

if __name__ == '__main__':
    main()
EOF

# Create multimodal inference script
COPY <<EOF /opt/ml/code/multimodal_inference.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import easyocr
import pytesseract
from PIL import Image
import cv2
import numpy as np
import json
import argparse
import os
from typing import Optional, List, Dict, Any

class MultimodalInference:
    def __init__(self):
        self.deepseek_model = None
        self.deepseek_tokenizer = None
        self.ocr_reader = None
        self.load_models()
    
    def load_models(self):
        """Load all models"""
        print("Loading models...")
        
        # Load DeepSeek
        try:
            local_tokenizer_path = "/opt/ml/models/deepseek-v3"
            if os.path.exists(local_tokenizer_path):
                self.deepseek_tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, trust_remote_code=True)
            else:
                self.deepseek_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3", trust_remote_code=True)
            
            self.deepseek_model = AutoModelForCausalLM.from_pretrained(
                "deepseek-ai/DeepSeek-V3",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            print("DeepSeek model loaded!")
        except Exception as e:
            print(f"Warning: Could not load DeepSeek model: {e}")
        
        # Load OCR
        try:
            self.ocr_reader = easyocr.Reader(["en"])
            print("OCR models loaded!")
        except Exception as e:
            print(f"Warning: Could not load OCR: {e}")
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        if not self.ocr_reader:
            return "OCR not available"
        
        try:
            results = self.ocr_reader.readtext(image_path)
            extracted_text = " ".join([result[1] for result in results])
            return extracted_text
        except Exception as e:
            return f"Error extracting text: {e}"
    
    def analyze_image_with_ai(self, image_path: str, question: str = "What do you see in this image?") -> str:
        """Analyze image by extracting text and using DeepSeek for analysis"""
        
        # Extract text from image
        extracted_text = self.extract_text_from_image(image_path)
        
        if not self.deepseek_model:
            return f"Extracted text: {extracted_text}"
        
        # Create prompt for DeepSeek
        prompt = f"""I have extracted the following text from an image:

Extracted Text: {extracted_text}

Question: {question}

Please analyze this text and provide insights:"""
        
        try:
            inputs = self.deepseek_tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.deepseek_model.generate(
                    **inputs,
                    max_length=2048,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.deepseek_tokenizer.eos_token_id
                )
            
            response = self.deepseek_tokenizer.decode(outputs[0], skip_special_tokens=True)
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
        
        except Exception as e:
            return f"Error in AI analysis: {e}. Extracted text: {extracted_text}"

def main():
    parser = argparse.ArgumentParser(description="Multimodal Inference with DeepSeek and OCR")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--question", type=str, default="What do you see in this image?", help="Question about the image")
    parser.add_argument("--text-only", action="store_true", help="Only extract text, no AI analysis")
    
    args = parser.parse_args()
    
    if not args.image:
        print("Please provide an image path with --image")
        return
    
    # Initialize multimodal inference
    inference = MultimodalInference()
    
    if args.text_only:
        # Only extract text
        text = inference.extract_text_from_image(args.image)
        print(f"Extracted Text: {text}")
    else:
        # Full analysis
        analysis = inference.analyze_image_with_ai(args.image, args.question)
        print(f"Analysis: {analysis}")

if __name__ == "__main__":
    main()
EOF

# Create model download script
COPY <<EOF /opt/ml/code/download_deepseek.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

def download_deepseek_model():
    """Download DeepSeek-V3 model to local storage"""
    model_name = "deepseek-ai/DeepSeek-V3"
    local_path = "/opt/ml/models/deepseek-v3-full"
    
    print(f"Downloading {model_name} to {local_path}...")
    
    try:
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.save_pretrained(local_path)
        print("Tokenizer downloaded successfully!")
        
        # Download model (this will be large)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        model.save_pretrained(local_path)
        print("Model downloaded successfully!")
        
        print(f"DeepSeek-V3 model saved to: {local_path}")
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("You can download it manually when needed.")

if __name__ == "__main__":
    download_deepseek_model()
EOF

# Make scripts executable
RUN chmod +x /opt/ml/code/*.py

# Create README with usage instructions
COPY <<EOF /opt/ml/README.md
# Enhanced DLC with DeepSeek-V3 and Text Recognition

This container includes:
- **DeepSeek-V3**: Latest large language model
- **Text Recognition**: EasyOCR, Tesseract support
- **Multimodal**: Combined image analysis with AI

## Usage:

### DeepSeek-V3 Inference:
```bash
# Simple text generation
python /opt/ml/code/deepseek_inference.py --prompt "Explain quantum computing"

# Interactive chat
python /opt/ml/code/deepseek_inference.py --chat
```

### Text Recognition:
```bash
# Extract text from image
python /opt/ml/code/text_recognition.py --image /path/to/image.jpg --model easyocr
```

### Multimodal Analysis:
```bash
# Analyze image with AI
python /opt/ml/code/multimodal_inference.py --image /path/to/image.jpg --question "What is this document about?"

# Extract text only
python /opt/ml/code/multimodal_inference.py --image /path/to/image.jpg --text-only
```

### Download Full Model (Optional):
```bash
# Download complete DeepSeek model for offline use
python /opt/ml/code/download_deepseek.py
```

## Model Information:
- **DeepSeek-V3**: Latest version with improved capabilities
- **Memory Requirements**: Recommend 24GB+ GPU memory for full model
- **Fallback**: Automatic CPU fallback if GPU memory insufficient

## Environment Variables:
- `HF_HOME`: Hugging Face cache directory
- `TRANSFORMERS_CACHE`: Transformers cache directory
- `TESSDATA_PREFIX`: Tesseract data directory
EOF

WORKDIR /opt/ml
