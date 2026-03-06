FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.7.1-gpu-py312-ec2

# Set environment variables
ENV DATA_PATH=/opt/ml/data
ENV MODEL_PATH=/opt/ml/model
ENV PYTHONPATH=/opt/ml/code:$PYTHONPATH
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

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
    wget \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    easyocr \
    pytesseract \
    pillow \
    opencv-python \
    transformers \
    datasets \
    accelerate \
    torch-audio \
    torchaudio \
    matplotlib \
    numpy \
    scipy

# Create directories for models and data
RUN mkdir -p /opt/ml/model /opt/ml/data /opt/ml/code

# Download TrOCR model for text recognition
RUN python -c "from transformers import TrOCRProcessor, VisionEncoderDecoderModel; processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed'); model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed'); processor.save_pretrained('/opt/ml/model/trocr-small'); model.save_pretrained('/opt/ml/model/trocr-small')"

# Create text recognition inference script
RUN cat > /opt/ml/code/text_recognition.py << 'EOF'
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2
import numpy as np
import json
import easyocr
import pytesseract
import argparse
import os

class TextRecognitionModel:
    def __init__(self, model_type='trocr'):
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_type == 'trocr':
            # Load TrOCR model
            self.processor = TrOCRProcessor.from_pretrained('/opt/ml/model/trocr-small')
            self.model = VisionEncoderDecoderModel.from_pretrained('/opt/ml/model/trocr-small')
            self.model.to(self.device)
        elif model_type == 'easyocr':
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
    
    def recognize_text_trocr(self, image_path):
        """Recognize text using TrOCR model"""
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
    
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
        if self.model_type == 'trocr':
            text = self.recognize_text_trocr(image_path)
            return {'text': text, 'model': 'TrOCR'}
        elif self.model_type == 'easyocr':
            results = self.recognize_text_easyocr(image_path)
            return {'results': results, 'model': 'EasyOCR'}
        elif self.model_type == 'tesseract':
            results = self.recognize_text_tesseract(image_path)
            return {'results': results, 'model': 'Tesseract'}

def main():
    parser = argparse.ArgumentParser(description='Text Recognition')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--model', choices=['trocr', 'easyocr', 'tesseract'], 
                       default='trocr', help='Model to use for text recognition')
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

# Create training script for custom text recognition
RUN cat > /opt/ml/code/train_text_recognition.py << 'EOF'
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AdamW
from PIL import Image
import json
import os
import argparse
from tqdm import tqdm

class TextRecognitionDataset(Dataset):
    def __init__(self, data_dir, processor, max_target_length=128):
        self.data_dir = data_dir
        self.processor = processor
        self.max_target_length = max_target_length
        
        # Load annotations
        with open(os.path.join(data_dir, 'annotations.json'), 'r') as f:
            self.annotations = json.load(f)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations[idx]
        image_path = os.path.join(self.data_dir, 'images', item['image'])
        text = item['text']
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # Process text
        labels = self.processor.tokenizer(text, 
                                        padding="max_length", 
                                        max_length=self.max_target_length, 
                                        truncation=True, 
                                        return_tensors="pt").input_ids
        
        return {
            'pixel_values': pixel_values.flatten(),
            'labels': labels.flatten()
        }

def train_model(data_dir, model_dir, epochs=5, batch_size=8, learning_rate=5e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load processor and model
    processor = TrOCRProcessor.from_pretrained('/opt/ml/model/trocr-small')
    model = VisionEncoderDecoderModel.from_pretrained('/opt/ml/model/trocr-small')
    model.to(device)
    
    # Create dataset and dataloader
    dataset = TextRecognitionDataset(data_dir, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch in progress_bar:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
    
    # Save fine-tuned model
    model.save_pretrained(os.path.join(model_dir, 'fine_tuned_trocr'))
    processor.save_pretrained(os.path.join(model_dir, 'fine_tuned_trocr'))
    print('Training completed and model saved!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/opt/ml/input/data')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    
    args = parser.parse_args()
    train_model(args.data_dir, args.model_dir, args.epochs, args.batch_size, args.learning_rate)
EOF

# Create a demo script
RUN cat > /opt/ml/code/demo_text_recognition.py << 'EOF'
import os
from text_recognition import TextRecognitionModel
import json

def demo():
    print("Text Recognition Demo")
    print("=" * 50)
    
    # Check if sample image exists
    sample_image = "/opt/ml/data/sample_text.jpg"
    if not os.path.exists(sample_image):
        print(f"Sample image not found at {sample_image}")
        print("Please provide an image file to test text recognition.")
        return
    
    models = ['trocr', 'easyocr', 'tesseract']
    
    for model_name in models:
        print(f"\nTesting {model_name.upper()}:")
        print("-" * 30)
        
        try:
            model = TextRecognitionModel(model_type=model_name)
            result = model.predict(sample_image)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error with {model_name}: {str(e)}")

if __name__ == '__main__':
    demo()
EOF

# Make scripts executable
RUN chmod +x /opt/ml/code/*.py

# Create a simple README
RUN cat > /opt/ml/README.md << 'EOF'
# Text Recognition DLC Image

This custom Deep Learning Container includes multiple text recognition models:

## Available Models:
1. **TrOCR** (microsoft/trocr-small-printed) - Transformer-based OCR
2. **EasyOCR** - Easy-to-use OCR library
3. **Tesseract** - Traditional OCR engine

## Usage:

### Text Recognition:
```bash
python /opt/ml/code/text_recognition.py --image /path/to/image.jpg --model trocr
python /opt/ml/code/text_recognition.py --image /path/to/image.jpg --model easyocr
python /opt/ml/code/text_recognition.py --image /path/to/image.jpg --model tesseract
```

### Training (Fine-tuning TrOCR):
```bash
python /opt/ml/code/train_text_recognition.py --data-dir /opt/ml/input/data --model-dir /opt/ml/model
```

### Demo:
```bash
python /opt/ml/code/demo_text_recognition.py
```

## Data Format for Training:
Your training data should be organized as:
```
data/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── annotations.json
```

Annotations format:
```json
[
    {"image": "image1.jpg", "text": "Hello World"},
    {"image": "image2.jpg", "text": "Sample Text"}
]
```
EOF

WORKDIR /opt/ml
