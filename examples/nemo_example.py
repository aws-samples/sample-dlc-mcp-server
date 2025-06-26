#!/usr/bin/env python3
"""
Simple NeMo toolkit example demonstrating core functionality
This script shows basic NeMo usage without GPU dependencies
"""

import nemo
import nemo.core as nemo_core
from nemo.core.classes import Dataset, IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType, LabelsType
import torch
import pytorch_lightning as pl

print("🚀 NeMo Toolkit Example")
print("=" * 50)

# Display versions
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch Lightning version: {pl.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"NeMo toolkit imported successfully")

print("\n📊 Neural Types Examples:")
print("-" * 30)

# Create some neural type examples
audio_signal = NeuralType(("B", "T"), AudioSignal())
lengths = NeuralType(tuple("B"), LengthsType())
labels = NeuralType(("B", "T"), LabelsType())

print(f"Audio Signal Type: {audio_signal}")
print(f"Lengths Type: {lengths}")
print(f"Labels Type: {labels}")

print("\n🔧 Core Classes:")
print("-" * 20)


# Example of a simple dataset class
class SimpleDataset(Dataset):
    def __init__(self, data_size=100):
        super().__init__()
        self.data_size = data_size

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        # Return dummy data for demonstration
        return {
            "audio": torch.randn(16000),  # 1 second of audio at 16kHz
            "text": f"Sample text {idx}",
            "length": 16000,
        }


# Create dataset instance
dataset = SimpleDataset(data_size=10)
print(f"✅ Created dataset with {len(dataset)} samples")
print(f"Sample data keys: {list(dataset[0].keys())}")

print("\n🎯 NeMo Configuration System:")
print("-" * 35)

# Demonstrate NeMo's configuration system
from omegaconf import OmegaConf

# Create a simple config
config = OmegaConf.create(
    {
        "model": {"name": "example_model", "sample_rate": 16000, "n_mels": 80, "n_fft": 1024},
        "training": {"batch_size": 32, "learning_rate": 0.001, "max_epochs": 100},
    }
)

print("Configuration:")
print(OmegaConf.to_yaml(config))

print("\n✨ Available NeMo Collections:")
print("-" * 35)

collections_info = {
    "ASR": "Automatic Speech Recognition - Speech to Text",
    "NLP": "Natural Language Processing - Text Understanding",
    "TTS": "Text-to-Speech - Text to Audio Generation",
    "Vision": "Computer Vision - Image Processing",
    "Multimodal": "Multi-modal AI - Cross-modal Understanding",
}

for collection, description in collections_info.items():
    print(f"📚 {collection}: {description}")

print("\n🎉 NeMo Toolkit Setup Complete!")
print("=" * 50)
print("Your NeMo environment is ready for:")
print("• Building conversational AI models")
print("• Speech recognition and synthesis")
print("• Natural language processing")
print("• Multi-modal AI applications")
print("• Large language model training")

print("\n💡 Next Steps:")
print("• Explore NeMo tutorials and examples")
print("• Download pre-trained models from NGC")
print("• Start building your AI applications!")
