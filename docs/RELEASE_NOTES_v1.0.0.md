# Release Notes - Face Liveness ViT v1.0.0

**Release Date:** January 11, 2026  
**Tag:** `v1.0.0`  
**Repository:** [affineinsai/face-liveness-vit](https://github.com/affineinsai/face-liveness-vit)

---

## 🎉 Initial Release

We're excited to announce the first stable release of **Face Liveness ViT**, a lightweight Vision Transformer model designed for real-time face liveness detection and anti-spoofing applications.

## 📦 What's Included

This release contains:

- **Pre-trained Model Checkpoint** (`model.pt`) - 12.5 MB
  - Trained on CelebA-Spoof dataset (intra-test split)
  - Optimized threshold: 0.1199
  - Model weights, optimizer state, and configuration embedded
  
- **Inference Script** (`inference.py`)
  - Complete model architecture implementation
  - Easy-to-use API for predictions
  - Command-line interface support
  - Automatic configuration loading from checkpoint
  
- **Training Configuration** (`config.yaml`)
  - All hyperparameters and data augmentation settings
  - Reproducible training setup
  
- **Documentation** (`README.md`)
  - Comprehensive usage examples
  - Installation instructions
  - Technical specifications

## ✨ Key Features

### Model Architecture
- **Lightweight Design**: Only 2 transformer layers with 4 attention heads
- **Efficient Processing**: 128-dimensional embeddings for fast inference
- **Vision Transformer**: Patch-based image encoding (16×16 patches)
- **Binary Classification**: Live face vs. Spoof attack detection

### Performance Characteristics
- **Input Size**: 224×224 RGB images
- **Optimal Threshold**: 0.1199 (balanced for false positives/negatives)
- **Training Dataset**: CelebA-Spoof with 625,537 images
- **Class Balancing**: Weighted loss [1.5, 1.0] for live/spoof classes

### Training Features
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) enabled
- **Advanced Augmentation**: 
  - Horizontal flipping
  - Color jittering
  - Random resized cropping
- **Optimization**: AdamW optimizer with weight decay
- **Drop Path Regularization**: 0.1 drop path rate for robustness

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/affineinsai/face-liveness-vit.git
cd face-liveness-vit

# Install dependencies
pip install torch torchvision pillow pyyaml
```

### Basic Usage

```bash
# Run inference on an image
python inference.py path/to/face_image.jpg
```

### Python API

```python
from inference import load_model, predict

# Load model
model, checkpoint = load_model('model.pt', device='cpu')

# Predict
result = predict(model, 'face.jpg', threshold=0.1199)
print(f"Result: {result['prediction']}")  # LIVE or SPOOF
```

## 📊 Technical Specifications

| Parameter | Value |
|-----------|-------|
| **Architecture** | Vision Transformer (ViT) |
| **Image Size** | 224×224 |
| **Patch Size** | 16×16 |
| **Embedding Dim** | 128 |
| **Attention Heads** | 4 |
| **Transformer Layers** | 2 |
| **Parameters** | ~1.2M |
| **Model Size** | 12.5 MB |

## 🔧 Training Configuration

```yaml
Model:
  - d_model: 128
  - nhead: 4
  - num_layers: 2
  - drop_path_rate: 0.1
  - patch_size: 16

Training:
  - batch_size: 64
  - learning_rate: 0.0003
  - weight_decay: 0.05
  - optimizer: AdamW
  - betas: (0.9, 0.999)
  - epochs: 30
  - amp: true

Data:
  - dataset: CelebA-Spoof
  - split: intra-test
  - num_workers: 8
  - class_weights: [1.5, 1.0]
```

## 🎯 Use Cases

This model is suitable for:

- **Mobile Authentication**: Lightweight enough for mobile deployment
- **Access Control Systems**: Real-time liveness verification
- **Identity Verification**: Anti-spoofing for KYC processes
- **Security Applications**: Preventing photo/video replay attacks
- **Research**: Baseline for liveness detection experiments

## ⚠️ Known Limitations

- Trained specifically on CelebA-Spoof dataset distribution
- Performance may vary across different:
  - Demographics and ethnicities
  - Lighting conditions
  - Camera qualities and angles
  - Spoofing attack types not in training data
- Should be part of a multi-factor authentication system
- Not recommended as sole authentication mechanism

## 🔄 Compatibility

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **CUDA**: Optional (CPU inference supported)
- **OS**: Linux, macOS, Windows

## 📝 Usage Examples

### Command Line Interface

```bash
# Basic inference
python inference.py face.jpg

# Custom checkpoint and threshold
python inference.py face.jpg model.pt 0.12

# With GPU
CUDA_VISIBLE_DEVICES=0 python inference.py face.jpg
```

### Batch Processing

```python
import torch
from inference import load_model, predict
from pathlib import Path

model, checkpoint = load_model('model.pt', device='cuda')
threshold = checkpoint.get('threshold', 0.1199)

for image_path in Path('faces').glob('*.jpg'):
    result = predict(model, str(image_path), threshold=threshold, device='cuda')
    print(f"{image_path.name}: {result['prediction']} (prob: {result['spoof_probability']:.4f})")
```

## 📚 Additional Resources

- **Repository**: https://github.com/affineinsai/face-liveness-vit
- **Documentation**: See README.md
- **Issues**: Report bugs on GitHub Issues
- **License**: Apache 2.0

## 🙏 Acknowledgments

- **Dataset**: CelebA-Spoof dataset creators
- **Framework**: PyTorch team
- **Architecture**: Inspired by Vision Transformer (ViT) paper

## 📄 Citation

If you use this model in your research, please cite:

```bibtex
@misc{face-liveness-vit-v1,
  author = {Affine Institute AI},
  title = {Face Liveness Detection with Vision Transformer v1.0.0},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/affineinsai/face-liveness-vit}},
  version = {1.0.0}
}
```

## 🔐 Security Notice

This model is provided for research and development purposes. For production security applications:

1. Conduct thorough testing on your specific use case
2. Implement multi-factor authentication
3. Monitor for adversarial attacks
4. Keep the model updated as new attack vectors emerge
5. Follow security best practices for deployment

## 📧 Contact

For questions, issues, or collaboration:
- **GitHub Issues**: https://github.com/affineinsai/face-liveness-vit/issues
- **Organization**: Affine Institute AI

---

**Full Changelog**: Initial Release v1.0.0

**Download Assets**:
- [model.pt](https://github.com/affineinsai/face-liveness-vit/releases/download/v1.0.0/model.pt)
- [Source code (zip)](https://github.com/affineinsai/face-liveness-vit/archive/refs/tags/v1.0.0.zip)
- [Source code (tar.gz)](https://github.com/affineinsai/face-liveness-vit/archive/refs/tags/v1.0.0.tar.gz)
