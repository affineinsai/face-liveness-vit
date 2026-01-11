---
tags:
- face-liveness-detection
- anti-spoofing
- computer-vision
- pytorch
- vision-transformer
license: apache-2.0
---

# Face Liveness Detection Model

[![Version](https://img.shields.io/badge/version-v1.0.0-blue.svg)](https://github.com/affineinsai/face-liveness-vit/releases/tag/v1.0.0)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

This model performs **face liveness detection** to distinguish between real faces and spoofing attempts (e.g., printed photos, video replay attacks, masks).

## Release History

### v1.0.0 (January 2026)
- Initial release of Face Liveness ViT model
- Trained on CelebA-Spoof dataset
- Achieves optimal threshold of 0.1199 for balanced accuracy
- Lightweight architecture (128 embedding dim, 2 layers, 4 heads)
- Includes pre-trained model checkpoint with configuration
- Ready-to-use inference script provided

## Files Included

- `model.pt` - Pre-trained model checkpoint with weights and configuration
- `inference.py` - Ready-to-use inference script with model architecture
- `config.yaml` - Training configuration and hyperparameters
- `README.md` - This documentation file

## Installation

```bash
pip install torch torchvision pillow pyyaml
```

For GPU support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Model Description

- **Architecture**: Vision Transformer (ViT) based encoder
- **Image Size**: 224x224
- **Patch Size**: 16x16
- **Embedding Dimension**: 128
- **Attention Heads**: 4
- **Transformer Layers**: 2
- **Classes**: Live (0) vs Spoof (1)
- **Optimal Threshold**: 0.1199

## Training Details

- **Dataset**: CelebA-Spoof (intra-test split)
- **Batch Size**: 64 (training), 128 (evaluation)
- **Learning Rate**: 0.0003
- **Betas**: (0.9, 0.999)
- **Weight Decay**: 0.05
- **Class Weights**: [1.5, 1.0] (live, spoof)
- **Epochs**: 30
- **Optimizer**: AdamW
- **Mixed Precision**: True (AMP)
- **Drop Path Rate**: 0.1

### Data Augmentation

- Horizontal flip (p=0.5)
- Color jitter (brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
- Random resized crop (scale 0.8-1.0)
- Resize to 256 then center crop to 224

## Usage

### Quick Start with Inference Script

The easiest way to use the model is via the provided inference script:

```bash
python inference.py path/to/face_image.jpg
```

With custom threshold:
```bash
python inference.py path/to/face_image.jpg model.pt 0.1199
```

### Python API Usage

```python
import torch
from inference import load_model, predict

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, checkpoint = load_model('model.pt', device)
model = model.to(device)

# Get optimal threshold from checkpoint
threshold = checkpoint.get('threshold', 0.1199)

# Predict on image
result = predict(model, 'face.jpg', threshold=threshold, device=device)

print(f"Result: {result['prediction']}")
print(f"Live probability: {result['live_probability']:.4f}")
print(f"Spoof probability: {result['spoof_probability']:.4f}")
```

### Manual Usage

```python
import torch
from inference import LivenessViT
from PIL import Image
import torchvision.transforms as transforms

# Load model
checkpoint = torch.load('model.pt', map_location='cpu', weights_only=False)
cfg = checkpoint.get('cfg', {}).get('model', {})

model = LivenessViT(
    img_size=cfg.get('image_size', 224),
    patch_size=cfg.get('patch_size', 16),
    d_model=cfg.get('d_model', 128),
    nhead=cfg.get('nhead', 4),
    num_layers=cfg.get('num_layers', 2),
    num_classes=2,
    dropout=cfg.get('drop_rate', 0.0),
)
model.load_state_dict(checkpoint['model'])
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open("face.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Inference
threshold = checkpoint.get('threshold', 0.1199)
with torch.no_grad():
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1)
    spoof_prob = probs[0, 1].item()
    
    is_live = spoof_prob < threshold
    print(f"Result: {'LIVE' if is_live else 'SPOOF'}")
    print(f"Spoof probability: {spoof_prob:.4f}")
```

## Model Performance

The model was trained on CelebA-Spoof dataset with class balancing and data augmentation.
For detailed evaluation metrics, please refer to the training logs.

## Limitations

- Trained specifically on CelebA-Spoof dataset
- Performance may vary on different demographics or imaging conditions
- Should be used as part of a comprehensive security system, not as sole authentication

## Technical Notes

- The model uses a lightweight Vision Transformer architecture optimized for liveness detection
- Images are normalized using ImageNet statistics
- The optimal threshold (0.1199) was determined during validation to balance false positives and false negatives
- The checkpoint includes model configuration, allowing automatic parameter loading

## Citation

If you use this model, please cite:

```bibtex
@misc{face-liveness-vit,
  author = {Affine Institute AI},
  title = {Face Liveness Detection with Vision Transformer},
  year = {2026},
  publisher = {Github},
  howpublished = {\url{https://github.com/affineinsai/face-liveness-vit}}
}
```

## License

Apache 2.0
