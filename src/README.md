# Cityscapes UNet–ResNet50 Segmentation

## Overview

This package trains a UNet with a ResNet-50 encoder on Cityscapes using 3-fold cross-validation. It produces per-fold mIoU metrics, training curves, and example visualizations.

## Contents

├── k_cross_validation.py # main cross-validation training script
├── models/
│ └── unet_resnet.py # UNet–ResNet50 model definition
├── run_crossval.sh # helper to launch training
└── requirements.txt # (generated—see below)


## Setup

1. **Unzip** the archive and `cd` into it.
2. Create a Python virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
    ```
3. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```
4. Download the Cityscapes Dataset (gtFine_trainvaltest.zip and leftImg8bit_trainvaltest.zip) and unzip it.


## How to run? 
All of the training parameters live in the `__main__` guard at the bottom of **k_cross_validation.py**.  Edit them there to suit your needs:

```python
if __name__ == "__main__":
    cross_validate(
        train_image_dir="cityscapes_data/leftImg8bit/train",
        train_mask_dir="cityscapes_data/gtFine/train",
        val_image_dir="cityscapes_data/leftImg8bit/val",
        val_mask_dir="cityscapes_data/gtFine/val",
        k_folds=3,          # number of CV folds
        epochs=65,          # epochs per fold
        batch_size=2,
        lr=1e-4,
        output_dir="cv_results"
    )
```

## Runing the script after adjusting the data.
Here make sure your virtual environment is active. Also make sure you are excetuing this from directory where the k_cross_validation.py file is.

```bash
python install -r requirements.txt
```




## Model

- **Encoder**: ResNet-50 pretrained on ImageNet; we extract features after `conv1`, `layer1`, `layer2`, `layer3`, `layer4`.  
- **Bottleneck**: Two consecutive 3×3 Conv–BN–ReLU blocks with Dropout2d.  
- **Decoder**: Four upsampling steps (ConvTranspose2d), each followed by skip-connection concatenation and a ConvBlock (Conv–BN–ReLU ×2).  
- **Output head**: 1×1 convolution to 19 classes, then bilinear upsample to match original input size.  

All convolutional blocks use batch-normalization and ReLU activation.

---

## Data & Augmentations

- **Source**: Cityscapes “leftImg8bit” images and “gtFine” semantic PNG masks.  
- **Label mapping**: Raw Cityscapes IDs → train IDs [0..18], with 255 as ignore index.  
- **Train transforms** (apply to each 1024×2048 image):
  - Random scale (0.8–1.2), pad to ≥1024×2048 if needed  
  - Resize to 512×1024  
  - Horizontal flip (p=0.5), color jitter  
  - Normalize (ImageNet mean/std), convert to tensor  
- **Val transforms**:
  - Resize to 512×1024, normalize, to tensor  

Visualization utilities decode model outputs to RGB using the Cityscapes palette.

---

## k-Fold Cross-Validation

1. **Load** the full Cityscapes train split with low-res transforms.  
2. **Shuffle & split** indices into k equally-sized folds.  
3. For each fold _i_:
   - Use folds ≠ _i_ as training set, fold _i_ as validation.  
   - Train for 65 epochs (batch size 2, Adam lr=1e-4, ReduceLROnPlateau).  
   - Every 5 epochs, compute low-res mIoU on held-out fold.  
   - Every 10 epochs (and at last epoch), compute full-res mIoU on official val split: 
     - Resize input to 512×1024, forward, then upsample logits to 1024×2048 and compare.  
   - Save per-fold training curves (`training_history.png`), a sample prediction (`final_prediction.png`), and model weights (`unet_resnet50_fold{i}.pth`).  
4. **Aggregate**: write `summary_metrics.csv` with each fold’s best low-res & final full-res mIoU, plus mean/std across folds.
