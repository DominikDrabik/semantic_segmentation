"""
Performs k-fold cross-validation training of a U-Net with ResNet-50 backbone on the
Cityscapes dataset. At the end of each fold it saves training curves, a sample visualization,
and model weights, and finally writes summary_metrics.csv with mean/std mIoU.

Inspired by: “Cityscapes Tutorial with torchvision.datasets.Cityscapes” 
(https://www.youtube.com/watch?v=hWN1Xqu8aRE) for dataset loading.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex
from typing import List
from models.unet_resnet import UNetResNet
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets import Cityscapes
import csv

# Labeling for matching
ignore_index = 255
void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [
    ignore_index,  7,  8, 11, 12, 13, 17, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33
]
n_classes = len(valid_classes)  # = 20

class_map = dict(zip(valid_classes, range(n_classes)))

def encode_segmap(mask_np: np.ndarray) -> np.ndarray:
    mask_copy = mask_np.copy()
    for voidc in void_classes:
        mask_copy[mask_copy == voidc] = ignore_index
    for validc in valid_classes:
        mask_copy[mask_copy == validc] = class_map[validc]
    mask_copy = mask_copy.astype(np.uint8)

    out = np.ones_like(mask_copy, dtype=np.uint8) * ignore_index
    for mapped_id in range(1, n_classes):
        out[mask_copy == mapped_id] = mapped_id - 1
    return out

raw_to_color = {
    7:  [128,  64, 128],
    8:  [244,  35, 232],
    11: [ 70,  70,  70],
    12: [102, 102, 156],
    13: [190, 153, 153],
    17: [153, 153, 153],
    19: [250, 170,  30],
    20: [220, 220,   0],
    21: [107, 142,  35],
    22: [152, 251, 152],
    23: [  0, 130, 180],
    24: [220,  20,  60],
    25: [255,   0,   0],
    26: [  0,   0, 142],   
    27: [  0,   0,  70],   
    28: [  0,  60, 100],   
    31: [  0,  80, 100],   
    32: [  0,   0, 230],   
    33: [119,  11,  32],  
}

train_colors = [None] * 19
for i in range(1, n_classes):
    raw_id = valid_classes[i]
    train_id = i - 1
    train_colors[train_id] = raw_to_color[raw_id]

label_colours = {idx: tuple(train_colors[idx]) for idx in range(19)}

def decode_segmap(mask_tensor: torch.Tensor) -> np.ndarray:
    if mask_tensor.device != torch.device("cpu"):
        mask_tensor = mask_tensor.cpu()
    temp = mask_tensor.numpy().astype(np.uint8)
    h, w = temp.shape

    r = np.zeros((h, w), dtype=np.uint8)
    g = np.zeros((h, w), dtype=np.uint8)
    b = np.zeros((h, w), dtype=np.uint8)

    for train_id in range(19):
        color = label_colours[train_id]
        mask_l = (temp == train_id)
        if mask_l.any():
            r[mask_l] = color[0]
            g[mask_l] = color[1]
            b[mask_l] = color[2]

    # ignore pixel remains black 
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = r / 255.0
    rgb[..., 1] = g / 255.0
    rgb[..., 2] = b / 255.0
    return rgb

def visualize(img_tensor, mask_tensor, pred_tensor, save_path):
    """
    Given:
      img_tensor:  (3, H, W), normalized
      mask_tensor: (H, W) in {0..18} or 255
      pred_tensor: (H, W) in {0..18}
    Save a side-by-side comparison: [input | gt_color | pred_color]
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * std[None, None, :]) + mean[None, None, :]
    img_np = np.clip(img_np, 0.0, 1.0)

    gt_color = decode_segmap(mask_tensor)

    pred_color = decode_segmap(pred_tensor)

    H, W, _ = img_np.shape
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 6))
    ax0.imshow(img_np)
    ax0.set_title("Input Image")
    ax0.axis("off")

    ax1.imshow(gt_color)
    ax1.set_title("Ground Truth")
    ax1.axis("off")

    ax2.imshow(pred_color)
    ax2.set_title("Prediction")
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

class CityscapesAlb(Cityscapes):
    def __init__(self, root, split, mode, target_type, transforms):
        super().__init__(
            root=root,
            split=split,
            mode=mode,
            target_type=target_type,
            transforms=transforms
        )
        filtered_imgs = []
        filtered_tgts = []
        for img_path, tgt_list in zip(self.images, self.targets):
            if img_path.lower().endswith(".png") \
               and all(p.lower().endswith(".png") for p in tgt_list):
                filtered_imgs.append(img_path)
                filtered_tgts.append(tgt_list)
        self.images = filtered_imgs
        self.targets = filtered_tgts

    def __getitem__(self, index: int):
        image = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.targets[index][0]) 

        img_np  = np.array(image)       
        mask_np = np.array(mask)       
        augmented = self.transforms(image=img_np, mask=mask_np)
        img_tensor  = augmented["image"]            
        mask_tensor = augmented["mask"].numpy()      

        encoded_mask = encode_segmap(mask_tensor)    

        return img_tensor, torch.as_tensor(encoded_mask, dtype=torch.long)

# Albumentations 
train_transform = A.Compose([
    A.RandomScale(scale_limit=(0.8, 1.2), p=0.25),
    A.PadIfNeeded(min_height=1024, min_width=2048, border_mode=0),
    A.Resize(512, 1024),              
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(512, 1024),              
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def plot_training_history(train_losses: List[float], val_mious: List[float], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_mious, label="Val mIoU")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Loss and Validation mIoU")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(save_path)
    plt.close()

def validate_model(model, val_loader, device):
    """Run one pass of validation and compute mIoU (ignoring 255)."""
    model.eval()
    miou_metric = MulticlassJaccardIndex(num_classes=19).to(device)

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)
            logits = model(images)                  
            preds  = logits.argmax(dim=1)          
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)
            valid = (masks_flat != 255)
            if valid.sum() == 0:
                continue
            preds_flat = preds_flat[valid]
            masks_flat = masks_flat[valid]

            miou_metric.update(preds_flat, masks_flat)

    return miou_metric.compute().item()

class CityscapesFull(Cityscapes):
    def __init__(self, root, split='val', mode='fine', target_type='semantic'):
        super().__init__(root=root, split=split, mode=mode, target_type=target_type)

        filtered_imgs = []
        filtered_tgts = []
        for img_path, tgt_list in zip(self.images, self.targets):
            if img_path.lower().endswith(".png"):
                filtered_imgs.append(img_path)
                filtered_tgts.append(tgt_list)
        self.images  = filtered_imgs
        self.targets = filtered_tgts

    def __getitem__(self, idx):
        img_path, (mask_path,) = self.images[idx], self.targets[idx]
        orig_img  = Image.open(img_path).convert('RGB')   
        orig_mask = Image.open(mask_path)                 
        return orig_img, orig_mask

def evaluate_full_res(model, city_root, device):
    """Evaluate model on the full 1024×2048 Cityscapes val set and return mIoU."""
    inference_transform = A.Compose([
        A.Resize(512, 1024),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    dataset_full = CityscapesFull(
        root=city_root,
        split='val',
        mode='fine',
        target_type='semantic'
    )

    miou_metric_full = MulticlassJaccardIndex(num_classes=19).to(device)
    with torch.no_grad():
        for idx in range(len(dataset_full)):
            orig_img_pil, orig_mask_pil = dataset_full[idx]

            img_np = np.array(orig_img_pil) 
            aug = inference_transform(image=img_np)
            img_mid = aug['image'].unsqueeze(0).to(device) 

            logits_mid = model(img_mid) 
            logits_hi  = F.interpolate(
                logits_mid,
                size=(1024, 2048),
                mode='bilinear',
                align_corners=False
            )  

            pred_hi = logits_hi.argmax(dim=1)[0]               
            pred_flat = pred_hi.flatten()
            gt_raw    = np.array(orig_mask_pil).flatten()      
            gt_enc    = encode_segmap(gt_raw.reshape(1024, 2048)).flatten()

            valid = (gt_enc != 255)
            if valid.sum() == 0:
                continue

            pred_valid = pred_flat[valid].long().to(device)
            gt_valid   = torch.from_numpy(gt_enc[valid]).long().to(device)
            miou_metric_full.update(pred_valid, gt_valid)

    final_miou_full = miou_metric_full.compute().item()
    return final_miou_full

# Cross-validation

def cross_validate(
    train_image_dir: str,
    train_mask_dir:  str,
    val_image_dir:   str,
    val_mask_dir:    str,
    k_folds:         int    = 5,
    epochs:          int    = 80,
    batch_size:      int    = 4,
    lr:              float  = 1e-4,
    output_dir:      str    = "crossval_output"
):
    val_every     = 5   
    fullres_every = 10 

    city_root = os.path.dirname(os.path.dirname(train_image_dir))
    print("Cityscapes root inferred as:", city_root)

    base_train     = CityscapesAlb(
        root=city_root,
        split="train",
        mode="fine",
        target_type="semantic",
        transforms=train_transform
    )
    base_train_val = CityscapesAlb(
        root=city_root,
        split="train",
        mode="fine",
        target_type="semantic",
        transforms=val_transform
    )

    num_samples = len(base_train)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    fold_sizes = np.full(k_folds, num_samples // k_folds, dtype=int)
    fold_sizes[: (num_samples % k_folds)] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        start, end = current, current + fold_size
        folds.append(indices[start:end])
        current = end

    best_lowres_mious = []
    fullres_mious     = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold_i in range(k_folds):
        print(f"\n=== Starting fold {fold_i + 1}/{k_folds} ===")
        val_idx   = folds[fold_i]
        train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != fold_i])

        train_subset = Subset(base_train, train_idx)
        val_subset   = Subset(base_train_val, val_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        model = UNetResNet(num_classes=19, pretrained=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )
        criterion = nn.CrossEntropyLoss(ignore_index=255)

        train_losses = []
        val_mious    = []
        best_val_miou = 0.0

        fold_dir = os.path.join(output_dir, f"fold_{fold_i + 1}")
        os.makedirs(fold_dir, exist_ok=True)

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            loader_iter = tqdm(train_loader, desc=f"Fold {fold_i+1}  Epoch {epoch}/{epochs}", unit="batch")
            for batch_idx, (images, masks) in enumerate(loader_iter, start=1):
                images = images.to(device, non_blocking=True)
                masks  = masks.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits = model(images)        
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                loader_iter.set_postfix(loss=(running_loss / batch_idx))

            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            print(f"→ Fold {fold_i+1}  Epoch {epoch}/{epochs}  Train Loss: {avg_train_loss:.4f}")

            if (epoch % val_every) == 0 or (epoch == epochs):
                val_miou = validate_model(model, val_loader, device)
                val_mious.append(val_miou)
                print(f"→ Fold {fold_i+1}  Epoch {epoch}/{epochs}  Val mIoU (256×512): {val_miou:.4f}")
                scheduler.step(val_miou)

                if val_miou > best_val_miou:
                    best_val_miou = val_miou
            else:
                val_mious.append(val_mious[-1] if val_mious else 0.0)

            if (epoch % fullres_every) == 0 or (epoch == epochs):
                full_miou = evaluate_full_res(model, city_root, device)
                print(f"→ Fold {fold_i+1}  Epoch {epoch}/{epochs}  FULL-RES Val mIoU (1024×2048): {full_miou:.4f}")

        plot_training_history(train_losses, val_mious, output_dir=fold_dir)

        model.eval()
        with torch.no_grad():
            sample_images, sample_masks = next(iter(val_loader))
            sample_images = sample_images.to(device)
            sample_masks  = sample_masks.to(device)
            sample_logits = model(sample_images)
            sample_preds  = sample_logits.argmax(dim=1)

            img0  = sample_images[0].cpu()
            m0    = sample_masks[0].cpu()
            p0    = sample_preds[0].cpu()
            vis_path = os.path.join(fold_dir, "final_prediction.png")
            visualize(img0, m0, p0, vis_path)
            print(f"Saved single-sample visualization at: {vis_path}")

        ckpt_path = os.path.join(fold_dir, f"unet_resnet50_fold{fold_i+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved model checkpoint at: {ckpt_path}")

        best_lowres_mious.append(best_val_miou)

        final_fullres_miou = evaluate_full_res(model, city_root, device)
        fullres_mious.append(final_fullres_miou)
        print(f"→ Fold {fold_i+1}  FINAL FULL-RES Val mIoU = {final_fullres_miou:.4f}")

    best_lowres_arr = np.array(best_lowres_mious)
    fullres_arr     = np.array(fullres_mious)

    mean_lowres = best_lowres_arr.mean()
    std_lowres  = best_lowres_arr.std()
    mean_full   = fullres_arr.mean()
    std_full    = fullres_arr.std()

    print("\n=== Cross-Validation Summary ===")
    print(f"Low‐Res Val mIoU per fold: {best_lowres_mious}")
    print(f"  → mean = {mean_lowres:.4f}, std = {std_lowres:.4f}")
    print(f"Full‐Res Val mIoU per fold: {fullres_mious}")
    print(f"  → mean = {mean_full:.4f}, std = {std_full:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    summary_csv = os.path.join(output_dir, "summary_metrics.csv")
    with open(summary_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold_id", "best_lowres_mIoU", "fullres_mIoU"])
        for i in range(k_folds):
            writer.writerow([i+1, f"{best_lowres_mious[i]:.4f}", f"{fullres_mious[i]:.4f}"])
        writer.writerow(["mean", f"{mean_lowres:.4f}", f"{mean_full:.4f}"])
        writer.writerow(["std",  f"{std_lowres:.4f}",  f"{std_full:.4f}"])
    print(f"Summary metrics written to: {summary_csv}")

if __name__ == "__main__":
    # Paths to your Cityscapes “train” and “val” folders
    TRAIN_IMAGE_DIR = "cityscapes_data/leftImg8bit/train"
    TRAIN_MASK_DIR  = "cityscapes_data/gtFine/train"
    VAL_IMAGE_DIR   = "cityscapes_data/leftImg8bit/val"
    VAL_MASK_DIR    = "cityscapes_data/gtFine/val"

    # adaptable via k_folds parameter
    cross_validate(
        train_image_dir=TRAIN_IMAGE_DIR,
        train_mask_dir=TRAIN_MASK_DIR,
        val_image_dir=VAL_IMAGE_DIR,
        val_mask_dir=VAL_MASK_DIR,
        k_folds=3,        
        epochs=65,        # epochs per fold
        batch_size=2,
        lr=1e-4,
        output_dir="cv_results"   # all fold subfolders + summary CSV go here
    )
