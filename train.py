import os
import json
import warnings
import gc
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A
from torch.utils.data import DataLoader, Dataset
from terratorch.registry import BACKBONE_REGISTRY

warnings.filterwarnings('ignore')

# ==========================================
# 1. CARICAMENTO CONFIGURAZIONE
# ==========================================
with open('config.json', 'r') as f:
    CONFIG_JSON = json.load(f)

# Indici Sentinel-2: [B02, B03, B04, B8A, B11, B12]
BAND_INDICES = [0, 1, 2, 7, 8, 9]

# ==========================================
# 2. DATASET MULTI-TEMPORALE (T=4)
# ==========================================
class CropTemporalDataset(Dataset):
    def __init__(self, file_list, data_dir, means, stds, augment=False):
        self.img_dir = Path(data_dir) / "images"
        self.mask_dir = Path(data_dir) / "masks"
        self.files = file_list
        self.means = np.array(means, dtype=np.float32)[BAND_INDICES].reshape(1, 6, 1, 1)
        self.stds = np.array(stds, dtype=np.float32)[BAND_INDICES].reshape(1, 6, 1, 1)

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        ]) if augment else None

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        image = np.load(self.img_dir / f"{fname}.npy").astype(np.float32)
        with rasterio.open(self.mask_dir / f"{fname}.tif") as src:
            mask = src.read(1).astype(np.int64)

        image = (image - self.means) / (self.stds + 1e-6)

        if self.transform:
            T, C, H, W = image.shape
            combined = image.reshape(-1, H, W).transpose(1, 2, 0)
            augmented = self.transform(image=combined, mask=mask)
            image = augmented['image'].transpose(2, 0, 1).reshape(T, C, H, W)
            mask = augmented['mask']

        return torch.from_numpy(image), torch.from_numpy(mask)

# ==========================================
# 3. MODELLO (PRITHVI + RESIDUAL DECODER)
# ==========================================
class ResidualUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.up(x)
        return F.relu(self.conv(x) + self.shortcut(x))

class PrithviSegmentation4090(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        d = backbone.out_channels[-1] if isinstance(backbone.out_channels, list) else backbone.out_channels
        self.decoder = nn.Sequential(
            ResidualUpBlock(d, 256),   # 16->32
            ResidualUpBlock(256, 128), # 32->64
            ResidualUpBlock(128, 64),  # 64->128
            ResidualUpBlock(64, 32),   # 128->256
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4) # [B, C, T, H, W]
        feats = self.backbone(x)
        tokens = feats[-1][:, 1:, :] # No CLS
        B, N, D = tokens.shape
        x_spat = tokens.reshape(B, -1, 16, 16, D).mean(dim=1).permute(0, 3, 1, 2)
        return self.decoder(x_spat)

# ==========================================
# 4. TRAINING LOOP CON MONITORING
# ==========================================
def train():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    params = CONFIG_JSON["training_params"]

    print(f"\n🚀 START: RTX 4090 - {params['precision'].upper()} MODE")
    
    # Dataloaders
    img_files = [f.stem for f in (Path(CONFIG_JSON["paths"]["input_dir"]) / "images").glob("*.npy")]
    train_files, val_files = train_test_split(img_files, test_size=0.15, random_state=42)
    
    train_loader = DataLoader(CropTemporalDataset(train_files, CONFIG_JSON["paths"]["input_dir"], CONFIG_JSON["data_specs"]["normalization"]["means"], CONFIG_JSON["data_specs"]["normalization"]["stds"], augment=True),
                              batch_size=params["batch_size"], shuffle=True, num_workers=params["num_workers"], pin_memory=True)
    val_loader = DataLoader(CropTemporalDataset(val_files, CONFIG_JSON["paths"]["input_dir"], CONFIG_JSON["data_specs"]["normalization"]["means"], CONFIG_JSON["data_specs"]["normalization"]["stds"], augment=False),
                            batch_size=params["batch_size"], shuffle=False, num_workers=params["num_workers"], pin_memory=True)

    # Modello & Ottimizzazione
    backbone = BACKBONE_REGISTRY.build("terratorch_prithvi_eo_v2_tiny_tl", pretrained=True)
    model = PrithviSegmentation4090(backbone, CONFIG_JSON["data_specs"]["num_classes"]).to(device)
    if params["compile"]: model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    
    # --- MONITOR 1: ReduceLROnPlateau ---
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=params["plateau_patience"], min_lr=params["min_lr"], verbose=True
    )
    
    weights = torch.tensor([0.1, 1.5, 1.0, 1.5, 1.9, 1.4, 2.9, 4.2, 1.6]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # --- MONITOR 2: Early Stopping & Checkpoint ---
    best_val_loss = float('inf')
    early_stop_counter = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "miou": []}

    for epoch in range(params["num_epochs"]):
        model.train()
        train_loss = 0
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{params['num_epochs']}")
        
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(imgs)
                loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validazione
        model.eval()
        val_loss, ious = 0, []
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                val_loss += criterion(logits, masks).item()
                preds = logits.argmax(dim=1)
                for c in range(CONFIG_JSON["data_specs"]["num_classes"]):
                    inter = ((preds == c) & (masks == c)).sum().item()
                    union = ((preds == c) | (masks == c)).sum().item()
                    if union > 0: ious.append(inter/union)
        
        avg_val_loss = val_loss / len(val_loader)
        miou = np.mean(ious)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"📊 Val Loss: {avg_val_loss:.4f} | mIoU: {miou:.4f} | LR: {current_lr:.2e}")

        # Update Scheduler
        scheduler.step(avg_val_loss)

        # Early Stopping & Checkpoint Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), CONFIG_JSON["paths"]["model_save_path"])
            print("⭐ Best model saved!")
        else:
            early_stop_counter += 1
            print(f"⚠️ No improvement for {early_stop_counter} epochs.")

        if early_stop_counter >= params["early_stop_patience"]:
            print(f"🛑 Early stopping triggered at epoch {epoch+1}")
            break

        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    train()