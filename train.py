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
# 1. CLASSI DI SUPPORTO (METRICHE E LOSS)
# ==========================================

class IoUMetric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.conf_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)

    def update(self, preds, targets):
        preds = preds.detach().cpu().view(-1)
        targets = targets.detach().cpu().view(-1)
        mask = (targets >= 0) & (targets < self.num_classes)
        self.conf_matrix += torch.bincount(
            self.num_classes * targets[mask] + preds[mask], 
            minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self):
        intersection = torch.diag(self.conf_matrix)
        union = self.conf_matrix.sum(0) + self.conf_matrix.sum(1) - intersection
        iou_per_class = intersection.float() / (union.float() + 1e-6)
        valid_mask = union > 0
        mean_iou = iou_per_class[valid_mask].mean().item() if valid_mask.any() else 0.0
        return iou_per_class.tolist(), mean_iou

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        return (1 - (2. * intersection + self.smooth) / (cardinality + self.smooth)).mean()

# ==========================================
# 2. ARCHITETTURA E DATASET
# ==========================================

class CropTemporalDataset(Dataset):
    def __init__(self, file_list, data_dir, means, stds, augment=False):
        self.img_dir = Path(data_dir) / "images"
        self.mask_dir = Path(data_dir) / "masks"
        self.files = file_list
        self.means = np.array(means, dtype=np.float32).reshape(1, 6, 1, 1)
        self.stds = np.array(stds, dtype=np.float32).reshape(1, 6, 1, 1)
        t = [A.Resize(224, 224)]
        if augment:
            t.extend([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)])
        self.transform = A.Compose(t)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        image = np.load(self.img_dir / f"{fname}.npy").astype(np.float32)
        with rasterio.open(self.mask_dir / f"{fname}.tif") as src:
            mask = src.read(1).astype(np.int64)
        image = (image - self.means) / (self.stds + 1e-6)
        T, C, H, W = image.shape
        combined = image.reshape(-1, H, W).transpose(1, 2, 0)
        aug = self.transform(image=combined, mask=mask)
        return torch.from_numpy(aug['image'].transpose(2, 0, 1).reshape(T, C, 224, 224)), torch.from_numpy(aug['mask'])

class ResidualUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch)
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
        d = backbone.out_channels[-1]
        self.decoder = nn.Sequential(
            ResidualUpBlock(d, 256), ResidualUpBlock(256, 128),
            ResidualUpBlock(128, 64), ResidualUpBlock(64, 32),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
    def forward(self, x):
        B, T, C, H, W = x.shape
        feats = self.backbone(x.permute(0, 2, 1, 3, 4))
        tokens = feats[-1][:, 1:, :]
        grid = int(np.sqrt(tokens.shape[1]))
        logits = self.decoder(tokens.transpose(1, 2).reshape(B, -1, grid, grid))
        return F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)

# ==========================================
# 3. TRAINING LOOP COMPLETO
# ==========================================

def train():
    with open('config.json', 'r') as f: config = json.load(f)
    device = torch.device("cuda")
    p = config["training_params"]
    
    # Dataset
    img_files = [f.stem for f in (Path(config["paths"]["input_dir"]) / "images").glob("*.npy")]
    train_f, val_f = train_test_split(img_files, test_size=0.15, random_state=42)
    ds_kwargs = {"data_dir": config["paths"]["input_dir"], "means": config["data_specs"]["normalization"]["means"], "stds": config["data_specs"]["normalization"]["stds"]}
    
    train_loader = DataLoader(CropTemporalDataset(train_f, augment=True, **ds_kwargs), batch_size=p["batch_size"], shuffle=True, num_workers=p["num_workers"], pin_memory=True)
    val_loader = DataLoader(CropTemporalDataset(val_f, augment=False, **ds_kwargs), batch_size=p["batch_size"], num_workers=p["num_workers"], pin_memory=True)

    # Modello
    backbone = BACKBONE_REGISTRY.build(config["project_meta"]["backbone_model"], pretrained=True)
    model = PrithviSegmentation4090(backbone, config["data_specs"]["num_classes"]).to(device)
    if p["compile"]: model = torch.compile(model)

    # Loss & Pesi (Aggiornati sulla tua distribuzione pixel)
    weights = torch.tensor([0.2, 1.2, 2.5, 2.3, 1.8, 0.6, 4.5, 12.0, 2.2]).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=weights)
    dice_loss_fn = DiceLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=p["plateau_patience"])

    best_miou = 0
    early_stop_cnt = 0
    history = []
    class_names = ["Sfondo", "Olivo", "Vite", "Agrumi", "Frutteto", "Grano", "Legumi", "Ortaggi", "Incolto"]
    iou_tracker = IoUMetric(config["data_specs"]["num_classes"])

    print(f"\n🚀 START TRAINING: RTX 4090 - {p['precision'].upper()} MODE")

    for epoch in range(p["num_epochs"]):
        model.train()
        train_l = 0
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{p['num_epochs']}")
        
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(imgs)
                loss = ce_loss(out, masks) + dice_loss_fn(out, masks)
            
            loss.backward()
            optimizer.step()
            train_l += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validazione
        model.eval()
        val_l = 0
        iou_tracker.reset() # FIX: chiamata separata
        
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device).long()
                out = model(imgs)
                val_l += (ce_loss(out, masks) + dice_loss_fn(out, masks)).item()
                iou_tracker.update(out.argmax(1), masks)

        ious, miou = iou_tracker.compute()
        avg_train_l, avg_val_l = train_l/len(train_loader), val_l/len(val_loader)
        
        print(f"\n📊 REPORT EPOCA {epoch+1} | Val Loss: {avg_val_l:.4f} | mIoU: {miou:.4f}")
        for i, name in enumerate(class_names): print(f"  - {name.ljust(10)}: {ious[i]:.4f}")

        history.append({"epoch": epoch+1, "train_loss": avg_train_l, "val_loss": avg_val_l, "miou": miou})
        pd.DataFrame(history).to_csv("training_metrics.csv", index=False)

        # Early Stopping & Checkpoint
        scheduler.step(avg_val_l)
        if miou > best_miou:
            best_miou = miou
            early_stop_cnt = 0
            torch.save(model.state_dict(), config["paths"]["model_save_path"])
            print("⭐ NUOVO RECORD mIoU! Modello salvato.")
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= p["early_stop_patience"]:
                print(f"🛑 EARLY STOPPING attivata all'epoca {epoch+1}")
                break

        gc.collect()
        torch.cuda.empty_cache()

    # Salvataggio Grafici
    df = pd.DataFrame(history)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.plot(df['train_loss'], label='Train'); plt.plot(df['val_loss'], label='Val'); plt.title('Combined Loss'); plt.legend()
    plt.subplot(1,2,2); plt.plot(df['miou'], label='mIoU', color='green'); plt.title('Mean IoU Performance'); plt.legend()
    plt.savefig("training_results_plot.png")
    print("\n✅ Training completato. Grafici salvati in training_results_plot.png")

if __name__ == "__main__":
    train()