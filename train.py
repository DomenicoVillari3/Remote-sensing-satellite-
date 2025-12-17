# %%
#!pip install terratorch torch torchvision rasterio albumentations tqdm scikit-learn wandb


# %%
import os
import json
import warnings
warnings.filterwarnings('ignore')
import gc
import tqdm
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
import rasterio
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A
from torch.utils.data import DataLoader,Dataset
# TerraTorch imports
from terratorch.cli_tools import LightningInferenceModel
from terratorch.registry import BACKBONE_REGISTRY




from terratorch.tasks import PixelwiseRegressionTask, SemanticSegmentationTask
from terratorch.datasets import GenericNonGeoPixelwiseRegressionDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, CSVLogger


# %%
# ==========================================
# 1. CONFIGURAZIONE
# ==========================================

with open('config.json', 'r') as f:
    json_config = json.load(f)

CONFIG = {
    "DATA_DIR": json_config["paths"]['input_dir'],
    "MODEL_SAVE_DIR": Path(json_config["paths"]["model_save_path"]).parent,
    "NUM_CLASSES": json_config["data_specs"]["num_classes"],
    "IMG_SIZE": 224,  # Mantieni 224 per compatibilità Prithvi
    
    # OTTIMIZZAZIONI CRITICHE PER 4GB
    "BATCH_SIZE": 1,  # Mini-batch singolo
    "GRAD_ACCUM_STEPS": 16,  # Simula batch_size=16
    "EFFECTIVE_BATCH_SIZE": 16,  # 1 * 16
    
    "EPOCHS": json_config["training_params"]["num_epochs"],
    "NUM_WORKERS": 2,  # Ridotto per risparmiare RAM
    "LEARNING_RATE": json_config["training_params"]["learning_rate"],
    
    # Normalizzazione (solo 6 bande HLS)
    "MEANS": json_config["data_specs"]["normalization"]["means"],
    "STDS": json_config["data_specs"]["normalization"]["stds"],
    
    # Model settings
    "PRETRAINED_WEIGHTS": "ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL",
    "USE_CHECKPOINT": True,  # Gradient checkpointing
    "MIXED_PRECISION": True,  # FP16
}

print("="*70)
print("🌍 CROP SEGMENTATION - Prithvi-100M (4GB GPU Optimized)")
print("="*70)
print(f"✓ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"✓ Batch Size: {CONFIG['BATCH_SIZE']} x {CONFIG['GRAD_ACCUM_STEPS']} = {CONFIG['EFFECTIVE_BATCH_SIZE']}")
print(f"✓ Mixed Precision: {CONFIG['MIXED_PRECISION']}")
print("="*70 + "\n")
# Pulizia memoria iniziale
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None


# %%
# ==========================================
# 2. DATASET OTTIMIZZATO - FIXED
# ==========================================

class CropDataset(Dataset):
    """Dataset per Sentinel-2 crop segmentation con resize automatico"""
    
    def __init__(self, file_list, root_dir, means, stds, img_size=224, augment=False):
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / "images"
        self.mask_dir = self.root_dir / "masks"
        self.files = file_list
        self.img_size = img_size
        self.augment = augment
        
        # Selezione 6 bande HLS
        self.band_indices = [0, 1, 2, 7, 8, 9]
        
        self.means = np.array(means, dtype=np.float32)[self.band_indices]
        self.stds = np.array(stds, dtype=np.float32)[self.band_indices]
        
        # Augmentation + Resize
        transforms_list = []
        
        if augment:
            transforms_list.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, 
                    scale_limit=0.05, 
                    rotate_limit=10, 
                    border_mode=0,
                    p=0.3
                ),
            ])
        
        # SEMPRE applica resize a img_size
        transforms_list.append(
            A.Resize(height=self.img_size, width=self.img_size, interpolation=1)
        )
        
        self.transform = A.Compose(transforms_list)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filename = self.files[idx]
        
        # Carica 6 bande
        with rasterio.open(self.img_dir / filename) as src:
            image = src.read([i+1 for i in self.band_indices]).astype(np.float32)
        
        # Carica mask
        with rasterio.open(self.mask_dir / filename) as src:
            mask = src.read(1).astype(np.int64)
        
        # Transpose per albumentations (H,W,C)
        image = np.transpose(image, (1, 2, 0))
        
        # Applica transform (augmentation + resize)
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # Normalizzazione
        image = (image - self.means.reshape(1, 1, 6)) / (self.stds.reshape(1, 1, 6) + 1e-6)
        
        # Ritorna a (C,H,W)
        image = np.transpose(image, (2, 0, 1))
        
        #restituisci tensori (6,224,224)
        return torch.from_numpy(image).float(), torch.from_numpy(mask).long()


# %%
class PrithviSegmentationModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        
        # Per Prithvi-100M TL embed_dim = 768
        # TerraTorch espone out_channels come lista di dimensioni token → prendiamo embed_dim
        if isinstance(getattr(backbone, "out_channels", None), (list, tuple)):
            encoder_channels = backbone.out_channels[-1]  # 768
        else:
            encoder_channels = backbone.out_channels

        self.encoder_channels = encoder_channels  # salva per debug

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, images_6chw):
        """
        images_6chw: [B, 6, H, W]
        backbone Prithvi TerraTorch: input [B, C=6, T, H, W] con T=1
        output (ultimo livello): [B, N, D] con N = 1 + H'*W' (CLS + patch)
        """
        B, C, H, W = images_6chw.shape

        # 1) aggiungi dimensione temporale T=1 sull'asse 2 → [B, 6, 1, H, W]
        x_in = images_6chw.unsqueeze(2)

        # 2) forward nel backbone
        feats = self.backbone(x_in)  # lista di feature

        # L’ultima entry è [B, N, D] = [B, 197, 768]
        if isinstance(feats, (list, tuple)):
            tokens = feats[-1]          # [B, N, D]
        else:
            tokens = feats              # fallback

        # 3) da token a feature map 2D
        Btok, N, D = tokens.shape      # N=197, D=768

        # rimuovi CLS token (primo token) → N_patches = 196
        tokens_no_cls = tokens[:, 1:, :]          # [B, 196, 768]
        N_patches = tokens_no_cls.shape[1]        # 196

        # griglia 14x14
        H_feat = W_feat = int(np.sqrt(N_patches))  # 14
        assert H_feat * W_feat == N_patches, \
            f"Non riesco a mappare N={N_patches} in griglia quadrata"

        # reshape: [B, 196, 768] → [B, 768, 14, 14]
        x = tokens_no_cls.permute(0, 2, 1).reshape(B, D, H_feat, W_feat)

        # 4) decoder 2D: upsample 14x14 → 224x224
        logits = self.decoder(x)  # [B, num_classes, H_dec, W_dec]

        # 5) resize a H,W originali se necessario
        if logits.shape[2:] != (H, W):
            logits = F.interpolate(
                logits, size=(H, W), mode="bilinear", align_corners=False
            )

        return logits


# %%
# Pesi calcolati approssimativamente sui tuoi dati
# Ordine: Class 0 (Background?), Class 1, Class 2, ... Class 8
# Se la Classe 0 non esiste o è background, gestiscila. Assumo qui classi 1-8.
# Normalizziamo affinché la classe più frequente (2) abbia peso ~1.0

# Counts: [?, 4670, 7057, 4556, 3596, 4751, 2436, 1678, 4273]
# Weights inversi (più è bassa la count, più alto il peso)
class_weights_list = [
    0.1,  # Class 0 (Se è background/nodata spesso si mette a 0 o basso)
    1.5,  # Class 1
    1.0,  # Class 2 (La più frequente = riferimento)
    1.55, # Class 3
    1.96, # Class 4
    1.48, # Class 5
    2.90, # Class 6
    4.20, # Class 7 (La più rara -> Peso più alto!)
    1.65  # Class 8
]

# %% [markdown]
# 1. Quando usi TerraTorch / BACKBONE_REGISTRY, il backbone Prithvi si aspetta:
# 	-	 pixel_values :dove:
# 	-	: batch size
# 	-	: numero di time‑steps (immagini nel tempo)
# 	-	: numero di bande (per 100M‑TL = 6 canali, non 10)
# 	-	: dimensioni spaziali (tipicamente 224×224 dopo crop)
# 
# 2. Esempio “canonical” usato nel paper per Prithvi‑2.0:
# 	-	Input grezzo: ,  → shape 
# 	-	Augmentation: random crop a 224×224 → 
# 	-	Eventuali flip orizzontali.
# 3. Nel tuo caso, se hai una sola data per chip:
# 	-	userai 
# 	-	quindi l’input al backbone sarà , dopo aver selezionato 6 bande dalle tue 10.
# 4. Banda / canali
# 	- Prithvi‑EO‑2.0‑100M‑TL tiny usa 6 bande HLS codificate in  config.json  del modello:
# 	- bands :  "B02", "B03", "B04", "B05", "B06", "B07"  oppure per alcune varianti B02,B03,B04,B08A,B11,B12 (dipende dalla release, ma sempre 6).Per la versione TL su HuggingFace è esplicitamente indicato  in_chans: 6 .
# 	Quindi: C = 6 per il backbone, non 10. Tu hai 10 bande → devi mappare le tue 10 bande S2 alle 6 che Prithvi conosce (tipicamente B2,B3,B4,B8A,B11,B12).

# %%
# ==========================================
# 4. LOSS & METRICHE
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        probs = F.softmax(inputs, dim=1)
        targets_oh = F.one_hot(targets, num_classes).permute(0,3,1,2).float()
        intersection = (probs * targets_oh).sum((0,2,3))
        cardinality = probs.sum((0,2,3)) + targets_oh.sum((0,2,3))
        dice = (2*intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, class_weights=None, focal_weight=0.5, dice_weight=0.5):
        super().__init__()
        alpha = class_weights if class_weights is not None else None
        self.focal = FocalLoss(alpha=alpha, gamma=2.0)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, inputs, targets):
        return self.focal_weight * self.focal(inputs, targets) + \
               self.dice_weight * self.dice(inputs, targets)



def calculate_iou(pred, target, num_classes):
    """IoU per classe (usato in validazione)."""
    pred = pred.view(-1)
    target = target.view(-1)
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        inter = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((inter / union).item())
    return ious


# %%
# ==========================================
# 3. CREAZIONE DATALOADER PER PRITHVI
# ==========================================

def create_dataloaders_for_prithvi(
    data_dir,
    means_10,
    stds_10,
    img_size=224,
    batch_size=1,
    num_workers=2,
    val_ratio=0.2,
    add_time_dim=False,
):
    """
    Crea train_loader e val_loader usando CropDataset.
    Output:
      - se add_time_dim=False: images [B, 6, H, W]
      - se add_time_dim=True:  images [B, 1, 6, H, W]
    """
    data_dir = Path(data_dir)
    img_dir = data_dir / "images"
    mask_dir = data_dir / "masks"

    # trova tutti i .tif
    img_files = sorted([f.name for f in img_dir.glob("*.tif")])

    # tieni solo quelli che hanno una mask corrispondente
    all_files = []
    for fname in img_files:
        if (mask_dir / fname).is_file():
            all_files.append(fname)
    else:
        print(f"[WARN] Mask mancante per {fname}, salto.")

    print(f"Totale chip con mask: {len(all_files)}")

    # stratified split se possibile
    try:
        labels = [int(f.split("_class")[1].split("_")[0]) for f in all_files]
        stratify = labels
    except Exception:
        stratify = None

    train_files, val_files = train_test_split(
        all_files,
        test_size=val_ratio,
        random_state=42,
        stratify=stratify,
    )

    print(f"Train: {len(train_files)} | Val: {len(val_files)}")

    # dataset (usa il CropDataset che hai già definito)
    train_dataset = CropDataset(
        train_files,
        data_dir,
        means_10,
        stds_10,
        img_size=img_size,
        augment=True,
    )
    val_dataset = CropDataset(
        val_files,
        data_dir,
        means_10,
        stds_10,
        img_size=img_size,
        augment=False,
    )

    # dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# %%
# ==========================================
# 5. TRAINING LOOP con LR scheduler dinamico + early stopping
# ==========================================

def train():
    print("\n🚀 INIZIO TRAINING Prithvi-100M + Decoder\n")
    history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "miou": [],
    }

    
    # ---- DataLoader ----
    train_loader, val_loader = create_dataloaders_for_prithvi(
        data_dir=CONFIG["DATA_DIR"],
        means_10=CONFIG["MEANS"],
        stds_10=CONFIG["STDS"],
        img_size=CONFIG["IMG_SIZE"],
        batch_size=CONFIG["BATCH_SIZE"],
        num_workers=CONFIG["NUM_WORKERS"],
        val_ratio=0.2,
        add_time_dim=False
    )
    
    device = "cpu"
    
    print("🔧 Carico backbone Prithvi-EO-2.0 Tiny-TL (TerraTorch)...")
    backbone = BACKBONE_REGISTRY.build(
        "terratorch_prithvi_eo_v2_tiny_tl",
        pretrained=True
    )
    model = PrithviSegmentationModel(backbone, CONFIG["NUM_CLASSES"]).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Parametri totali modello: {total_params/1e6:.1f}M")
    
    # ---- Loss & Optimizer ----
    class_weights = torch.tensor(class_weights_list, dtype=torch.float32).to(device)
    criterion = CombinedLoss(class_weights=class_weights, focal_weight=0.5, dice_weight=0.5)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["LEARNING_RATE"],
        weight_decay=0.01
    )
    
    # Scheduler dinamico: riduce LR se val_loss non migliora
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
    
    # Mixed precision off su CPU
    CONFIG["MIXED_PRECISION"] = False
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    
    best_miou = 0.0
    best_val_loss = float('inf')
    patience = 7          # early stopping patience
    epochs_no_improve = 0
    
    for epoch in range(CONFIG["EPOCHS"]):
        # ========= TRAIN =========
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']} [TRAIN]")
        
        for step, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            with torch.cuda.amp.autocast(enabled=False):
                logits = model(images)
                loss = criterion(logits, masks)
                loss = loss / CONFIG["GRAD_ACCUM_STEPS"]
            
            loss.backward()
            
            if (step + 1) % CONFIG["GRAD_ACCUM_STEPS"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            train_loss += loss.item() * CONFIG["GRAD_ACCUM_STEPS"]
            pbar.set_postfix({"loss": f"{loss.item() * CONFIG['GRAD_ACCUM_STEPS']:.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # ========= VALIDATION =========
        model.eval()
        val_loss = 0.0
        all_ious = []
        
        with torch.no_grad():
            for images, masks in tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']} [VAL]  "):
                images = images.to(device)
                masks = masks.to(device)
                
                logits = model(images)
                loss = criterion(logits, masks)
                
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                ious = calculate_iou(preds, masks, CONFIG["NUM_CLASSES"])
                all_ious.append(ious)
        
        avg_val_loss = val_loss / len(val_loader)
        mean_ious = np.nanmean(all_ious, axis=0)
        miou = np.nanmean(mean_ious)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["miou"].append(miou)

        
        # Scheduler dinamico sul val_loss
        scheduler.step(avg_val_loss)
        
        # ---- logging epoca ----
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{CONFIG['EPOCHS']}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  mIoU:       {miou:.4f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.2e}")
        print("  Per-class IoU:")
        for cid, iou in enumerate(mean_ious):
            if not np.isnan(iou):
                print(f"    Class {cid}: {iou:.4f}")
        print(f"{'='*70}\n")
        
        # ---- salvataggio best model (basato su mIoU) ----
        if miou > best_miou:
            best_miou = miou
        
        # ---- early stopping basato su val_loss ----
        if avg_val_loss < best_val_loss :
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            
            CONFIG["MODEL_SAVE_DIR"].mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "miou": miou,
                    "class_ious": mean_ious.tolist(),
                    "config": CONFIG,
                },
                CONFIG["MODEL_SAVE_DIR"] / "best_prithvi_tiny_seg.pth",
            )
            print(f"✓ Nuovo best model salvato (ValLoss={best_val_loss:.4f}, mIoU={miou:.4f})\n")
        else:
            epochs_no_improve += 1
            print(f"⚠ Nessun miglioramento per {epochs_no_improve} epoche.")
        
        if epochs_no_improve >= patience:
            print(f"\n⏹ Early stopping dopo {patience} epoche senza miglioramento.")
            break
        
        gc.collect()

    

# salva CSV con le metriche
    metrics_path = CONFIG["MODEL_SAVE_DIR"] / "training_metrics.csv"
    pd.DataFrame(history).to_csv(metrics_path, index=False)
    print(f"✓ Metriche salvate in {metrics_path}")

    # plot loss
    plt.figure(figsize=(8,4))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(CONFIG["MODEL_SAVE_DIR"] / "loss_curves.png", dpi=150)
    plt.close()

    # plot mIoU
    plt.figure(figsize=(8,4))
    plt.plot(history["epoch"], history["miou"], label="mIoU", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.title("Validation mIoU")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(CONFIG["MODEL_SAVE_DIR"] / "miou_curve.png", dpi=150)
    plt.close()

    print("✓ Curve di loss e mIoU salvate come PNG")


    print("\n✅ TRAINING COMPLETATO")
    print(f"Best mIoU: {best_miou:.4f}")
    print(f"Best Val Loss: {best_val_loss:.4f}")


# %% [markdown]
# Logica del training (schematizzata)
# 1.	Dataset & DataLoader
# 	- 	Ogni sample: immagine  → selezioni 6 bande HLS → augment + resize a  → normalizzi → tensore .
# 	- 	DataLoader produce batch  images: B, 6, 224, 224 ,  masks: B, 224, 224 .
# 2.	Backbone + Decoder
# 	- 	Nel  forward :
# 	- 	 images  → aggiungi time‑dim :  B, 1, 6, 224, 224 
# 	- 	passi a Prithvi‐100M‐TL via  BACKBONE_REGISTRY , che restituisce feature map 2D  B, D, H', W' .
# 	- 	un decoder leggero (serie di  ConvTranspose2d ) upsampled a  B, num_classes, 224, 224 .
# 3.	Loss e class imbalance
# 	- 	Usi una loss composta: Focal + Dice, pesata con i  class_weights_list  (frequenze inverse delle classi).
# 	- 	Focal concentra il gradiente sui pixel difficili / classi rare, Dice ottimizza l’overlap spaziale (IoU).
# 4.	Ottimizzazione con 4GB GPU
# 	- 	 BATCH_SIZE = 1  e  GRAD_ACCUM_STEPS=16  → effettivo batch 16, ma un solo sample in VRAM per volta.
# 	- 	Mixed precision ( autocast + GradScaler ) dimezza la memoria.
# 	- 	Gradient clipping e CosineAnnealingLR per stabilità e convergenza più dolce.

# %%
if __name__ == "__main__":
    train()


# %%
from terratorch.registry import BACKBONE_REGISTRY

#print([name for name in BACKBONE_REGISTRY])  # lista stringhe disponibili
print([n for n in BACKBONE_REGISTRY if "prithvi" in n and "tiny"])  # solo modelli Prithvi



