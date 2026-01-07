import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import rasterio
from terratorch.registry import BACKBONE_REGISTRY
import torch.nn.functional as F
from PIL import Image
from train import PrithviSegmentation4090

# ==========================================
# CONFIGURAZIONE COLORI E CLASSI
# ==========================================
CLASS_COLORS = [
    [0, 0, 0],       # 0: Sfondo (Trasparente)
    [50, 255, 50],   # 1: Olivo (Verde Neon)
    [255, 0, 255],   # 2: Vite (Magenta)
    [255, 140, 0],   # 3: Agrumi (Arancione)
    [0, 102, 255],   # 4: Frutteto (Blu Elettrico)
    [255, 255, 0],   # 5: Grano (Giallo)
    [0, 255, 255],   # 6: Legumi (Cyan)
    [255, 0, 0],     # 7: Ortaggi (Rosso)
    [255, 255, 255]  # 8: Incolto (Bianco)
]

CLASS_NAMES = ["Sfondo", "Olivo", "Vite", "Agrumi", "Frutteto", "Grano", "Legumi", "Ortaggi", "Incolto"]

def colorize_mask_rgb(mask):
    """Trasforma indici in RGB solido"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(CLASS_COLORS):
        rgb[mask == i] = color
    return rgb

def colorize_mask_rgba(mask, opacity=0.6):
    """Trasforma indici in RGBA per overlay"""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for i, color in enumerate(CLASS_COLORS):
        if i == 0: continue # Sfondo resta trasparente
        norm_color = [c / 255.0 for c in color]
        class_mask = (mask == i)
        rgba[class_mask, 0:3] = norm_color
        rgba[class_mask, 3] = opacity
    return rgba

def run_pro_inference(num_samples=10):
    # 1. Setup
    with open('config.json', 'r') as f: config = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(config["paths"]["input_dir"])
    model_path = config["paths"]["model_save_path"]
    
    # 2. Caricamento Modello
    print(f"🔄 Caricamento modello da: {model_path}")
    backbone = BACKBONE_REGISTRY.build(config["project_meta"]["backbone_model"], pretrained=False)
    model = PrithviSegmentation4090(backbone, config["data_specs"]["num_classes"])
    
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device).eval()
    
    means = torch.tensor(config["data_specs"]["normalization"]["means"]).view(1, 1, 6, 1, 1).to(device).float()
    stds = torch.tensor(config["data_specs"]["normalization"]["stds"]).view(1, 1, 6, 1, 1).to(device).float()

    img_files = list((data_dir / "images").glob("*.npy"))
    selected_files = np.random.choice(img_files, num_samples, replace=False)
    os.makedirs("inference_img", exist_ok=True)

    print(f"🚀 Generazione di {num_samples} pannelli a 4 colonne...")

    for idx, img_path in enumerate(selected_files):
        fname = img_path.stem
        mask_path = data_dir / "masks" / f"{fname}.tif"
        
        # Caricamento
        image_np = np.load(img_path).astype(np.float32)
        with rasterio.open(mask_path) as src:
            gt_mask_orig = src.read(1)
        
        # --- PREPARAZIONE BASE (Satellite RGB Estate) ---
        rgb_tensor = torch.from_numpy(image_np[2, [2, 1, 0], :, :]).unsqueeze(0)
        rgb_resized = F.interpolate(rgb_tensor, size=(224, 224), mode='bilinear').squeeze(0).permute(1, 2, 0).numpy()
        p2, p98 = np.percentile(rgb_resized, 2), np.percentile(rgb_resized, 98)
        base_img = np.clip((rgb_resized - p2) / (p98 - p2 + 1e-6), 0, 1)

        # --- PREPARAZIONE MASCHERE ---
        gt_224 = np.array(Image.fromarray(gt_mask_orig.astype(np.uint8)).resize((224, 224), resample=Image.NEAREST))
        
        img_t = torch.from_numpy(image_np).unsqueeze(0).float().to(device)
        B, T, C, H, W = img_t.shape
        img_t = F.interpolate(img_t.view(B*T, C, H, W), size=(224, 224), mode='bilinear').view(B, T, C, 224, 224)
        img_t = (img_t - means) / (stds + 1e-6)

        with torch.no_grad():
            logits = model(img_t)
            pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        
        # --- PLOTTING A 4 COLONNE ---
        fig, axes = plt.subplots(1, 4, figsize=(24, 8))
        
        # 1. Satellite Originale
        axes[0].imshow(base_img)
        axes[0].set_title(f"1. Satellite RGB\n({fname})", fontsize=14)
        
        # 2. Ground Truth Mask (Solida)
        axes[1].imshow(colorize_mask_rgb(gt_224))
        axes[1].set_title("2. Ground Truth Mask", fontsize=14)
        
        # 3. GT Overlay (Trasparente su satellite)
        axes[2].imshow(base_img)
        axes[2].imshow(colorize_mask_rgba(gt_224, opacity=0.5))
        axes[2].set_title("3. GT Overlay", fontsize=14)
        
        # 4. Prediction Overlay (Modello su satellite)
        axes[3].imshow(base_img)
        axes[3].imshow(colorize_mask_rgba(pred_mask, opacity=0.6))
        axes[3].set_title("4. Prithvi Prediction Overlay\n(mIoU 80%)", fontsize=14)

        for ax in axes: ax.axis('off')

        # Legenda in basso
        legend_patches = [mpatches.Patch(color=[c/255 for c in CLASS_COLORS[i]], label=CLASS_NAMES[i]) for i in range(1, 9)]
        fig.legend(handles=legend_patches, loc='lower center', ncol=4, fontsize=12, frameon=False)
        
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(f"inference_img/panel_{fname}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f" ✅ Pannello {idx+1}/{num_samples} salvato.")

if __name__ == "__main__":
    run_pro_inference(num_samples=20)