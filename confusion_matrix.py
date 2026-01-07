import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tqdm
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report

# Importiamo l'architettura dal tuo file train.py
from train import PrithviSegmentation4090

# ==========================================
# CONFIGURAZIONE
# ==========================================
CLASS_NAMES = ["Sfondo", "Olivo", "Vite", "Agrumi", "Frutteto", "Grano", "Legumi", "Ortaggi", "Incolto"]

def run_evaluation():
    # 1. Caricamento Config e Liste
    with open('config.json', 'r') as f: config = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_list_path = Path("val_files.txt")
    if not val_list_path.exists():
        print("❌ Errore: val_files.txt non trovato!")
        return
    
    with open(val_list_path, "r") as f:
        val_filenames = [line.strip() for line in f.readlines()]

    # 2. Inizializzazione Modello
    from terratorch.registry import BACKBONE_REGISTRY
    print(f"🔄 Caricamento modello per valutazione...")
    backbone = BACKBONE_REGISTRY.build(config["project_meta"]["backbone_model"], pretrained=False)
    model = PrithviSegmentation4090(backbone, config["data_specs"]["num_classes"])
    
    state_dict = torch.load(config["paths"]["model_save_path"], map_location=device)
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device).eval()

    means = torch.tensor(config["data_specs"]["normalization"]["means"]).view(1, 1, 6, 1, 1).to(device).float()
    stds = torch.tensor(config["data_specs"]["normalization"]["stds"]).view(1, 1, 6, 1, 1).to(device).float()

    all_preds = []
    all_targets = []

    # 3. Ciclo di Valutazione su tutto il set di Validation
    print(f"🧪 Analisi di {len(val_filenames)} file di validazione...")
    
    with torch.no_grad():
        for fname in tqdm.tqdm(val_filenames):
            img_path = Path(config["paths"]["input_dir"]) / "images" / f"{fname}.npy"
            mask_path = Path(config["paths"]["input_dir"]) / "masks" / f"{fname}.tif"
            
            # Caricamento e Preprocessing (stessa logica di visual_inference)
            import rasterio
            image_np = np.load(img_path).astype(np.float32)
            with rasterio.open(mask_path) as src:
                gt_mask = src.read(1).astype(np.uint8)
            
            # Resize GT a 224 per matchare la predizione
            from PIL import Image
            gt_224 = np.array(Image.fromarray(gt_mask).resize((224, 224), resample=Image.NEAREST))
            
            img_t = torch.from_numpy(image_np).unsqueeze(0).float().to(device)
            B, T, C, H, W = img_t.shape
            img_t = F.interpolate(img_t.view(B*T, C, H, W), size=(224, 224), mode='bilinear').view(B, T, C, 224, 224)
            img_t = (img_t - means) / (stds + 1e-6)

            # Predizione
            logits = model(img_t)
            preds = logits.argmax(dim=1).squeeze(0).cpu().numpy()

            # Accumulo dati (appiattiamo i pixel)
            all_preds.append(preds.flatten())
            all_targets.append(gt_224.flatten())

    # 4. Calcolo Metriche
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)

    print("\n📊 CALCOLO MATRICE DI CONFUSIONE...")
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    # Normalizzazione per riga (Recall)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)

    # 5. Visualizzazione con Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Matrice di Confusione Normalizzata (Validation Set)\nmIoU Finale: {config.get("miou_target", "81.8")}%', fontsize=15)
    plt.ylabel('Classe Reale (Ground Truth)')
    plt.xlabel('Classe Predetta dal Modello')
    
    plt.savefig("confusion_matrix_final.png", dpi=300, bbox_inches='tight')
    print("✅ Matrice salvata in: confusion_matrix_final.png")

    # 6. Report testuale dettagliato
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    print("✅ Report testuale salvato in: classification_report.txt")
    print("\n" + report)

if __name__ == "__main__":
    run_evaluation()