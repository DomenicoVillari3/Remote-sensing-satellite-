# Smart Food — Segmentazione Agricola Satellitare (Sicilia & Malta)

**Progetto INTERREG Italia–Malta | ALMA DIGIT**

Classificazione automatica delle colture mediterranee tramite Geospatial Foundation Models (Prithvi-2.0) su dati Sentinel-2 multi-temporali.

---

## Panoramica

Il sistema analizza immagini Sentinel-2 (6 bande, 4 stagioni, 10m/pixel) e classifica ogni pixel del territorio in 9 classi di copertura del suolo: Sfondo, Olivo, Vite, Agrumi, Frutteto, Grano, Legumi, Ortaggi, Incolto.

Il backbone è **Prithvi-2.0** (NASA/IBM), un Vision Transformer pre-addestrato su petabyte di dati Landsat e Sentinel-2, fine-tuned su dati mediterranei spagnoli e applicato in transfer learning su Sicilia e Malta.

### Performance raggiunte

| Metrica | Valore |
|---------|--------|
| Accuracy globale | 92% |
| mIoU | 85.1% |
| F1-Score medio | 0.91 |
| Recall medio per classe | 93% |

---

## Struttura del Progetto

```
Smart-Food/
│
├── config.json                    # Configurazione globale (paths, iperparametri, normalizzazione)
├── classes_mapping.json           # Mappatura classi originali → 9 classi target
│
├── src/                           # Codice sorgente principale
│   ├── train.py                   # Training: dataset, modello, loss, loop di addestramento
│   ├── visual_inference.py        # Inferenza su validation set + pannelli 4-colonne
│   ├── confusion_matrix.py        # Matrice di confusione e classification report
│   ├── inference.py               # Inferenza su bbox arbitrari (STAC diretto)
│   └── calculate_stats.py         # Calcolo mean/std per normalizzazione dataset
│
├── data/
│   ├── worker_minio.py            # Download regionale Sicilia → MinIO (griglia 0.1°)
│   ├── convert_npy_to_images.py   # Conversione NPY → PNG preview su MinIO
│   └── dataset/                   # Split file
│       ├── train_files.txt
│       ├── val_files.txt
│       └── test_files.txt
│
├── demo/                          # Demo interattiva e testing
│   ├── demo_gui.py                # Interfaccia Gradio (punto + bbox)
│   ├── minio_inference.py         # Engine di inferenza (MinIO + STAC fallback)
│   ├── automated_testing.py       # Suite di 13 test automatici
│   ├── generate_interactive_map.py # Genera mappa Leaflet dai GeoTIFF
│   └── launcher.py                # Launcher CLI
│
├── models/
│   └── prithvi_4090_best.pth      # Pesi del modello (checkpoint migliore)
│
├── outputs/                       # Risultati inferenza (generati a runtime)
│   ├── tif/                       # GeoTIFF georeferenziati (EPSG:4326)
│   ├── reports/                   # Report PNG a 3 pannelli
│   └── interactive_map.html       # Mappa Leaflet (generata da generate_interactive_map.py)
│
├── docs/
│   └── doc_tecnico_remote_sensing.pdf
│
└── docker-compose.yml             # Infrastruttura: MinIO + Redpanda
```

---

## Quick Start

### 1. Requisiti

```bash
# Ambiente Python 3.12+ con CUDA
pip install torch torchvision terratorch rasterio stackstac pystac-client
pip install albumentations gradio matplotlib seaborn pillow boto3
```

### 2. Training (opzionale — pesi già forniti)

```bash
python src/train.py
```

Il training richiede una RTX 4090 (24GB VRAM) e il dataset in `config.json → paths.input_dir`. Lo split stratificato è salvato in `data/dataset/`.

### 3. Demo interattiva

```bash
cd demo
python demo_gui.py
# Apri http://localhost:7860
```

La demo supporta due modalità di input:
- **Punto**: inserisci lat/lon, il sistema crea un bbox di ~11km centrato sul punto
- **Bounding Box**: inserisci direttamente i 4 vertici del rettangolo geografico

Sono disponibili 19 punti predefiniti (13 Sicilia + 6 Malta).

### 4. Mappa interattiva

Dopo aver eseguito alcune inferenze:

```bash
python demo/generate_interactive_map.py
# Apri interactive_map.html nel browser
```

---

## Architettura del Modello

```
Input: (B, T=4, C=6, H=224, W=224)
  │
  ▼
Prithvi-2.0 ViT Backbone (pre-trained, frozen layers iniziali)
  │  → Masked Autoencoder pre-training su Landsat + Sentinel-2
  │  → Token sequence con posizional embedding temporale
  │
  ▼
Feature Tokens (B, N, D) → reshape → (B, D, 14, 14)
  │
  ▼
Decoder con 4 ResidualUpBlock:
  D → 256 → 128 → 64 → 32 → 9 classi
  │  → Ogni blocco: Upsample 2x + Conv 3x3 + BN + ReLU + Shortcut
  │
  ▼
Bilinear Interpolation → (B, 9, 224, 224)
  │
  ▼
argmax → Mappa di segmentazione
```

### Loss Function

Combinata **CrossEntropy pesata + Dice Loss**, con pesi inversi alla frequenza delle classi:

| Classe | Peso CE |
|--------|---------|
| Sfondo | 0.2 |
| Olivo | 1.2 |
| Vite | 2.5 |
| Agrumi | 2.3 |
| Frutteto | 1.8 |
| Grano | 0.6 |
| Legumi | 4.5 |
| **Ortaggi** | **12.0** |
| Incolto | 2.2 |

---

## Pipeline di Inferenza

```
Coordinate (lat, lon)
  │
  ▼
1. Ricerca cache MinIO (tile pre-scaricata?)
  │  ├── Sì → Carica NPY + smart_crop centrato
  │  └── No → Download on-the-fly da STAC (4 stagioni, <20% nuvole)
  │
  ▼
2. Cubo 4D: (4, 6, H, W) in float32
  │
  ▼
3. Padding reflect + Sliding Window (224×224, overlap opzionale)
  │
  ▼
4. Inferenza batch GPU (bf16) + ricomposizione mosaico
  │
  ▼
5. Post-processing:
  │  ├── Rimozione padding
  │  ├── Filtro acqua (NIR estate < 400 → sfondo)
  │  └── Unpad al formato originale
  │
  ▼
6. Output:
   ├── GeoTIFF (EPSG:4326, georeferenziato)
   ├── Report PNG (RGB + Segmentazione + Overlay)
   └── Statistiche (ettari per classe)
```

---

## Classi di Copertura

| ID | Classe | Colore | Recall |
|----|--------|--------|--------|
| 0 | Sfondo | Nero (trasparente) | 89% |
| 1 | Olivo | Verde neon | 95% |
| 2 | Vite | Magenta | 97% |
| 3 | Agrumi | Arancione | 97% |
| 4 | Frutteto | Blu elettrico | 87% |
| 5 | Grano | Giallo | 94% |
| 6 | Legumi | Cyan | 94% |
| 7 | Ortaggi | Rosso vivo | 94% |
| 8 | Incolto | Bianco | 92% |

---

## Configurazione (config.json)

```json
{
  "project_meta": {
    "backbone_model": "terratorch_prithvi_eo_v2_100_tl"
  },
  "data_specs": {
    "img_size": 224,
    "num_classes": 9,
    "num_channels": 6,
    "normalization": {
      "means": [724.81, 1070.41, 1344.19, 2834.34, 2902.63, 2228.62],
      "stds":  [393.22, 490.50, 682.72, 834.26, 937.53, 874.68]
    }
  },
  "training_params": {
    "batch_size": 32,
    "num_epochs": 150,
    "learning_rate": 1e-4,
    "early_stop_patience": 15
  }
}
```

---

## Testing Automatico

```bash
cd demo
python automated_testing.py    # ~10-15 min, 13 scenari
```

Genera un report HTML con success rate, tempi, confusion tra classi e grafici riassuntivi su 5 categorie: agricoltura intensiva, colture permanenti, cereali, zone costiere, stress test (mare/urbano).

---

## Infrastruttura

```bash
docker-compose up -d    # Avvia MinIO (porta 9000/9001) + Redpanda
python data/worker_minio.py  # Scarica griglia regionale Sicilia
```

MinIO funge da Data Lake locale con cache-first strategy: le tile già scaricate vengono riutilizzate per inferenze successive senza ri-download da STAC.

---

## Limitazioni Note

- **Frutteto vs Incolto**: confusione sporadica in estate (riflettanza SWIR/NIR simile su frutteti non irrigati)
- **Zone urbane**: non esiste una classe "urbano" — le città vengono classificate come mix di Incolto e Sfondo
- **Malta**: il modello opera in transfer learning implicito (non addestrato su tile maltesi), performance leggermente inferiori
- **Bordi immagine**: le predizioni ai margini dei chip possono mostrare artefatti — mitigato con padding reflect e overlap opzionale

---

## Autore

**Dott. Domenico Villari** — ALMA DIGIT  
Progetto INTERREG Italia–Malta, Marzo 2026