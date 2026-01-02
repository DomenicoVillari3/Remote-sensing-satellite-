import os
import numpy as np
from tqdm import tqdm

# ==========================================
# CONFIGURAZIONE PERCORSI
# ==========================================
IMAGES_DIR = "/export/mimmo/output/images"
EXT = ".npy"

def calculate_temporal_stats():
    print(f"--- Calcolo Statistiche Dataset (NPY 4D) in: {IMAGES_DIR} ---")

    # 1. Trova tutti i file .npy
    files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(EXT)]
    if len(files) == 0:
        print(f"Errore: Nessun file {EXT} trovato!")
        return

    print(f"Trovati {len(files)} chip multi-temporali. Inizio calcolo...")

    # Inizializzazione accumulatori
    channel_sum = None
    channel_sq_sum = None
    total_obs_per_channel = 0  # Pixel totali * Numero di step temporali
    num_channels = 0

    # 2. Itera sulle immagini
    for file_name in tqdm(files):
        path = os.path.join(IMAGES_DIR, file_name)
        
        try:
            # Carica l'array numpy: (T=4, C=6, H=256, W=256)
            img = np.load(path).astype(np.float64)
            
            # Inizializza alla prima iterazione
            if channel_sum is None:
                T, num_channels, H, W = img.shape
                channel_sum = np.zeros(num_channels, dtype=np.float64)
                channel_sq_sum = np.zeros(num_channels, dtype=np.float64)
                # Osservazioni per canale in una singola immagine: T * H * W
                pixels_in_image = T * H * W
            
            # Somma su assi 0 (Time), 2 (Height), 3 (Width)
            # Risultato: un valore per ogni canale (C)
            channel_sum += np.sum(img, axis=(0, 2, 3))
            channel_sq_sum += np.sum(img ** 2, axis=(0, 2, 3))
            
            total_obs_per_channel += pixels_in_image

        except Exception as e:
            print(f"Errore leggendo {file_name}: {e}")
            continue

    if total_obs_per_channel == 0:
        print("Errore: Nessun dato analizzato.")
        return

    # 3. Calcolo Finale
    means = channel_sum / total_obs_per_channel
    # Varianza = E[x^2] - (E[x])^2
    stds = np.sqrt((channel_sq_sum / total_obs_per_channel) - (means ** 2))

    # 4. Output per CONFIG
    print("\n" + "="*50)
    print("RISULTATI DA COPIARE NEL CONFIG (6 BANDE):")
    print("="*50)

    mean_str = ", ".join([f"{m:.4f}" for m in means])
    std_str = ", ".join([f"{s:.4f}" for s in stds])

    print(f'"means": [{mean_str}],')
    print(f'"stds":  [{std_str}]')
    
    print("\nNote:")
    print(f"- Bande rilevate: {num_channels} (Standard Prithvi: 6)")
    print(f"- Step temporali analizzati per chip: {T}")
    print(f"- Totale osservazioni per canale: {total_obs_per_channel}")
    print("="*50 + "\n")

if __name__ == "__main__":
    calculate_temporal_stats()