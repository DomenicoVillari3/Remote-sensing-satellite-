import numpy as np
from pathlib import Path
import tqdm
import rasterio
import numpy as np
from pathlib import Path
import tqdm
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

def analisi_distr():
    mask_path = Path("/home/almadigit/CropSemanticSegmentation/output1/SmartFood/crop_data/masks")
    class_counts = {i: 0 for i in range(9)}

    print("🧐 Analisi distribuzione classi...")
    for m_file in list(mask_path.glob("*.tif")):
        with rasterio.open(m_file) as src:
            mask = src.read(1)
            unique, counts = np.unique(mask, return_counts=True)
            for u, c in zip(unique, counts):
                if u in class_counts:
                    class_counts[u] += c

    total_pixels = sum(class_counts.values())
    print("\n📊 DISTRIBUZIONE PIXEL:")
    for cls, count in class_counts.items():
        perc = (count / total_pixels) * 100
        print(f"Classe {cls}: {perc:.2f}%")



def analisi_normalizzazione():
    img_path = Path("/home/almadigit/CropSemanticSegmentation/output1/SmartFood/crop_data/images")
    files = list(img_path.glob("*.npy"))

    mean_sum = np.zeros(6)
    std_sum = np.zeros(6)

    print("🧪 Calcolo Normalizzazione (6 bande)...")
    for f in files[:500]: # Basta analizzare 500 file per una buona stima
        data = np.load(f).astype(np.float32) # [T, C, H, W]
        # Calcoliamo la media lungo le dimensioni spaziali e temporali
        mean_sum += np.mean(data, axis=(0, 2, 3))
        std_sum += np.std(data, axis=(0, 2, 3))

    final_mean = mean_sum / 500
    final_std = std_sum / 500

    print(f"\n✅ Da mettere nel CONFIG JSON:")
    print(f"Means: {final_mean.tolist()}")
    print(f"Stds:  {final_std.tolist()}")
    # ---------------------------------------------------
    import json
from pathlib import Path
from sklearn.model_selection import train_test_split

def recover_lists():
    # 1. Carica il config per i percorsi
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    input_dir = Path(config["paths"]["input_dir"])
    img_dir = input_dir / "images"

    # 2. Recupera i file (esattamente come nel tuo script di training)
    # NOTA: se vuoi essere sicuro al 100% per il futuro, aggiungi .sort() 
    # ma per RECUPERARE quello vecchio usiamo la tua logica originale
    img_files = [f.stem for f in img_dir.glob("*.npy")]
    
    # 3. Riesegui lo split con lo stesso random_state
    print(f"🔄 Rieseguo lo split su {len(img_files)} file...")
    train_f, val_f = train_test_split(img_files, test_size=0.15, random_state=42)

    # 4. Salva le liste su file
    with open("train_files.txt", "w") as f:
        for item in train_f:
            f.write(f"{item}\n")
            
    with open("val_files.txt", "w") as f:
        for item in val_f:
            f.write(f"{item}\n")

    print(f"✅ Recupero completato!")
    print(f"📁 File creati: train_files.txt ({len(train_f)}) e val_files.txt ({len(val_f)})")


if __name__=="__main__":
    #analisi_distr()
    #analisi_normalizzazione()
    recover_lists()