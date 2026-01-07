import numpy as np
from pathlib import Path
import tqdm
import rasterio
import numpy as np
from pathlib import Path
import tqdm


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

if __name__=="__main__":
    analisi_distr()
    analisi_normalizzazione()