import os
import json
import torch
import numpy as np
import rasterio
import pystac_client
import stackstac
import rioxarray
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from train import PrithviSegmentation4090 
from terratorch.registry import BACKBONE_REGISTRY
from visual_inference import CLASS_COLORS, CLASS_NAMES, colorize_mask_rgb
import time


# Costanti per il calcolo delle aree (Sentinel-2 = 10m x 10m per pixel)
PIXEL_AREA_M2 = 100 
M2_TO_HECTARES = 10000

class SicilyInferencePipeline:
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chip_size = 224
        self.assets = ["blue", "green", "red", "nir08", "swir16", "swir22"]
        
        # Creazione cartelle di output
        os.makedirs("tif", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        
        self._load_model()
        
    def _load_model(self):
        model_path = self.config["paths"]["model_save_path"]
        print(f"🔄 Caricamento modello Prithvi: {model_path}")
        backbone = BACKBONE_REGISTRY.build(self.config["project_meta"]["backbone_model"], pretrained=False)
        self.model = PrithviSegmentation4090(backbone, self.config["data_specs"]["num_classes"])
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state_dict.items()})
        self.model.to(self.device).eval()
        
        self.means = torch.tensor(self.config["data_specs"]["normalization"]["means"]).view(1, 1, 6, 1, 1).to(self.device).float()
        self.stds = torch.tensor(self.config["data_specs"]["normalization"]["stds"]).view(1, 1, 6, 1, 1).to(self.device).float()

    def download_area_cube(self, bbox):
        init=time.time()
        seasonal_periods = {
            'winter': "2023-01-01/2023-02-28",
            'spring': "2023-04-15/2023-05-30",
            'summer': "2023-07-01/2023-08-15",
            'autumn': "2023-10-01/2023-11-15"
        }
        ordered_seasons = ['winter', 'spring', 'summer', 'autumn']
        stac_items = []
        catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
        
        print(f"🛰️ Download Sentinel-2 (T=4) per BBox: {bbox}")
        for season in ordered_seasons:
            search = catalog.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=seasonal_periods[season], query={"eo:cloud_cover": {"lt": 20}})
            items = search.item_collection()
            if not items: raise ValueError(f"Mancano dati per la stagione: {season}")
            stac_items.append(min(items, key=lambda x: x.properties['eo:cloud_cover']))

        data = stackstac.stack(stac_items, assets=self.assets, bounds_latlon=bbox, resolution=10, epsg=32633, fill_value=0, rescale=False)
        return data.astype("float32").compute()

    def run_inference(self, bbox, output_name="output_sicilia.tif", output_img=True):
        cube = self.download_area_cube(bbox)
        T, C, H, W = cube.shape
        
        # Padding
        pad_h = (self.chip_size - H % self.chip_size) % self.chip_size
        pad_w = (self.chip_size - W % self.chip_size) % self.chip_size
        cube_padded = np.pad(cube, ((0,0), (0,0), (0, pad_h), (0, pad_w)), mode='constant')
        
        new_h, new_w = cube_padded.shape[2], cube_padded.shape[3]
        prediction_map = np.zeros((new_h, new_w), dtype=np.uint8)

        print(f"🧠 Segmentazione territoriale in corso...")
        for y in tqdm(range(0, new_h, self.chip_size)):
            for x in range(0, new_w, self.chip_size):
                chip = cube_padded[:, :, y:y+self.chip_size, x:x+self.chip_size]
                input_tensor = torch.from_numpy(chip).unsqueeze(0).to(self.device)
                input_tensor = (input_tensor - self.means) / (self.stds + 1e-6)
                
                with torch.no_grad():
                    logits = self.model(input_tensor)
                    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
                prediction_map[y:y+self.chip_size, x:x+self.chip_size] = pred

        final_map = prediction_map[:H, :W]
        
        # Salvataggio GeoTIFF
        output_da = cube.isel(time=0, band=0).copy()
        output_da.values = final_map
        output_da.rio.to_raster(f"tif/{output_name}")
        print(f"✅ GeoTIFF salvato: tif/{output_name}")

        # Statistiche di Area
        self.calculate_area_stats(final_map, output_name)

        if output_img:
            # Estrazione RGB Estivo (Indice 2)
            rgb_summer = cube.isel(time=2).isel(band=[2, 1, 0]).values.transpose(1, 2, 0)
            region_name = output_name.split(".")[0]
            create_3_panel_visualization(rgb_summer, final_map, f"reports/{region_name}_panel.png", region_name)

    def calculate_area_stats(self, mask, name):
        unique, counts = np.unique(mask, return_counts=True)
        stats = {}
        print(f"\n📊 STATISTICHE SUPERFICI ({name}):")
        for cls_id, count in zip(unique, counts):
            if cls_id == 0: continue
            hectares = (count * PIXEL_AREA_M2) / M2_TO_HECTARES
            stats[CLASS_NAMES[cls_id]] = hectares
            print(f"  - {CLASS_NAMES[cls_id].ljust(10)}: {hectares:>8.2f} ha")
        
        with open(f"reports/{name.split('.')[0]}_stats.json", "w") as f:
            json.dump(stats, f, indent=4)

def create_3_panel_visualization(rgb_image, prediction_mask, output_filename, region_name):
    # (Logica 3 pannelli come definita prima, con percentile stretching 2-98%)
    p2, p98 = np.percentile(rgb_image, (2, 98))
    rgb_norm = np.clip((rgb_image - p2) / (p98 - p2 + 1e-6), 0, 1)
    seg_rgb = colorize_mask_rgb(prediction_mask)
    
    h, w = prediction_mask.shape
    overlay_rgba = np.zeros((h, w, 4))
    overlay_rgba[..., :3] = seg_rgb / 255.0
    overlay_rgba[prediction_mask > 0, 3] = 0.5 

    fig, axs = plt.subplots(1, 3, figsize=(24, 8))
    axs[0].imshow(rgb_norm); axs[0].set_title("Satellite RGB"); axs[0].axis('off')
    axs[1].imshow(seg_rgb); axs[1].set_title("Predizione"); axs[1].axis('off')
    axs[2].imshow(rgb_norm); axs[2].imshow(overlay_rgba); axs[2].set_title("Overlay"); axs[2].axis('off')
    
    legend_patches = [mpatches.Patch(color=np.array(c)/255, label=n) for i, (c, n) in enumerate(zip(CLASS_COLORS, CLASS_NAMES)) if i > 0]
    fig.legend(handles=legend_patches, loc='lower center', ncol=len(legend_patches)//2 + 1, bbox_to_anchor=(0.5, 0.02))
    plt.savefig(output_filename, dpi=200, bbox_inches='tight')
    plt.close()

def get_bbox_from_point(lat, lon, size=0.05):
    half = size / 2
    # Ritorna [min_lon, min_lat, max_lon, max_lat]
    return [lon - half, lat - half, lon + half, lat + half]

if __name__ == "__main__":
    pipeline = SicilyInferencePipeline() # Carica il modello Prithvi
    
    # Esempio: Test sui Vigneti di Marsala
    ponto_marsala = [37.810, 12.510]
    bbox_test = get_bbox_from_point(ponto_marsala[0], ponto_marsala[1], size=0.05)
    
    print("🍷 Avvio test su zona vinicola nota...")
    pipeline.run_inference(
        bbox=bbox_test, 
        output_name="test_vigneti_marsala.tif", 
        output_img=True # Genera il report a 3 pannelli per il confronto visivo
    )
    
    # Esempio: Test sugli Agrumeti di Catania
    punto_catania = [37.380, 14.910]
    bbox_test_agrumi = get_bbox_from_point(punto_catania[0], punto_catania[1], size=0.05)
    
    print("🍊 Avvio test su agrumeti noti...")
    pipeline.run_inference(
        bbox=bbox_test_agrumi, 
        output_name="test_agrumeti_catania.tif", 
        output_img=True
    )