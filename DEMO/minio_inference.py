import os
import sys
import json
import torch
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import boto3
import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm import tqdm
import warnings
import pystac_client
import stackstac

# ==========================================
# 1. SETUP IMPORT (Gestione percorsi)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from train import PrithviSegmentation4090 
    from terratorch.registry import BACKBONE_REGISTRY
    from visual_inference import CLASS_COLORS, CLASS_NAMES, colorize_mask_rgb
except ImportError as e:
    print("ERRORE CRITICO: Impossibile importare i moduli del modello.")
    print(f"Assicurati che 'train.py' sia in: {parent_dir}")
    sys.exit(1)

warnings.filterwarnings('ignore')

# Costanti Geografiche
PIXEL_AREA_M2 = 100 
M2_TO_HECTARES = 10000

class SicilyInferencePoint:
    def __init__(self, config_path='../config.json'):
        # Gestione path config
        if not os.path.exists(config_path):
            config_path = 'config.json' # Fallback locale
            
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chip_size = 224 # Dimensione patch per il modello
        
        # Configurazione per download STAC
        self.assets = ["blue", "green", "red", "nir08", "swir16", "swir22"]
        
        # Connessione MinIO
        self.s3_client = boto3.client(
            's3',
            endpoint_url='http://localhost:9000',
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin'
        )
        self.bucket_name = "sicily-sentinel-data"
        
        # Output
        os.makedirs("inference_results/tif", exist_ok=True)
        os.makedirs("inference_results/reports", exist_ok=True)
        
        self._load_model()
        
    def _load_model(self):
        """Carica il modello Prithvi (pesi e architettura)"""
        model_path = self.config["paths"]["model_save_path"]
        if not os.path.exists(model_path):
            model_path = os.path.join(parent_dir, model_path)

        print(f"🔄 Caricamento modello: {model_path}")
        
        try:
            backbone = BACKBONE_REGISTRY.build(self.config["project_meta"]["backbone_model"], pretrained=False)
            self.model = PrithviSegmentation4090(backbone, self.config["data_specs"]["num_classes"])
            
            state_dict = torch.load(model_path, map_location=self.device)
            clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(clean_state_dict)
            self.model.to(self.device).eval()
            
            # Statistiche Normalizzazione (HLS 6 bande)
            self.means = torch.tensor(self.config["data_specs"]["normalization"]["means"]).view(1, 1, 6, 1, 1).to(self.device).float()
            self.stds = torch.tensor(self.config["data_specs"]["normalization"]["stds"]).view(1, 1, 6, 1, 1).to(self.device).float()
            print("✅ Modello caricato su GPU.")
        except Exception as e:
            print(f"❌ Errore modello: {e}")
            raise e

    def find_tile_containing_point(self, lat, lon):
        """
        Cerca su MinIO quale tile contiene le coordinate date.
        Ritorna (key, bbox) se trovato, altrimenti (None, None)
        """
        print(f"🔎 Ricerca Tile per Lat: {lat}, Lon: {lon} su MinIO...")
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix="raw_cubes/")
            
            for page in pages:
                if 'Contents' not in page: continue
                
                for obj in page['Contents']:
                    head = self.s3_client.head_object(Bucket=self.bucket_name, Key=obj['Key'])
                    meta = head['Metadata']
                    
                    if 'bbox' in meta:
                        try:
                            bbox = json.loads(meta['bbox'])
                            
                            if (bbox[0] <= lon <= bbox[2]) and (bbox[1] <= lat <= bbox[3]):
                                print(f"✅ Trovato Tile su MinIO: {obj['Key']}")
                                buffer = io.BytesIO()
                                self.s3_client.download_fileobj(self.bucket_name, obj['Key'], buffer)
                                buffer.seek(0)
                                
                                cube = np.load(buffer)
                                # ==========================================
                                # 🔥 CONTROLLO INTEGRITÀ (ANTI-NERO) 🔥
                                # ==========================================
                                non_zero = np.count_nonzero(cube)
                                total_px = cube.size
                                empty_ratio = 1.0 - (non_zero / total_px)
                                
                                if empty_ratio > 0.5: # Se più del 50% è nero/vuoto
                                    print(f"⚠️  ATTENZIONE: Il file in cache è vuoto al {empty_ratio:.1%}. Riscrico...")
                                    break
                                return obj['Key'], bbox
                            
                        except:
                            continue
        except Exception as e:
            print(f"⚠️  Errore durante la ricerca su MinIO: {e}")
        
        print("❌ Nessun tile trovato su MinIO per queste coordinate.")
        return None, None

    def download_tile_from_minio(self, key):
        """
        Scarica il file .npy da MinIO e converte correttamente a float32
        """
        print(f"⬇️  Download da MinIO: {key} ...")
        buffer = io.BytesIO()
        self.s3_client.download_fileobj(self.bucket_name, key, buffer)
        buffer.seek(0)
        
        # ✅ CORREZIONE: uint16 -> float32 (mantiene i valori originali)
        cube = np.load(buffer).astype(np.float32)
        print(f"   📦 Cubo caricato: {cube.shape}, Range: [{cube.min():.0f}, {cube.max():.0f}]")
        return cube  # Shape: (4, 6, H, W)

    def get_bbox_from_point(self, lat, lon, size=0.1):
        """
        Crea un BBox centrato sul punto di interesse.
        size=0.1 gradi ≈ 11km x 11km (adeguato per l'inferenza)
        """
        half = size / 2
        return [lon - half, lat - half, lon + half, lat + half]

    def download_cube_on_the_fly(self, bbox):
        """
        Scarica il cubo 4D direttamente da STAC (identico al downloader MinIO).
        Ritorna numpy array (4, 6, H, W) in float32.
        """
        print(f"🛰️  Download On-The-Fly da STAC per BBox: {bbox}")
        
        seasonal_periods = {
            'winter': "2023-01-01/2023-02-28",
            'spring': "2023-04-15/2023-05-30",
            'summer': "2023-07-01/2023-08-15",
            'autumn': "2023-10-01/2023-11-15"
        }
        ordered_seasons = ['winter', 'spring', 'summer', 'autumn']
        stac_items = []
        
        try:
            catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
            
            for season in ordered_seasons:
                print(f"   🔍 Ricerca stagione: {season}...")
                search = catalog.search(
                    collections=["sentinel-2-l2a"],
                    bbox=bbox,
                    datetime=seasonal_periods[season],
                    query={"eo:cloud_cover": {"lt": 20}}
                )
                items = search.item_collection()
                
                if not len(items):
                    print(f"   ⚠️  ATTENZIONE: Nessuna immagine trovata per {season}")
                    print(f"   → Questo può causare dati incompleti. Prova ad aumentare il BBox o la tolleranza cloud.")
                    return None
                
                # Prendi la migliore (meno nuvole)
                best_item = min(items, key=lambda x: x.properties['eo:cloud_cover'])
                cloud_cover = best_item.properties['eo:cloud_cover']
                print(f"   ✅ {season}: {cloud_cover:.1f}% cloud cover")
                stac_items.append(best_item)
                
            # Stack temporale (4, 6, H, W)
            print("   📥 Stacking delle immagini...")
            data = stackstac.stack(
                stac_items,
                assets=self.assets,
                bounds_latlon=bbox,
                resolution=10,  # 10m/pixel (identico al downloader)
                epsg=32633,     # UTM Zone 33N (Sicilia)
                fill_value=0,
                rescale=False
            )
            
            if data.sizes['time'] < 4:
                print(f"   ❌ Errore: Solo {data.sizes['time']} timesteps trovati (servono 4)")
                return None
            
            # Compute e converti a float32
            cube = data.astype("float32").compute().values
            print(f"   ✅ Download completato: {cube.shape}, Range: [{cube.min():.0f}, {cube.max():.0f}]")
            return cube
        
        except Exception as e:
            print(f"   ❌ Errore download STAC: {e}")
            return None

    def smart_crop(self, cube, bbox_tile, target_lat, target_lon, crop_size_px=800):
        """
        Ritaglia una finestra centrata sul punto di interesse.
        Se bbox_tile è None, usa l'intero cubo (caso download on-the-fly).
        """
        _, _, H, W = cube.shape
        
        if bbox_tile is None:
            # Caso download on-the-fly: bbox_tile è il BBox usato per il download
            # In questo caso, NON serve crop, usiamo tutto
            print(f"✂️  Nessun crop necessario (download on-the-fly): {cube.shape}")
            return cube, None
        
        # Caso MinIO: facciamo crop centrato sul punto
        min_lon, min_lat, max_lon, max_lat = bbox_tile
        
        # 1. Mappatura Coordinate -> Pixel
        x_pct = (target_lon - min_lon) / (max_lon - min_lon)
        y_pct = (max_lat - target_lat) / (max_lat - min_lat)
        
        px_x = int(x_pct * W)
        px_y = int(y_pct * H)
        
        # 2. Calcolo Finestra di Crop
        half = crop_size_px // 2
        
        x_start = max(0, px_x - half)
        x_end = min(W, x_start + crop_size_px)
        if x_end - x_start < crop_size_px: 
            x_start = max(0, x_end - crop_size_px)
            
        y_start = max(0, px_y - half)
        y_end = min(H, y_start + crop_size_px)
        if y_end - y_start < crop_size_px: 
            y_start = max(0, y_end - crop_size_px)
        
        # 3. Esegui il ritaglio
        cropped_cube = cube[:, :, y_start:y_end, x_start:x_end]
        
        # 4. Calcola il nuovo BBox Geografico
        deg_per_px_x = (max_lon - min_lon) / W
        deg_per_px_y = (max_lat - min_lat) / H 
        
        new_min_lon = min_lon + (x_start * deg_per_px_x)
        new_max_lon = min_lon + (x_end * deg_per_px_x)
        new_max_lat = max_lat - (y_start * deg_per_px_y)
        new_min_lat = max_lat - (y_end * deg_per_px_y)
        
        new_bbox = [new_min_lon, new_min_lat, new_max_lon, new_max_lat]
        
        print(f"✂️  Smart Crop: {cropped_cube.shape} attorno a ({target_lat}, {target_lon})")
        return cropped_cube, new_bbox

    def run_inference(self, lat, lon, region_name="Query_Result"):
        """
        Pipeline completa con fallback automatico:
        1. Cerca su MinIO
        2. Se non trova, scarica on-the-fly da STAC
        3. Esegue inferenza
        """
        
        # ==========================================
        # STRATEGIA 1: CERCA SU MINIO
        # ==========================================
        key, bbox_tile = self.find_tile_containing_point(lat, lon)
        
        if key is not None:
            # ✅ TROVATO SU MINIO
            print(f"📦 Utilizzo dati da MinIO...")
            cube_full = self.download_tile_from_minio(key)
            cube, bbox_crop = self.smart_crop(cube_full, bbox_tile, lat, lon, crop_size_px=800)
            source = "MinIO"
        else:
            # ❌ NON TROVATO SU MINIO → FALLBACK
            print(f"\n{'='*60}")
            print(f"🔄 FALLBACK: Download on-the-fly da STAC")
            print(f"{'='*60}\n")
            
            # Crea un BBox centrato sul punto (0.1° ≈ 11km)
            bbox_download = self.get_bbox_from_point(lat, lon, size=0.1)
            
            # Scarica da STAC
            cube = self.download_cube_on_the_fly(bbox_download)
            
            if cube is None:
                print("❌ ERRORE: Impossibile scaricare i dati. Verifica le coordinate o la copertura cloud.")
                return
            
            bbox_crop = bbox_download  # Usiamo il BBox di download come riferimento
            source = "STAC (On-The-Fly)"
        
        # ==========================================
        # PREPARAZIONE DATI PER INFERENZA
        # ==========================================
        T, C, h_orig, w_orig = cube.shape  # Dovrebbe essere (4, 6, H, W)
        
        print(f"\n📊 Cubo finale: {cube.shape} (T={T}, C={C}, H={h_orig}, W={w_orig})")
        print(f"📍 Sorgente: {source}")
        
        # Padding Reflect (per gestire bordi inferenza)
        pad_h = (self.chip_size - h_orig % self.chip_size) % self.chip_size
        pad_w = (self.chip_size - w_orig % self.chip_size) % self.chip_size
        
        # Padding su dimensioni spaziali (H, W)
        cube_padded = np.pad(cube, ((0,0), (0,0), (0, pad_h), (0, pad_w)), mode='reflect')
        h_pad, w_pad = cube_padded.shape[2], cube_padded.shape[3]
        
        prediction_map = np.zeros((h_pad, w_pad), dtype=np.uint8)

        # ==========================================
        # INFERENZA SLIDING WINDOW
        # ==========================================
        print(f"🧠 Elaborazione Neurale (con temporalità completa)...")
        with torch.no_grad():
            for y in tqdm(range(0, h_pad, self.chip_size), desc="Rows"):
                for x in range(0, w_pad, self.chip_size):
                    # Chip: estrae TUTTE le 4 stagioni
                    chip = cube_padded[:, :, y:y+self.chip_size, x:x+self.chip_size]  # (4, 6, 224, 224)
                    
                    # ✅ CORREZIONE: Mantieni la dimensione temporale
                    # Tensor (B=1, T=4, C=6, H=224, W=224)
                    t_chip = torch.from_numpy(chip).unsqueeze(0).to(self.device)
                    
                    # Normalizzazione con media/std del training
                    t_chip = (t_chip - self.means) / (self.stds + 1e-6)
                    
                    logits = self.model(t_chip)
                    pred = logits.argmax(dim=1).squeeze().cpu().numpy()
                    
                    prediction_map[y:y+self.chip_size, x:x+self.chip_size] = pred

        # ==========================================
        # POST-PROCESSING E OUTPUT
        # ==========================================
        # Unpad (Ritaglio finale)
        final_mask = prediction_map[:h_orig, :w_orig]
        # Filtra automaticamente il mare usando il canale NIR
        nir_summer = cube[2, 3, :h_orig, :w_orig]
        water_mask = nir_summer < 400  # Acqua ha NIR < 400
        final_mask[water_mask] = 0     # Azzera predizioni sul mare
        
        # Estrai RGB per visualizzazione (Estate = indice 2)
        final_rgb = cube[2, [2, 1, 0], :, :].transpose(1, 2, 0)  # (H, W, 3)

        # Output e Report
        print(f"\n💾 Salvataggio risultati per: {region_name}")
        print(f"   Sorgente dati: {source}")
        
        self.create_visual_report(final_rgb, final_mask, region_name, source)
        self.save_geotiff(final_mask, bbox_crop, region_name)
        self.print_stats(final_mask, region_name)
        
        # Opzionale: Salva su MinIO per future query (se era download on-the-fly)
        if source == "STAC (On-The-Fly)":
            self.save_to_minio_cache(cube, bbox_crop, lat, lon)

    def save_to_minio_cache(self, cube, bbox, lat, lon):
        """
        Salva il cubo scaricato on-the-fly su MinIO per future query.
        """
        try:
            print("\n💾 Salvataggio cache su MinIO per future query...")
            
            # Serializza (converte a uint16 per risparmiare spazio)
            buffer = io.BytesIO()
            np.save(buffer, cube.astype("uint16"))
            buffer.seek(0)

            # Metadati
            center_lat = (bbox[1] + bbox[3]) / 2
            center_lon = (bbox[0] + bbox[2]) / 2

            metadata = {
                'bbox': json.dumps(bbox),
                'center_lat': str(center_lat),
                'center_lon': str(center_lon),
                'task_id': 'onthefly',
                'shape': str(cube.shape)
            }

            # Naming
            lat_short = f"{center_lat:.4f}"
            lon_short = f"{center_lon:.4f}"
            object_name = f"raw_cubes/lat_{lat_short}_lon_{lon_short}_onthefly.npy"
            
            self.s3_client.upload_fileobj(
                buffer, 
                self.bucket_name, 
                object_name,
                ExtraArgs={'Metadata': metadata}
            )
            print(f"   ✅ Cache salvata: {object_name}")
            
        except Exception as e:
            print(f"   ⚠️  Impossibile salvare cache: {e}")

    def create_visual_report(self, rgb, mask, name, source="Unknown"):
        # Normalizzazione RGB 2-98%
        p2, p98 = np.percentile(rgb, (2, 98))
        rgb_norm = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
        
        mask_rgb = colorize_mask_rgb(mask)
        
        # Overlay
        h, w = mask.shape
        overlay = np.zeros((h, w, 4))
        overlay[..., :3] = mask_rgb / 255.0
        overlay[mask > 0, 3] = 0.4
        
        fig, axs = plt.subplots(1, 3, figsize=(24, 8))
        axs[0].imshow(rgb_norm); axs[0].set_title("Satellite RGB"); axs[0].axis('off')
        axs[1].imshow(mask_rgb); axs[1].set_title("Predizione"); axs[1].axis('off')
        axs[2].imshow(rgb_norm); axs[2].imshow(overlay); axs[2].set_title("Overlay"); axs[2].axis('off')
        
        # Legenda
        patches = [mpatches.Patch(color=np.array(c)/255, label=n) for i, (c, n) in enumerate(zip(CLASS_COLORS, CLASS_NAMES)) if i > 0]
        fig.legend(handles=patches, loc='lower center', ncol=len(patches)//2 + 1, bbox_to_anchor=(0.5, 0.02), fontsize=12)
        
        # Aggiungi info sorgente
        fig.suptitle(f"Analisi: {name} | Sorgente: {source}", fontsize=16, y=0.98)
        
        out_path = f"inference_results/reports/{name}_report.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  🖼️  Report: {out_path}")

    def save_geotiff(self, mask, bbox, name):
        h, w = mask.shape
        transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], w, h)
        
        out_path = f"inference_results/tif/{name}_map.tif"
        with rasterio.open(
            out_path, 'w', driver='GTiff', height=h, width=w, count=1, dtype='uint8',
            crs='EPSG:4326', transform=transform, nodata=0
        ) as dst:
            dst.write(mask, 1)
        print(f"  🌍 GeoTIFF: {out_path}")

    def print_stats(self, mask, name):
        unique, counts = np.unique(mask, return_counts=True)
        print(f"\n📊 Statistiche {name}:")
        for cls_id, count in zip(unique, counts):
            if cls_id == 0: continue
            ha = (count * PIXEL_AREA_M2) / M2_TO_HECTARES
            print(f"  - {CLASS_NAMES[cls_id]:<15}: {ha:>8.2f} ha")

# ==========================================
# ESEMPI DI UTILIZZO
# ==========================================
if __name__ == "__main__":
    '''
    pipeline = SicilyInferencePoint()
    
    print("\n" + "="*80)
    print("🌍 SISTEMA DI INFERENZA SICILIA CON FALLBACK AUTOMATICO")
    print("="*80)
    print("\nFunzionalità:")
    print("1️⃣  Cerca dati pre-scaricati su MinIO")
    print("2️⃣  Se non trova → Download automatico on-the-fly da STAC")
    print("3️⃣  Cache automatica per future query")
    print("="*80 + "\n")
    
    # ==========================================
    # TEST 1: Coordinate probabilmente su MinIO (se hai eseguito il downloader)
    # ==========================================
    print("\n🧪 TEST 1: Area probabilmente cached (Pachino)")
    target_lat_1 = 36.715
    target_lon_1 = 15.010
    
    pipeline.run_inference(
        lat=target_lat_1, 
        lon=target_lon_1, 
        region_name="Test_Pachino_Cache"
    )
    
    # ==========================================
    # TEST 2: Coordinate probabilmente NON su MinIO (fuori dalla griglia)
    # ==========================================
    print("\n" + "="*80)
    print("🧪 TEST 2: Area fuori cache (Palermo - Download on-the-fly)")
    print("="*80 + "\n")
    
    target_lat_2 = 38.115
    target_lon_2 = 13.361
    
    pipeline.run_inference(
        lat=target_lat_2, 
        lon=target_lon_2, 
        region_name="Test_Palermo_OnTheFly"
    )
    
    print("\n" + "="*80)
    print("✅ TESTING COMPLETATO")
    print("="*80)
    '''