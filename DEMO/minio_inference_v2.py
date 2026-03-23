"""
Engine di inferenza Sicilia & Malta — con fix anti-crop e supporto bbox diretto.

Cambiamenti rispetto alla versione originale:
  1. get_bbox_from_point aggiunge un margine extra e lo trimma dopo l'inferenza
  2. Sliding window con overlap configurabile (elimina artefatti a griglia)
  3. Nuovo metodo run_inference_bbox per bbox espliciti
  4. smart_crop rivisto per non tagliare mai l'immagine in modo asimmetrico
"""

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
from PIL import Image

# ==========================================
# SETUP IMPORT
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from train import PrithviSegmentation4090
    from terratorch.registry import BACKBONE_REGISTRY
    from visual_inference import CLASS_COLORS, CLASS_NAMES, colorize_mask_rgb
except ImportError as e:
    print(f"ERRORE CRITICO: {e}")
    print(f"Assicurati che 'train.py' sia in: {parent_dir}")
    sys.exit(1)

warnings.filterwarnings("ignore")

PIXEL_AREA_M2 = 100
M2_TO_HECTARES = 10000


class SicilyInferencePoint:
    def __init__(self, config_path="../config.json"):
        if not os.path.exists(config_path):
            config_path = "config.json"
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chip_size = 224
        self.overlap = 32  # pixel di overlap tra chip adiacenti

        self.assets = ["blue", "green", "red", "nir08", "swir16", "swir22"]

        # MinIO
        self.s3_client = boto3.client(
            "s3",
            endpoint_url="http://localhost:9000",
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadmin",
        )
        self.bucket_name = "sicily-sentinel-data"

        os.makedirs("inference_results/tif", exist_ok=True)
        os.makedirs("inference_results/reports", exist_ok=True)

        self._load_model()

    # ------------------------------------------------------------------
    # MODELLO
    # ------------------------------------------------------------------
    def _load_model(self):
        model_path = self.config["paths"]["model_save_path"]
        if not os.path.exists(model_path):
            model_path = os.path.join(parent_dir, model_path)

        print(f"🔄 Caricamento modello: {model_path}")
        backbone = BACKBONE_REGISTRY.build(
            self.config["project_meta"]["backbone_model"], pretrained=False
        )
        self.model = PrithviSegmentation4090(
            backbone, self.config["data_specs"]["num_classes"]
        )
        state_dict = torch.load(model_path, map_location=self.device)
        clean = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(clean)
        self.model.to(self.device).eval()

        self.means = (
            torch.tensor(self.config["data_specs"]["normalization"]["means"])
            .view(1, 1, 6, 1, 1)
            .to(self.device)
            .float()
        )
        self.stds = (
            torch.tensor(self.config["data_specs"]["normalization"]["stds"])
            .view(1, 1, 6, 1, 1)
            .to(self.device)
            .float()
        )
        print("✅ Modello caricato su GPU.")

    # ------------------------------------------------------------------
    # BBOX HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def get_bbox_from_point(lat, lon, size=0.1, margin=0.015):
        """
        Crea bbox centrato con margine extra.
        Il margine viene rimosso dopo l'inferenza per evitare bordi tagliati.
        
        Args:
            size: dimensione dell'area utile in gradi (~11km per 0.1)
            margin: buffer extra per lato (~1.7km per 0.015)
        
        Returns:
            bbox [min_lon, min_lat, max_lon, max_lat], margin_deg
        """
        half = size / 2 + margin
        bbox = [lon - half, lat - half, lon + half, lat + half]
        return bbox, margin

    @staticmethod
    def _trim_margin(mask, cube_shape_hw, margin_deg, bbox_with_margin):
        """Rimuove il margine extra dalla maschera e dal cubo dopo l'inferenza."""
        h, w = mask.shape
        min_lon, min_lat, max_lon, max_lat = bbox_with_margin
        total_deg_lon = max_lon - min_lon
        total_deg_lat = max_lat - min_lat

        margin_px_x = int(round(margin_deg / total_deg_lon * w))
        margin_px_y = int(round(margin_deg / total_deg_lat * h))

        # Clamp per sicurezza
        margin_px_x = min(margin_px_x, w // 4)
        margin_px_y = min(margin_px_y, h // 4)

        trimmed = mask[margin_px_y : h - margin_px_y, margin_px_x : w - margin_px_x]

        # Bbox senza margine
        deg_per_px_x = total_deg_lon / w
        deg_per_px_y = total_deg_lat / h
        trimmed_bbox = [
            min_lon + margin_px_x * deg_per_px_x,
            min_lat + margin_px_y * deg_per_px_y,
            max_lon - margin_px_x * deg_per_px_x,
            max_lat - margin_px_y * deg_per_px_y,
        ]
        return trimmed, trimmed_bbox, (margin_px_y, margin_px_x)

    # ------------------------------------------------------------------
    # MINIO CACHE
    # ------------------------------------------------------------------
    def find_tile_containing_point(self, lat, lon):
        print(f"🔎 Ricerca tile MinIO per ({lat}, {lon})...")
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix="raw_cubes/")
            for page in pages:
                if "Contents" not in page:
                    continue
                for obj in page["Contents"]:
                    head = self.s3_client.head_object(
                        Bucket=self.bucket_name, Key=obj["Key"]
                    )
                    meta = head.get("Metadata", {})
                    if "bbox" not in meta:
                        continue
                    try:
                        bbox = json.loads(meta["bbox"])
                        if (bbox[0] <= lon <= bbox[2]) and (bbox[1] <= lat <= bbox[3]):
                            buf = io.BytesIO()
                            self.s3_client.download_fileobj(
                                self.bucket_name, obj["Key"], buf
                            )
                            buf.seek(0)
                            cube = np.load(buf)
                            empty_ratio = 1.0 - (np.count_nonzero(cube) / cube.size)
                            if empty_ratio > 0.5:
                                print(f"⚠️  Cache vuota al {empty_ratio:.0%}, skip.")
                                break
                            print(f"✅ Trovato: {obj['Key']}")
                            return obj["Key"], bbox
                    except Exception:
                        continue
        except Exception as e:
            print(f"⚠️  Errore MinIO: {e}")
        print("❌ Nessun tile trovato su MinIO.")
        return None, None

    def download_tile_from_minio(self, key):
        print(f"⬇️  Download MinIO: {key}")
        buf = io.BytesIO()
        self.s3_client.download_fileobj(self.bucket_name, key, buf)
        buf.seek(0)
        cube = np.load(buf).astype(np.float32)
        print(f"   Cubo: {cube.shape}, Range: [{cube.min():.0f}, {cube.max():.0f}]")
        return cube

    # ------------------------------------------------------------------
    # STAC DOWNLOAD
    # ------------------------------------------------------------------
    def download_cube_on_the_fly(self, bbox):
        print(f"🛰️  Download STAC per bbox: {bbox}")
        seasonal_periods = {
            "winter": "2023-01-01/2023-02-28",
            "spring": "2023-04-15/2023-05-30",
            "summer": "2023-07-01/2023-08-15",
            "autumn": "2023-10-01/2023-11-15",
        }
        ordered = ["winter", "spring", "summer", "autumn"]
        stac_items = []
        try:
            catalog = pystac_client.Client.open(
                "https://earth-search.aws.element84.com/v1"
            )
            for season in ordered:
                print(f"   🔍 {season}...", end=" ")
                search = catalog.search(
                    collections=["sentinel-2-l2a"],
                    bbox=bbox,
                    datetime=seasonal_periods[season],
                    query={"eo:cloud_cover": {"lt": 20}},
                )
                items = search.item_collection()
                if not len(items):
                    print("❌ Nessuna immagine!")
                    return None
                best = min(items, key=lambda x: x.properties["eo:cloud_cover"])
                print(f"✅ {best.properties['eo:cloud_cover']:.1f}% cloud")
                stac_items.append(best)

            print("   📥 Stacking...")
            data = stackstac.stack(
                stac_items,
                assets=self.assets,
                bounds_latlon=bbox,
                resolution=10,
                epsg=32633,
                fill_value=0,
                rescale=False,
            )
            if data.sizes["time"] < 4:
                print(f"   ❌ Solo {data.sizes['time']} timestep (servono 4)")
                return None
            cube = data.astype("float32").compute().values
            print(f"   ✅ {cube.shape}, Range: [{cube.min():.0f}, {cube.max():.0f}]")
            return cube
        except Exception as e:
            print(f"   ❌ Errore STAC: {e}")
            return None

    # ------------------------------------------------------------------
    # SMART CROP (rivisto — non taglia asimmetricamente)
    # ------------------------------------------------------------------
    def smart_crop(self, cube, bbox_tile, target_lat, target_lon, crop_size_px=800):
        _, _, H, W = cube.shape

        if bbox_tile is None:
            print(f"✂️  Nessun crop (download on-the-fly): {cube.shape}")
            return cube, None

        min_lon, min_lat, max_lon, max_lat = bbox_tile
        x_pct = (target_lon - min_lon) / (max_lon - min_lon)
        y_pct = (max_lat - target_lat) / (max_lat - min_lat)
        px_x = int(x_pct * W)
        px_y = int(y_pct * H)

        half = crop_size_px // 2

        # Centra la finestra, ma se non c'è spazio sufficiente espandi
        # dal lato opposto piuttosto che tagliare
        x_start = max(0, px_x - half)
        x_end = x_start + crop_size_px
        if x_end > W:
            x_end = W
            x_start = max(0, W - crop_size_px)

        y_start = max(0, px_y - half)
        y_end = y_start + crop_size_px
        if y_end > H:
            y_end = H
            y_start = max(0, H - crop_size_px)

        cropped = cube[:, :, y_start:y_end, x_start:x_end]

        deg_per_px_x = (max_lon - min_lon) / W
        deg_per_px_y = (max_lat - min_lat) / H
        new_bbox = [
            min_lon + x_start * deg_per_px_x,
            max_lat - y_end * deg_per_px_y,
            min_lon + x_end * deg_per_px_x,
            max_lat - y_start * deg_per_px_y,
        ]
        print(f"✂️  Crop: {cropped.shape} centrato su ({target_lat:.4f}, {target_lon:.4f})")
        return cropped, new_bbox

    # ------------------------------------------------------------------
    # INFERENZA CON OVERLAP
    # ------------------------------------------------------------------
    def _run_sliding_window(self, cube):
        """
        Sliding window con overlap configurabile.
        Tiene solo la regione centrale di ogni chip → elimina artefatti ai bordi.
        """
        T, C, h_orig, w_orig = cube.shape
        cs = self.chip_size
        ov = self.overlap
        step = cs - ov

        pad_h = (cs - h_orig % step) % step + ov
        pad_w = (cs - w_orig % step) % step + ov
        cube_padded = np.pad(
            cube, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="reflect"
        )
        h_pad, w_pad = cube_padded.shape[2], cube_padded.shape[3]
        prediction = np.zeros((h_pad, w_pad), dtype=np.uint8)

        half_ov = ov // 2

        print(f"🧠 Inferenza sliding window (chip={cs}, overlap={ov}, step={step})")
        with torch.no_grad():
            for y in tqdm(range(0, h_pad - cs + 1, step), desc="Righe"):
                for x in range(0, w_pad - cs + 1, step):
                    chip = cube_padded[:, :, y : y + cs, x : x + cs]
                    t_chip = torch.from_numpy(chip).unsqueeze(0).to(self.device)
                    t_chip = (t_chip - self.means) / (self.stds + 1e-6)
                    logits = self.model(t_chip)
                    pred = logits.argmax(dim=1).squeeze().cpu().numpy()

                    # Scrivi solo la regione centrale (scarta bordi inaffidabili)
                    wy_s = y + half_ov if y > 0 else 0
                    wx_s = x + half_ov if x > 0 else 0
                    wy_e = min(y + cs - half_ov, h_pad)
                    wx_e = min(x + cs - half_ov, w_pad)

                    cy = wy_s - y
                    cx = wx_s - x

                    prediction[wy_s:wy_e, wx_s:wx_e] = pred[
                        cy : cy + (wy_e - wy_s), cx : cx + (wx_e - wx_s)
                    ]

        return prediction[:h_orig, :w_orig]

    # ------------------------------------------------------------------
    # ENTRY POINT: PUNTO
    # ------------------------------------------------------------------
    def _tile_covers_request(self, bbox_tile, lat, lon, bbox_size):
        """
        Verifica se la tile MinIO copre completamente l'area richiesta.
        Ritorna True solo se il bbox richiesto è interamente contenuto nella tile.
        """
        req_bbox, _ = self.get_bbox_from_point(lat, lon, size=bbox_size, margin=0)
        # req_bbox = [min_lon, min_lat, max_lon, max_lat]
        covers = (
            bbox_tile[0] <= req_bbox[0] and   # tile min_lon <= richiesta min_lon
            bbox_tile[1] <= req_bbox[1] and   # tile min_lat <= richiesta min_lat
            bbox_tile[2] >= req_bbox[2] and   # tile max_lon >= richiesta max_lon
            bbox_tile[3] >= req_bbox[3]        # tile max_lat >= richiesta max_lat
        )
        return covers

    def run_inference(self, lat, lon, region_name="Query_Result", bbox_size=0.1):
        """
        Pipeline completa per query da punto.
        1. Cerca cache MinIO
        2. Verifica che la tile copra l'area richiesta (bbox_size)
        3. Se non copre → fallback STAC con l'area corretta
        4. Aggiunge margine, esegue inferenza, trimma margine
        """
        use_minio = False
        key, bbox_tile = self.find_tile_containing_point(lat, lon)

        if key is not None:
            # Controlla se la tile MinIO è abbastanza grande
            if self._tile_covers_request(bbox_tile, lat, lon, bbox_size):
                cube_full = self.download_tile_from_minio(key)
                cube, bbox_crop = self.smart_crop(
                    cube_full, bbox_tile, lat, lon,
                    crop_size_px=int(bbox_size / 0.1 * 800)  # scala crop_size con bbox_size
                )
                margin_deg = 0
                source = "MinIO"
                use_minio = True
            else:
                tile_size_lon = bbox_tile[2] - bbox_tile[0]
                tile_size_lat = bbox_tile[3] - bbox_tile[1]
                print(
                    f"⚠️  Tile MinIO troppo piccola: "
                    f"{tile_size_lon:.3f}° × {tile_size_lat:.3f}° "
                    f"ma richiesti {bbox_size:.3f}°. Scarico da STAC."
                )

        if not use_minio:
            print(f"\n{'='*60}\n🔄 Download da STAC (area: {bbox_size}°)\n{'='*60}")
            bbox_dl, margin_deg = self.get_bbox_from_point(lat, lon, size=bbox_size)
            cube = self.download_cube_on_the_fly(bbox_dl)
            if cube is None:
                print("❌ Download fallito.")
                return
            bbox_crop = bbox_dl
            source = "STAC (On-The-Fly)"

        self._process_and_output(cube, bbox_crop, margin_deg, region_name, source)

    # ------------------------------------------------------------------
    # ENTRY POINT: BBOX DIRETTO
    # ------------------------------------------------------------------
    def _tile_covers_bbox(self, bbox_tile, requested_bbox):
        """Verifica se la tile MinIO copre completamente il bbox richiesto."""
        return (
            bbox_tile[0] <= requested_bbox[0] and
            bbox_tile[1] <= requested_bbox[1] and
            bbox_tile[2] >= requested_bbox[2] and
            bbox_tile[3] >= requested_bbox[3]
        )

    def run_inference_bbox(self, bbox, region_name="BBox_Query", margin=0.015):
        """
        Inferenza su un bounding box definito dall'utente.
        bbox: [min_lon, min_lat, max_lon, max_lat]
        
        Cerca prima su MinIO se una tile copre il bbox richiesto.
        Se no, scarica da STAC.
        """
        bbox_with_margin = [
            bbox[0] - margin,
            bbox[1] - margin,
            bbox[2] + margin,
            bbox[3] + margin,
        ]

        # Prova MinIO: cerca tile che copre il centro del bbox
        center_lat = (bbox[1] + bbox[3]) / 2
        center_lon = (bbox[0] + bbox[2]) / 2
        key, bbox_tile = self.find_tile_containing_point(center_lat, center_lon)

        if key is not None and self._tile_covers_bbox(bbox_tile, bbox_with_margin):
            print("✅ Tile MinIO copre il bbox richiesto.")
            cube_full = self.download_tile_from_minio(key)
            # Crop al bbox con margine
            cube, bbox_crop = self.smart_crop(
                cube_full, bbox_tile, center_lat, center_lon,
                crop_size_px=max(
                    int((bbox_with_margin[3] - bbox_with_margin[1]) / (bbox_tile[3] - bbox_tile[1]) * cube_full.shape[2]),
                    int((bbox_with_margin[2] - bbox_with_margin[0]) / (bbox_tile[2] - bbox_tile[0]) * cube_full.shape[3]),
                )
            )
            self._process_and_output(cube, bbox_crop or bbox_with_margin, margin, region_name, "MinIO (BBox)")
        else:
            if key is not None:
                print("⚠️  Tile MinIO non copre il bbox richiesto. Scarico da STAC.")
            cube = self.download_cube_on_the_fly(bbox_with_margin)
            if cube is None:
                print("❌ Download fallito.")
                return
            self._process_and_output(cube, bbox_with_margin, margin, region_name, "STAC (BBox)")

    # ------------------------------------------------------------------
    # CORE: PROCESS + OUTPUT
    # ------------------------------------------------------------------
    def _process_and_output(self, cube, bbox_with_margin, margin_deg, region_name, source):
        T, C, h_orig, w_orig = cube.shape
        print(f"\n📊 Cubo: {cube.shape} | Sorgente: {source}")

        # Inferenza
        mask_full = self._run_sliding_window(cube)

        # Filtro acqua (NIR estate < 400)
        nir_summer = cube[2, 3, :h_orig, :w_orig]
        mask_full[nir_summer < 400] = 0

        # Trim margine se presente
        if margin_deg > 0:
            mask_trimmed, bbox_final, (my, mx) = self._trim_margin(
                mask_full, (h_orig, w_orig), margin_deg, bbox_with_margin
            )
            rgb_full = cube[2, [2, 1, 0], :, :].transpose(1, 2, 0)
            rgb_trimmed = rgb_full[my : h_orig - my, mx : w_orig - mx]
        else:
            mask_trimmed = mask_full
            bbox_final = bbox_with_margin
            rgb_trimmed = cube[2, [2, 1, 0], :, :].transpose(1, 2, 0)

        # Output
        print(f"\n💾 Output: {region_name} | Maschera: {mask_trimmed.shape}")
        self.create_visual_report(rgb_trimmed, mask_trimmed, region_name, source)
        self.save_geotiff(mask_trimmed, bbox_final, region_name)
        self.print_stats(mask_trimmed, region_name)

        # Cache su MinIO se download on-the-fly
        if "STAC" in source:
            self.save_to_minio_cache(cube, bbox_with_margin, 0, 0)

    # ------------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------------
    def create_visual_report(self, rgb, mask, name, source="Unknown"):
        p2, p98 = np.percentile(rgb, (2, 98))
        rgb_norm = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
        mask_rgb = colorize_mask_rgb(mask)

        h, w = mask.shape
        overlay = np.zeros((h, w, 4))
        overlay[..., :3] = mask_rgb / 255.0
        overlay[mask > 0, 3] = 0.4

        fig, axs = plt.subplots(1, 3, figsize=(24, 8))
        axs[0].imshow(rgb_norm); axs[0].set_title("Satellite RGB"); axs[0].axis("off")
        axs[1].imshow(mask_rgb); axs[1].set_title("Predizione"); axs[1].axis("off")
        axs[2].imshow(rgb_norm); axs[2].imshow(overlay); axs[2].set_title("Overlay"); axs[2].axis("off")

        patches = [
            mpatches.Patch(color=np.array(c) / 255, label=n)
            for i, (c, n) in enumerate(zip(CLASS_COLORS, CLASS_NAMES))
            if i > 0
        ]
        fig.legend(handles=patches, loc="lower center", ncol=len(patches) // 2 + 1,
                   bbox_to_anchor=(0.5, 0.02), fontsize=12)
        fig.suptitle(f"Analisi: {name} | Sorgente: {source}", fontsize=16, y=0.98)

        out = f"inference_results/reports/{name}_report.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  🖼️  Report: {out}")

    def save_geotiff(self, mask, bbox, name):
        h, w = mask.shape
        transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], w, h)
        out = f"inference_results/tif/{name}_map.tif"
        with rasterio.open(
            out, "w", driver="GTiff", height=h, width=w, count=1,
            dtype="uint8", crs="EPSG:4326", transform=transform, nodata=0,
        ) as dst:
            dst.write(mask, 1)
        print(f"  🌍 GeoTIFF: {out}")

    def print_stats(self, mask, name):
        unique, counts = np.unique(mask, return_counts=True)
        print(f"\n📊 Statistiche {name}:")
        for cls_id, count in zip(unique, counts):
            if cls_id == 0:
                continue
            ha = (count * PIXEL_AREA_M2) / M2_TO_HECTARES
            print(f"  - {CLASS_NAMES[cls_id]:<15}: {ha:>8.2f} ha")

    def save_to_minio_cache(self, cube, bbox, lat, lon):
        try:
            center_lat = (bbox[1] + bbox[3]) / 2
            center_lon = (bbox[0] + bbox[2]) / 2
            metadata = {
                "bbox": json.dumps(bbox),
                "center_lat": str(center_lat),
                "center_lon": str(center_lon),
                "task_id": "onthefly",
                "shape": str(cube.shape),
            }
            key = f"raw_cubes/lat_{center_lat:.4f}_lon_{center_lon:.4f}_onthefly.npy"
            buf = io.BytesIO()
            np.save(buf, cube.astype("uint16"))
            buf.seek(0)
            self.s3_client.upload_fileobj(buf, self.bucket_name, key,
                                          ExtraArgs={"Metadata": metadata})
            print(f"   💾 Cache: {key}")
        except Exception as e:
            print(f"   ⚠️  Cache non salvata: {e}")

    def create_rgb_image_from_cube(self, img_cube):
        summer = img_cube[2]
        rgb = np.stack([summer[2], summer[1], summer[0]], axis=-1)
        p2, p98 = np.percentile(rgb, (2, 98))
        stretched = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
        return Image.fromarray((stretched * 255).astype(np.uint8), mode="RGB")
