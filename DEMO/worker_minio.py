import logging
import stackstac
import pystac_client
import numpy as np
import os
import warnings
import boto3
import json
import io
from PIL import Image

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("download_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SicilyRegionalDownloader:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            endpoint_url='http://localhost:9000',
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin'
        )
        self.bucket_name = "sicily-sentinel-data"

        try:
            self.s3_client.create_bucket(Bucket=self.bucket_name)
        except:
            pass 

        self.assets = ["blue", "green", "red", "nir08", "swir16", "swir22"]
        self.total_cubes_saved = 0
        self.total_images_saved = 0

    def get_sub_bboxes(self, large_bbox, step=0.1):
        min_lon, min_lat, max_lon, max_lat = large_bbox
        bboxes = []
        
        lon_ranges = np.arange(min_lon, max_lon, step)
        lat_ranges = np.arange(min_lat, max_lat, step)
        
        for lon in lon_ranges:
            for lat in lat_ranges:
                sub_bbox = [
                    round(lon, 4), 
                    round(lat, 4), 
                    round(min(lon + step, max_lon), 4), 
                    round(min(lat + step, max_lat), 4)
                ]
                bboxes.append(sub_bbox)
        
        return bboxes

    def download_area_cube(self, bbox):
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
                search = catalog.search(
                    collections=["sentinel-2-l2a"],
                    bbox=bbox,
                    datetime=seasonal_periods[season],
                    query={"eo:cloud_cover": {"lt": 20}}
                )
                items = search.item_collection()
                
                if not len(items):
                    return None
                
                best_item = min(items, key=lambda x: x.properties['eo:cloud_cover'])
                stac_items.append(best_item)
                
            data = stackstac.stack(
                stac_items,
                assets=self.assets,
                bounds_latlon=bbox,
                resolution=10, 
                epsg=32633, 
                fill_value=0,
                rescale=False
            )
            
            if data.sizes['time'] < 4:
                return None
            
            return data.astype("uint16").compute().values
        
        except Exception as e:
            logger.error(f"  ❌ Errore download STAC: {e}")
            return None

    def is_mostly_water_or_empty(self, cube):
        """
        ✅ VERSIONE MIGLIORATA: Filtro più robusto per identificare mare/acqua.
        Ritorna True se deve essere SCARTATO.
        """
        # 1. Controllo Vuoto
        non_zero_pixels = np.count_nonzero(cube)
        total_pixels = cube.size
        
        if non_zero_pixels < (total_pixels * 0.05):
            logger.info(f"  🌊 Scartato: {(1 - non_zero_pixels/total_pixels)*100:.1f}% pixel vuoti")
            return True
            
        # 2. Controllo Mare con METRICHE MULTIPLE
        nir_summer = cube[2, 3, :, :]  # NIR estate
        
        # Metrica 1: Media NIR (escludendo no-data)
        valid_nir = nir_summer[nir_summer > 0]
        if len(valid_nir) == 0:
            return True
        nir_mean = np.mean(valid_nir)
        
        # Metrica 2: Percentile 95
        nir_p95 = np.percentile(valid_nir, 95)
        
        # Metrica 3: Percentuale pixel con NIR < 400 (acqua)
        water_pixels = np.sum(nir_summer < 400)
        water_ratio = water_pixels / nir_summer.size
        
        # DECISIONE: È mare se soddisfa ALMENO 2 condizioni
        is_water_by_mean = nir_mean < 500
        is_water_by_p95 = nir_p95 < 800
        is_water_by_ratio = water_ratio > 0.7
        
        water_indicators = sum([is_water_by_mean, is_water_by_p95, is_water_by_ratio])
        
        if water_indicators >= 2:
            logger.info(f"  🌊 Scartato (Mare): NIR_mean={nir_mean:.0f}, NIR_p95={nir_p95:.0f}, water_ratio={water_ratio:.2f}")
            return True
            
        # 3. Check RGB (acqua è scura anche in RGB)
        rgb_summer = cube[2, [0, 1, 2], :, :]
        valid_rgb = rgb_summer[rgb_summer > 0]
        rgb_mean = np.mean(valid_rgb) if len(valid_rgb) > 0 else 0
        
        if rgb_mean < 300 and nir_mean < 600:
            logger.info(f"  🌊 Scartato (Mare scuro): RGB_mean={rgb_mean:.0f}, NIR_mean={nir_mean:.0f}")
            return True
            
        return False

    def percentile_stretch(self, img, p_low=2, p_high=98):
        """
        🆕 Normalizzazione percentile per visualizzazione ottimale
        """
        p_low_val, p_high_val = np.percentile(img, (p_low, p_high))
        img_stretched = np.clip((img - p_low_val) / (p_high_val - p_low_val + 1e-6), 0, 1)
        return (img_stretched * 255).astype(np.uint8)

    def create_rgb_image(self, img_cube):
        """
        🆕 Crea immagine RGB visualizzabile dal cubo 4D
        
        Input: (4, 6, H, W) [Time, Bands, Height, Width]
        Output: PIL.Image RGB (stagione estiva)
        """
        # Estrai stagione estiva (indice 2)
        summer_cube = img_cube[2]  # Shape: (6, H, W)
        
        # Estrai bande RGB (Blue=0, Green=1, Red=2)
        blue = summer_cube[0]
        green = summer_cube[1]
        red = summer_cube[2]
        
        # Stack come RGB
        rgb = np.stack([red, green, blue], axis=-1)  # (H, W, 3)
        
        # Normalizzazione per visualizzazione
        rgb_normalized = self.percentile_stretch(rgb)
        
        # Converti in PIL Image
        return Image.fromarray(rgb_normalized, mode='RGB')

    def upload_cube_to_minio(self, img_cube, bbox, task_id):
        """
        ✅ MODIFICATO: Carica sia NPY che PNG su MinIO
        """
        try:
            center_lat = (bbox[1] + bbox[3]) / 2
            center_lon = (bbox[0] + bbox[2]) / 2

            # Metadati comuni
            metadata = {
                'bbox': json.dumps(bbox),
                'center_lat': str(center_lat),
                'center_lon': str(center_lon),
                'task_id': str(task_id),
                'shape': str(img_cube.shape)
            }

            lat_short = f"{center_lat:.4f}"
            lon_short = f"{center_lon:.4f}"
            
            # ===== 1. SALVA NPY (Dati grezzi 4D) =====
            npy_buffer = io.BytesIO()
            np.save(npy_buffer, img_cube.astype("uint16"))
            npy_buffer.seek(0)
            
            npy_key = f"raw_cubes/lat_{lat_short}_lon_{lon_short}_task{task_id}.npy"
            
            self.s3_client.upload_fileobj(
                npy_buffer, 
                self.bucket_name, 
                npy_key,
                ExtraArgs={'Metadata': metadata}
            )
            logger.info(f"  📦 NPY salvato: {npy_key}")
            self.total_cubes_saved += 1
            
            # ===== 2. SALVA PNG (Preview RGB) =====
            try:
                rgb_image = self.create_rgb_image(img_cube)
                
                png_buffer = io.BytesIO()
                rgb_image.save(png_buffer, format='PNG', optimize=True)
                png_buffer.seek(0)
                
                png_key = f"rgb_images/lat_{lat_short}_lon_{lon_short}_task{task_id}.png"
                
                # Aggiungi info immagine ai metadata
                png_metadata = metadata.copy()
                png_metadata['image_type'] = 'rgb_summer'
                png_metadata['image_size'] = f"{rgb_image.width}x{rgb_image.height}"
                png_metadata['source_npy'] = npy_key
                
                self.s3_client.upload_fileobj(
                    png_buffer,
                    self.bucket_name,
                    png_key,
                    ExtraArgs={
                        'Metadata': png_metadata,
                        'ContentType': 'image/png'
                    }
                )
                logger.info(f"  🖼️  PNG salvato: {png_key} ({rgb_image.width}×{rgb_image.height} px)")
                self.total_images_saved += 1
                
            except Exception as e:
                logger.warning(f"  ⚠️  PNG non creato (errore: {e}), ma NPY salvato correttamente")
            
        except Exception as e:
            logger.error(f"  ❌ Errore Upload MinIO: {e}")

    def run(self):
        logger.info("🚀 Avvio Download Regionale Sicilia -> MinIO (NPY + PNG)")
        
        sicily_bbox = (12.40, 36.60, 15.70, 38.30)
        tiles = self.get_sub_bboxes(sicily_bbox, step=0.1)
        
        logger.info(f"🗺️ Griglia generata: {len(tiles)} tiles totali.")
        
        for i, bbox in enumerate(tiles):
            logger.info(f"📍 Processing Tile {i+1}/{len(tiles)}: {bbox}")
            
            cube = self.download_area_cube(bbox)
            
            if cube is None:
                logger.warning(f"  ⚠️ Download fallito o dati mancanti. Skip.")
                continue
                
            if self.is_mostly_water_or_empty(cube):
                continue
            
            self.upload_cube_to_minio(cube, bbox, task_id=i)

        logger.info("\n" + "="*70)
        logger.info("🏁 Download Completato!")
        logger.info("="*70)
        logger.info(f"📦 Cubi NPY salvati: {self.total_cubes_saved}")
        logger.info(f"🖼️  Immagini PNG salvate: {self.total_images_saved}")
        logger.info(f"📁 Bucket: {self.bucket_name}")
        logger.info(f"   ├── raw_cubes/  ({self.total_cubes_saved} .npy files)")
        logger.info(f"   └── rgb_images/ ({self.total_images_saved} .png files)")
        logger.info("="*70 + "\n")

if __name__ == "__main__":
    downloader = SicilyRegionalDownloader()
    downloader.run()