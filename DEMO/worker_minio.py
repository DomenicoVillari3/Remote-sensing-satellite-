import logging
import stackstac
import pystac_client
import numpy as np
import os
import time
import random
import warnings
import boto3
import json
import io

# Ignora warning geospaziali
warnings.filterwarnings('ignore')

# Configurazione Logging
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
        # Configurazione MinIO
        self.s3_client = boto3.client(
            's3',
            endpoint_url='http://localhost:9000',
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin'
        )
        self.bucket_name = "sicily-sentinel-data"

        # Crea bucket se non esiste
        try:
            self.s3_client.create_bucket(Bucket=self.bucket_name)
        except:
            pass 

        # Bande richieste da Prithvi
        self.assets = ["blue", "green", "red", "nir08", "swir16", "swir22"]
        self.total_cubes_saved = 0

    def get_sub_bboxes(self, large_bbox, step=0.1):
        """
        Divide la Sicilia in una griglia di BBox più piccoli.
        step=0.1 gradi corrisponde a circa 11km x 11km.
        """
        min_lon, min_lat, max_lon, max_lat = large_bbox
        bboxes = []
        
        # Genera range usando np.arange
        lon_ranges = np.arange(min_lon, max_lon, step)
        lat_ranges = np.arange(min_lat, max_lat, step)
        
        for lon in lon_ranges:
            for lat in lat_ranges:
                # Crea bbox [min_x, min_y, max_x, max_y]
                sub_bbox = [
                    round(lon, 4), 
                    round(lat, 4), 
                    round(min(lon + step, max_lon), 4), 
                    round(min(lat + step, max_lat), 4)
                ]
                bboxes.append(sub_bbox)
        
        return bboxes

    def download_area_cube(self, bbox):
        """
        Scarica il cubo 4D usando stackstac (Logica presa dalla tua InferencePipeline).
        """
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
                    # Se manca una stagione, saltiamo questo blocco per coerenza
                    return None
                
                # Prendi la migliore (meno nuvole)
                best_item = min(items, key=lambda x: x.properties['eo:cloud_cover'])
                stac_items.append(best_item)
                
            # Stack temporale (4, 6, H, W)
            # EPSG:32633 è ottimale per la Sicilia
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
            
            # Ritorna numpy array (uint16) e calcola subito
            return data.astype("uint16").compute().values
        
        except Exception as e:
            logger.error(f"  ❌ Errore download STAC: {e}")
            return None

    def is_mostly_water_or_empty(self, cube):
        """
        Controlla se il cubo è valido (Terraferma) o inutile (Mare/Vuoto).
        Ritorna True se deve essere SCARTATO.
        """
        # 1. Controllo Vuoto (Tutti zeri o quasi)
        if np.max(cube) == 0:
            return True
            
        # 2. Controllo Mare Aperto (Heuristic)
        # Il canale NIR (Indice 3) è molto scuro sull'acqua (< 500-1000 solitamente).
        # Prendiamo la stagione estiva (indice 2), canale NIR (indice 3)
        nir_summer = cube[2, 3, :, :]
        
        # Se la media del NIR è molto bassa, è probabile mare aperto
        if np.mean(nir_summer) < 400: 
            return True
            
        return False

    def upload_cube_to_minio(self, img_cube, bbox, task_id):
        """Carica su MinIO con metadati"""
        try:
            # Serializza
            buffer = io.BytesIO()
            np.save(buffer, img_cube.astype("uint16"))
            buffer.seek(0)

            # Centroidi per il nome file
            center_lat = (bbox[1] + bbox[3]) / 2
            center_lon = (bbox[0] + bbox[2]) / 2

            metadata = {
                'bbox': json.dumps(bbox),
                'center_lat': str(center_lat),
                'center_lon': str(center_lon),
                'task_id': str(task_id),
                'shape': str(img_cube.shape)
            }

            # Naming: lat_XX_lon_YY_taskID.npy
            lat_short = f"{center_lat:.4f}"
            lon_short = f"{center_lon:.4f}"
            object_name = f"raw_cubes/lat_{lat_short}_lon_{lon_short}_task{task_id}.npy"
            
            self.s3_client.upload_fileobj(
                buffer, 
                self.bucket_name, 
                object_name,
                ExtraArgs={'Metadata': metadata}
            )
            logger.info(f"  📤 Upload OK: {object_name}")
            self.total_cubes_saved += 1
            
        except Exception as e:
            logger.error(f"  ❌ Errore Upload MinIO: {e}")

    def run(self):
        """Esegue il download a griglia su tutta la Sicilia"""
        logger.info("🚀 Avvio Download Regionale Sicilia -> MinIO")
        
        # 1. Definisci i Bounds della Sicilia (Tutta la regione)
        # (min_lon, min_lat, max_lon, max_lat)
        sicily_bbox = (12.40, 36.60, 15.70, 38.30)
        
        # 2. Genera la griglia
        # Usa step=0.1 (circa 11km) per bilanciare velocità e dettaglio
        tiles = self.get_sub_bboxes(sicily_bbox, step=0.1)
        
        logger.info(f"🗺️ Griglia generata: {len(tiles)} tiles totali.")
        
        # 3. Itera e Processa
        for i, bbox in enumerate(tiles):
            
            logger.info(f"📍 Processing Tile {i+1}/{len(tiles)}: {bbox}")
            
            # Download
            cube = self.download_area_cube(bbox)
            
            # Validazione
            if cube is None:
                logger.warning(f"  ⚠️ Download fallito o dati mancanti. Skip.")
                continue
                
            # Filtro Acqua/Vuoto
            if self.is_mostly_water_or_empty(cube):
                logger.info(f"  🌊 Tile scartato (Mare o Vuoto).")
                continue
            
            # Upload
            self.upload_cube_to_minio(cube, bbox, task_id=i)

        logger.info("🏁 Download Completato!")
        logger.info(f"Totale Cubi Salvati su MinIO: {self.total_cubes_saved}")

if __name__ == "__main__":
    downloader = SicilyRegionalDownloader()
    downloader.run()