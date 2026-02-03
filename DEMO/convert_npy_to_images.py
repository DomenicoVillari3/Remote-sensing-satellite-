"""
🖼️ CONVERSIONE NPY → IMMAGINI RGB
===================================
Converte i file .npy esistenti su MinIO in immagini RGB visualizzabili
Estrae la stagione ESTIVA (indice 2) e crea preview PNG
"""

import boto3
import numpy as np
import io
from PIL import Image
import json
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NPYtoImageConverter:
    def __init__(self):
        # Client MinIO
        self.s3_client = boto3.client(
            's3',
            endpoint_url='http://localhost:9000',
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin'
        )
        self.bucket_name = "sicily-sentinel-data"
        
        self.converted_count = 0
        self.failed_count = 0
    
    def percentile_stretch(self, img, p_low=2, p_high=98):
        """
        Normalizzazione percentile per visualizzazione ottimale
        Rimuove outliers e porta in range [0, 255]
        """
        p_low_val, p_high_val = np.percentile(img, (p_low, p_high))
        img_stretched = np.clip((img - p_low_val) / (p_high_val - p_low_val + 1e-6), 0, 1)
        return (img_stretched * 255).astype(np.uint8)
    
    def npy_to_rgb_image(self, npy_array):
        """
        Converte cubo 4D NPY → Immagine RGB visualizzabile
        
        Input: (4, 6, H, W)  [Time, Bands, Height, Width]
        Output: PIL.Image RGB (H, W, 3)
        
        Estrae:
        - Time=2 (Summer, indice 2)
        - Bands: Red(2), Green(1), Blue(0) → RGB
        """
        # 1. Estrai stagione estiva (indice 2)
        summer_cube = npy_array[2]  # Shape: (6, H, W)
        
        # 2. Estrai bande RGB (ordine Sentinel-2: B, G, R, NIR, SWIR1, SWIR2)
        blue = summer_cube[0]   # Banda 0
        green = summer_cube[1]  # Banda 1
        red = summer_cube[2]    # Banda 2
        
        # 3. Stack come RGB (Red, Green, Blue)
        rgb = np.stack([red, green, blue], axis=-1)  # Shape: (H, W, 3)
        
        # 4. Normalizzazione per visualizzazione
        # Valori grezzi Sentinel-2: 0-10000 → 0-255
        rgb_normalized = self.percentile_stretch(rgb)
        
        # 5. Converti in PIL Image
        return Image.fromarray(rgb_normalized, mode='RGB')
    
    def create_thumbnail(self, pil_image, max_size=512):
        """Crea thumbnail per preview veloce"""
        pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return pil_image
    
    def convert_single_npy(self, npy_key):
        """
        Converte un singolo file NPY e lo carica su MinIO come PNG
        
        Returns:
            bool: True se successo, False se fallito
        """
        try:
            # 1. Download NPY da MinIO
            logger.info(f"📥 Download: {npy_key}")
            buffer = io.BytesIO()
            self.s3_client.download_fileobj(self.bucket_name, npy_key, buffer)
            buffer.seek(0)
            
            # 2. Carica array
            npy_array = np.load(buffer)
            
            # Validazione shape
            if npy_array.shape[0] != 4 or npy_array.shape[1] != 6:
                logger.warning(f"  ⚠️  Shape invalido: {npy_array.shape}. Atteso (4, 6, H, W)")
                return False
            
            # 3. Converti a RGB
            rgb_image = self.npy_to_rgb_image(npy_array)
            
            # 4. Salva come PNG in memoria
            png_buffer = io.BytesIO()
            rgb_image.save(png_buffer, format='PNG', optimize=True)
            png_buffer.seek(0)
            
            # 5. Genera chiave output (raw_cubes/*.npy → rgb_images/*.png)
            png_key = npy_key.replace('raw_cubes/', 'rgb_images/').replace('.npy', '.png')
            
            # 6. Recupera metadata originali
            head = self.s3_client.head_object(Bucket=self.bucket_name, Key=npy_key)
            original_metadata = head.get('Metadata', {})
            
            # Aggiungi info immagine
            new_metadata = original_metadata.copy()
            new_metadata['image_type'] = 'rgb_summer'
            new_metadata['image_size'] = f"{rgb_image.width}x{rgb_image.height}"
            new_metadata['source_npy'] = npy_key
            
            # 7. Upload PNG su MinIO
            self.s3_client.upload_fileobj(
                png_buffer,
                self.bucket_name,
                png_key,
                ExtraArgs={
                    'Metadata': new_metadata,
                    'ContentType': 'image/png'
                }
            )
            
            logger.info(f"  ✅ Creato: {png_key} ({rgb_image.width}×{rgb_image.height} px)")
            self.converted_count += 1
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Errore conversione {npy_key}: {e}")
            self.failed_count += 1
            return False
    
    def convert_all_npy(self):
        """
        Converte tutti i file NPY nella directory raw_cubes/
        """
        logger.info("\n" + "="*70)
        logger.info("🎨 CONVERSIONE BATCH NPY → PNG")
        logger.info("="*70 + "\n")
        
        # 1. Lista tutti i file NPY
        logger.info("📋 Ricerca file NPY su MinIO...")
        
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket_name, Prefix='raw_cubes/')
        
        npy_keys = []
        for page in pages:
            if 'Contents' not in page:
                continue
            
            for obj in page['Contents']:
                if obj['Key'].endswith('.npy'):
                    npy_keys.append(obj['Key'])
        
        logger.info(f"✅ Trovati {len(npy_keys)} file NPY da convertire\n")
        
        if len(npy_keys) == 0:
            logger.warning("⚠️  Nessun file NPY trovato in raw_cubes/")
            return
        
        # 2. Converti con progress bar
        logger.info("🔄 Avvio conversione...\n")
        
        for npy_key in tqdm(npy_keys, desc="Conversione"):
            self.convert_single_npy(npy_key)
        
        # 3. Riepilogo finale
        logger.info("\n" + "="*70)
        logger.info("📊 RIEPILOGO CONVERSIONE")
        logger.info("="*70)
        logger.info(f"✅ Convertiti con successo: {self.converted_count}")
        logger.info(f"❌ Falliti: {self.failed_count}")
        logger.info(f"📁 Immagini salvate in: {self.bucket_name}/rgb_images/")
        logger.info("="*70 + "\n")

def main():
    """Esegue la conversione batch"""
    converter = NPYtoImageConverter()
    converter.convert_all_npy()
    
    # Verifica finale
    print("\n🔍 Verifica contenuti bucket:")
    print("="*70)
    
    try:
        s3 = boto3.client(
            's3',
            endpoint_url='http://localhost:9000',
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin'
        )
        
        # Conta NPY
        response_npy = s3.list_objects_v2(
            Bucket='sicily-sentinel-data',
            Prefix='raw_cubes/'
        )
        npy_count = response_npy.get('KeyCount', 0)
        
        # Conta PNG
        response_png = s3.list_objects_v2(
            Bucket='sicily-sentinel-data',
            Prefix='rgb_images/'
        )
        png_count = response_png.get('KeyCount', 0)
        
        print(f"📦 raw_cubes/  : {npy_count} file NPY")
        print(f"🖼️  rgb_images/ : {png_count} file PNG")
        print("="*70)
        
    except Exception as e:
        print(f"⚠️  Impossibile verificare: {e}")

if __name__ == "__main__":
    main()