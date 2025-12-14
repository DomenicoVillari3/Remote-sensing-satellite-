

print(f"--- Calcolo Statistiche Dataset in: {IMAGES_DIR} ---")

# 1. Trova tutti i file
files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(EXT)]
if len(files) == 0:
    print("Errore: Nessun file .tif trovato nella directory!")
    raise FileNotFoundError("Nessun file .tif trovato nella directory specificata.")

print(f"Trovate {len(files)} immagini. Inizio calcolo...")

# Inizializza accumulatori per il metodo di Welford (o somma semplice)
# Usiamo float64 per evitare overflow durante le somme di numeri grandi (uint16)
channel_sum = None
channel_sq_sum = None
total_pixel_count = 0
num_channels = 0

# 2. Itera su tutte le immagini
for file_name in tqdm(files):
    path = os.path.join(IMAGES_DIR, file_name)
    
    try:
        with rasterio.open(path) as src:
            # Legge l'immagine: (Channels, Height, Width)
            img = src.read().astype(np.float64)
            
            # Inizializza i vettori alla prima iterazione (così si adatta a 10, 12 o 13 bande)
            if channel_sum is None:
                num_channels = img.shape[0]
                channel_sum = np.zeros(num_channels, dtype=np.float64)
                channel_sq_sum = np.zeros(num_channels, dtype=np.float64)
            
            # Calcola il numero di pixel in questa immagine (H * W)
            # Nota: Se hai nodata=0 e vuoi ignorarli, la logica diventerebbe più complessa.
            # Per ora assumiamo che le patch siano piene o che lo 0 sia significativo.
            num_pixels = img.shape[1] * img.shape[2]
            
            # Aggiorna le somme per ogni canale
            # Sum over axis (1,2) schiaccia H e W, lasciando un valore per canale
            channel_sum += np.sum(img, axis=(1, 2))
            channel_sq_sum += np.sum(img ** 2, axis=(1, 2))
            
            total_pixel_count += num_pixels

    except Exception as e:
        print(f"Errore leggendo {file_name}: {e}")
        continue

if total_pixel_count == 0:
    print("Errore: Conteggio pixel zero.")
    raise ValueError("Nessun pixel valido trovato nelle immagini.")

# 3. Calcolo Finale
# Mean = Sum / N
means = channel_sum / total_pixel_count

# Std = Sqrt( E[x^2] - (E[x])^2 )
stds = np.sqrt((channel_sq_sum / total_pixel_count) - (means ** 2))

# 4. Stampa risultati formattati per il copia-incolla
print("\n" + "="*40)
print("RISULTATI DA COPIARE IN CONFIG:")
print("="*40)

# Formattazione precisa per lista Python
mean_str = ", ".join([f"{m:.4f}" for m in means])
std_str = ", ".join([f"{s:.4f}" for s in stds])

print(f'"MEANS": [{mean_str}],')
print(f'"STDS":  [{std_str}]')
print("\nNote:")
print(f"- Numero canali rilevati: {num_channels}")
print(f"- Pixel totali analizzati: {total_pixel_count}")
print("Incolla le due righe sopra nel dizionario CONFIG in train_sentinel.py")

print("="*40 + "\n")