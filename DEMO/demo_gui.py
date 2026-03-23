"""
🌍 DEMO INTERATTIVA - Sistema di Segmentazione Agricola Sicilia
================================================================
Interfaccia web per query geografiche e visualizzazione risultati
"""

import gradio as gr
import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import io
from minio_inference import SicilyInferencePoint

# Import del sistema di inferenza
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from visual_inference import CLASS_COLORS, CLASS_NAMES

# ==========================================
# CONFIGURAZIONE GLOBALE
# ==========================================
class DemoConfig:
    # Punti di interesse predefiniti per test rapidi
    POI_PRESETS = {
        "🍊 Agrumeti Catania": (37.380, 14.910),
        #"🍷 Vigneti Marsala": (37.810, 12.510),
        #"🫒 Oliveti Ragusa": (36.926, 14.726),

        #"🍅 Serre Pachino": (36.715, 15.010),
        "🌾 Grano Enna": (37.567, 14.279),
        #"🏔️ Etna (Mix)": (37.751, 15.004),
        "🏖️ Palermo Costa": (38.115, 13.361),
        "🌾 Seminativo – Caltanissetta":    (37.490, 14.060),
        "🍊 Agrumeti – Lentini":            (37.285, 14.990),
       # "🍑 Frutteti – Bronte":             (37.790, 14.830),

         # --- Malta ---
        "🇲🇹 Terreni agricoli – Mosta":     (35.910, 14.425),
        #"🇲🇹 Colture – Rabat":              (35.880, 14.400),
        #"🇲🇹 Gozo – Xewkija":              (36.033, 14.260),
       # "🇲🇹 Gozo – Nadur":                 (36.040, 14.290),
       # "🇲🇹 Siġġiewi":                     (35.855, 14.435),
       # "🇲🇹 Dingli Cliffs":                (35.860, 14.385),
    }
    
    # Palette colori per la legenda
    CLASS_INFO = list(zip(CLASS_NAMES[1:], CLASS_COLORS[1:]))  # Skip background

# Inizializzazione pipeline (caricamento modello)
print("🔄 Inizializzazione sistema...")
try:
    pipeline = SicilyInferencePoint(config_path='../config.json')
    MODEL_LOADED = True
    print("✅ Sistema pronto!")
except Exception as e:
    print(f"❌ Errore inizializzazione: {e}")
    MODEL_LOADED = False

# ==========================================
# FUNZIONI CORE
# ==========================================

def run_query(lat, lon, region_name, use_preset):
    """
    Esegue una query di inferenza e ritorna le visualizzazioni
    
    Returns:
        - Image RGB satellitare
        - Image predizione segmentata
        - Image overlay
        - Statistiche testuali
    """
    if not MODEL_LOADED:
        return None, None, None, "❌ Modello non caricato. Controlla i log."
    
    # Validazione input
    if not (-90 <= lat <= 90):
        return None, None, None, "❌ Latitudine non valida (deve essere tra -90 e 90)"
    if not (-180 <= lon <= 180):
        return None, None, None, "❌ Longitudine non valida (deve essere tra -180 e 180)"
    
    # Crea nome file temporaneo
    temp_name = f"demo_{region_name.replace(' ', '_')}"
    
    try:
        # Esegui inferenza (salva automaticamente i report)
        pipeline.run_inference(lat, lon, region_name=temp_name)
        
        # Carica il report generato
        report_path = f"inference_results/reports/{temp_name}_report.png"
        
        if not os.path.exists(report_path):
            return None, None, None, "❌ Errore: Report non generato. Controlla i dati."
        
        # Carica l'immagine del report (3 pannelli)
        report_img = Image.open(report_path)
        
        # Estrai i 3 pannelli separati
        w, h = report_img.size
        panel_width = w // 3
        
        rgb_panel = report_img.crop((0, 0, panel_width, h))
        seg_panel = report_img.crop((panel_width, 0, 2*panel_width, h))
        overlay_panel = report_img.crop((2*panel_width, 0, w, h))
        
        # Genera statistiche testuali
        stats_text = generate_stats_text(temp_name)
        
        return rgb_panel, seg_panel, overlay_panel, stats_text
        
    except Exception as e:
        error_msg = f"❌ Errore durante l'inferenza:\n{str(e)}"
        print(error_msg)
        return None, None, None, error_msg

def generate_stats_text(region_name):
    """Genera testo formattato con le statistiche di area"""
    tif_path = f"inference_results/tif/{region_name}_map.tif"
    
    if not os.path.exists(tif_path):
        return "⚠️ **Statistiche non disponibili**"
    
    import rasterio
    with rasterio.open(tif_path) as src:
        mask = src.read(1)
    
    # Calcola statistiche
    unique, counts = np.unique(mask, return_counts=True)
    
    # 1. Calcolo del totale prima del loop (essenziale per percentuali corrette)
    total_pixels = sum(count for cls_id, count in zip(unique, counts) if cls_id != 0)
    if total_pixels == 0:
        return "ℹ️ Nessuna area agricola rilevata in questa zona."
        
    total_ha_final = (total_pixels * 100) / 10000 
    
    # 2. Formattazione Tabellare per Gradio Markdown
    stats_lines = [
        f"### 📊 ANALISI SUPERFICI: {region_name.replace('demo_', '')}",
        "| Classe | Superficie (ha) | % su Totale |",
        "| :--- | :---: | :---: |"
    ]
    
    for cls_id, count in zip(unique, counts):
        if cls_id == 0:  # Salta background
            continue
        
        hectares = (count * 100) / 10000
        percentage = (hectares / total_ha_final * 100)
        
        # FIX: Accesso alla lista CLASS_NAMES usando l'indice invece di .get()
        # Usiamo un try/except per sicurezza se l'ID classe non esiste nella lista
        try:
            name = CLASS_NAMES[cls_id]
        except IndexError:
            name = f"Classe {cls_id}"
            
        stats_lines.append(f"| **{name}** | {hectares:.2f} | {percentage:.1f}% |")
    
    stats_lines.append("---")
    stats_lines.append(f"*TOTALE AREA CLASSIFICATA**: `{total_ha_final:.2f} ha`")
    
    return "\n".join(stats_lines)

def load_preset(preset_name):
    """Carica le coordinate da un preset"""
    if preset_name in DemoConfig.POI_PRESETS:
        lat, lon = DemoConfig.POI_PRESETS[preset_name]
        return lat, lon, preset_name.split()[1]  # Ritorna anche nome pulito
    return 37.5, 14.2, "Custom"

# ==========================================
# INTERFACCIA GRADIO
# ==========================================

def create_demo_interface():
    """Crea l'interfaccia Gradio"""
    
    with gr.Blocks(
        title="🌍 Demo Segmentazione Agricola Sicilia",
        theme=gr.themes.Soft(),
        css="""
        .header {text-align: center; padding: 20px;}
        .stat-box {background: #f0f0f0; padding: 15px; border-radius: 8px; margin: 10px 0;}
        """
    ) as demo:
        
        # Header
        gr.Markdown(
            """
            # 🌍 Piattaforma di Monitoraggio Satellitare
            ## Analisi Territoriale Avanzata con AI | Classificazione automatica su dati Sentinel-2
            ---
            
            **Classi rilevabili:** Olivo, Vigneto, Agrumi, Frutteto, Grano, Legumi, Ortaggi, Incolto
            """
        )
        
        with gr.Row():
            # Colonna sinistra: Input
            with gr.Column(scale=1):
                gr.Markdown("### 📍 Query Geografica")
                
                # Preset dropdown
                preset_dropdown = gr.Dropdown(
                    choices=list(DemoConfig.POI_PRESETS.keys()),
                    label="🎯 Usa un punto predefinito",
                    value=None
                )
                
                gr.Markdown("**Oppure inserisci coordinate manualmente:**")
                
                lat_input = gr.Number(
                    label="Latitudine",
                    value=37.5,
                    precision=6,
                    info="Range: 36.6 - 38.3 (Sicilia)"
                )
                
                lon_input = gr.Number(
                    label="Longitudine",
                    value=14.2,
                    precision=6,
                    info="Range: 12.4 - 15.7 (Sicilia)"
                )
                
                name_input = gr.Textbox(
                    label="Nome Analisi",
                    value="Query_Custom",
                    placeholder="Es: Vigneti_Marsala"
                )
                
                run_btn = gr.Button("🚀 Avvia Analisi", variant="primary", size="lg")
                
                # Info box
                gr.Markdown(
                    """
                    ---
                    ℹ️ **Info Sistema:**
                    - Area coperta: ~11km x 11km
                    - Risoluzione: 10m/pixel
                    - Dati: Sentinel-2 (4 stagioni 2023)
                    - Cache intelligente (MinIO)
                    """
                )
            
            # Colonna destra: Output
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Risultati Analisi")
                
                with gr.Tab("🖼️ Visualizzazione"):
                    with gr.Row():
                        rgb_output = gr.Image(label="Satellite RGB", type="pil")
                        seg_output = gr.Image(label="Segmentazione", type="pil")
                    
                    overlay_output = gr.Image(label="Overlay (RGB + Predizione)", type="pil")
                
                with gr.Tab("📈 Statistiche"):
                    stats_output = gr.Markdown("Esegui un'analisi per vedere le statistiche")
                
                with gr.Tab("🗺️ Legenda"):
                    # Genera tabella legenda
                    legend_md = "| Colore | Classe | Descrizione |\n|--------|--------|-------------|\n"
                    for name, color in DemoConfig.CLASS_INFO:
                        color_hex = "#{:02x}{:02x}{:02x}".format(*color)
                        legend_md += f"| <span style='background:{color_hex}; padding:5px 15px;'>&nbsp;</span> | **{name}** | Coltivazione {name.lower()} |\n"
                    
                    gr.Markdown(legend_md)
        
        # Eventi
        # Carica preset
        preset_dropdown.change(
            fn=load_preset,
            inputs=[preset_dropdown],
            outputs=[lat_input, lon_input, name_input]
        )
        
        # Esegui query
        run_btn.click(
            fn=run_query,
            inputs=[lat_input, lon_input, name_input, preset_dropdown],
            outputs=[rgb_output, seg_output, overlay_output, stats_output]
        )
    
    return demo

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 Avvio Demo Interattiva")
    print("="*70 + "\n")
    
    if not MODEL_LOADED:
        print("❌ ERRORE: Impossibile caricare il modello!")
        print("   Verifica che config.json e i pesi del modello siano disponibili.")
        exit(1)
    
    # Crea e lancia interfaccia
    demo = create_demo_interface()
    
    demo.launch(
        server_name="0.0.0.0",  # Accessibile da rete locale
        server_port=7860,
        share=False,  # Metti True per link pubblico temporaneo
        show_error=True
    )