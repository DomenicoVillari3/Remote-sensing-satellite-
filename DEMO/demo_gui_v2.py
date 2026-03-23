"""
🌍 DEMO INTERATTIVA — Segmentazione Agricola Sicilia & Malta
=============================================================
Supporta: input punto, input bbox diretto, 19 punti predefiniti.
Fix: niente più immagini tagliate grazie a margine + overlap.
"""

import gradio as gr
import os
import sys
import json
import numpy as np
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from minio_inference_v2 import SicilyInferencePoint
from visual_inference import CLASS_COLORS, CLASS_NAMES


# ==========================================
# CONFIGURAZIONE
# ==========================================
class Config:
    LAT_MIN, LAT_MAX = 35.7, 38.4
    LON_MIN, LON_MAX = 12.3, 15.8

    POI = {
        # --- Sicilia ---
        "🍊 Agrumeti – Piana di Catania":   (37.380, 14.910),
        #"🍅 Serre – Pachino":               (36.715, 15.010),
        "🌾 Grano duro – Enna":             (37.567, 14.279),
        "🏔️ Etna – Colture miste":          (37.751, 15.004),
        "🏖️ Costa – Palermo":               (38.115, 13.361),
        "🫒 Oliveti – Castelvetrano":        (37.680, 12.800),
        #"🍇 Vigneti – Menfi":               (37.600, 13.100),
        #"🍇 Vigneti – Marsala":             (37.810, 12.510),
        #"🫒 Oliveti – Ragusa":              (36.926, 14.726),
        #"🌿 Ortaggi – Vittoria":            (36.950, 14.530),
        "🌾 Seminativo – Caltanissetta":    (37.490, 14.060),
        "🍊 Agrumeti – Lentini":            (37.285, 14.990),
        "🍑 Frutteti – Bronte":             (37.790, 14.830),
        # --- Malta ---
        "🇲🇹 Terreni agricoli – Mosta":     (35.910, 14.425),
        "🇲🇹 Colture – Rabat":              (35.880, 14.400),
        "🇲🇹 Gozo – Xewkija":              (36.033, 14.260),
        "🇲🇹 Gozo – Nadur":                 (36.040, 14.290),
        "🇲🇹 Siġġiewi":                     (35.855, 14.435),
        "🇲🇹 Dingli Cliffs":                (35.860, 14.385),
    }

    CLASS_INFO = list(zip(CLASS_NAMES[1:], CLASS_COLORS[1:]))

    BBOX_SIZE_OPTIONS = {
        "Piccola (~5.5 km)":  0.05,
        "Media (~11 km)":     0.10,
        "Grande (~16.5 km)":  0.15,
       ## "XL (~22 km)":        0.20,
    }
    


# ==========================================
# INIZIALIZZAZIONE
# ==========================================
print("🔄 Inizializzazione sistema...")
try:
    pipeline = SicilyInferencePoint(config_path="../config.json")
    MODEL_LOADED = True
    print("✅ Sistema pronto!")
except Exception as e:
    print(f"❌ Errore: {e}")
    MODEL_LOADED = False


# ==========================================
# HELPERS
# ==========================================
def _validate_point(lat, lon):
    if not (Config.LAT_MIN <= lat <= Config.LAT_MAX):
        return f"❌ Lat {lat:.4f} fuori [{Config.LAT_MIN}–{Config.LAT_MAX}]"
    if not (Config.LON_MIN <= lon <= Config.LON_MAX):
        return f"❌ Lon {lon:.4f} fuori [{Config.LON_MIN}–{Config.LON_MAX}]"
    return None


def _validate_bbox(s, w, n, e):
    if s >= n:
        return "❌ Min Lat deve essere < Max Lat"
    if w >= e:
        return "❌ Min Lon deve essere < Max Lon"
    for lat in (s, n):
        for lon in (w, e):
            err = _validate_point(lat, lon)
            if err:
                return err
    return None


def _extract_panels(report_path):
    img = Image.open(report_path)
    w, h = img.size
    pw = w // 3
    return img.crop((0, 0, pw, h)), img.crop((pw, 0, 2*pw, h)), img.crop((2*pw, 0, w, h))


def _stats_text(region_name):
    tif = f"inference_results/tif/{region_name}_map.tif"
    if not os.path.exists(tif):
        return "⚠️ Statistiche non disponibili"

    import rasterio
    with rasterio.open(tif) as src:
        mask = src.read(1)

    unique, counts = np.unique(mask, return_counts=True)
    total_px = sum(c for u, c in zip(unique, counts) if u != 0)
    if total_px == 0:
        return "ℹ️ Nessuna area agricola rilevata."

    total_ha = (total_px * 100) / 10_000
    display = region_name.replace("demo_", "").replace("bbox_", "").replace("_", " ")

    lines = [
        f"### 📊 Analisi — {display}", "",
        "| Classe | ha | % |",
        "| :--- | ---: | ---: |",
    ]
    for u, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
        if u == 0:
            continue
        ha = (c * 100) / 10_000
        pct = ha / total_ha * 100
        try:
            name = CLASS_NAMES[u]
        except IndexError:
            name = f"Classe {u}"
        lines.append(f"| **{name}** | {ha:.2f} | {pct:.1f}% |")
    lines += ["", f"**Totale:** `{total_ha:.2f} ha`"]
    return "\n".join(lines)


# ==========================================
# QUERY HANDLERS
# ==========================================
def run_point(lat, lon, name, _preset, area_size):
    if not MODEL_LOADED:
        return None, None, None, "❌ Modello non caricato."
    err = _validate_point(lat, lon)
    if err:
        return None, None, None, err

    tag = f"demo_{name.strip().replace(' ', '_')}"
    size_deg = Config.BBOX_SIZE_OPTIONS.get(area_size, 0.10)

    try:
        pipeline.run_inference(lat, lon, region_name=tag, bbox_size=size_deg)
        report = f"inference_results/reports/{tag}_report.png"
        if not os.path.exists(report):
            return None, None, None, "❌ Report non generato."
        rgb, seg, overlay = _extract_panels(report)
        return rgb, seg, overlay, _stats_text(tag)
    except Exception as e:
        return None, None, None, f"❌ Errore:\n```\n{e}\n```"


def run_bbox(s, w, n, e, name):
    if not MODEL_LOADED:
        return None, None, None, "❌ Modello non caricato."
    err = _validate_bbox(s, w, n, e)
    if err:
        return None, None, None, err

    tag = f"demo_bbox_{name.strip().replace(' ', '_')}"
    bbox = [w, s, e, n]  # [min_lon, min_lat, max_lon, max_lat]

    try:
        pipeline.run_inference_bbox(bbox, region_name=tag)
        report = f"inference_results/reports/{tag}_report.png"
        if not os.path.exists(report):
            return None, None, None, "❌ Report non generato."
        rgb, seg, overlay = _extract_panels(report)
        return rgb, seg, overlay, _stats_text(tag)
    except Exception as e:
        return None, None, None, f"❌ Errore:\n```\n{e}\n```"


def load_preset(preset):
    if preset and preset in Config.POI:
        lat, lon = Config.POI[preset]
        clean = preset.split("–")[-1].strip() if "–" in preset else preset
        return lat, lon, clean.replace(" ", "_")
    return 37.5, 14.2, "Custom"


def build_legend():
    rows = ["| Colore | Classe |", "| :---: | :--- |"]
    for name, color in Config.CLASS_INFO:
        hex_c = "#{:02x}{:02x}{:02x}".format(*color)
        rows.append(f"| <span style='background:{hex_c};padding:4px 14px;border-radius:3px'>&nbsp;</span> | **{name}** |")
    return "\n".join(rows)


# ==========================================
# INTERFACCIA GRADIO
# ==========================================
def create_interface():
    with gr.Blocks(
        title="🌍 Segmentazione Agricola – Sicilia & Malta",
        theme=gr.themes.Soft(),
    ) as app:

        gr.Markdown(
            """
            # 🌍 Piattaforma di Monitoraggio Satellitare — Sicilia & Malta
            **ALMA DIGIT · Progetto INTERREG · Prithvi-2.0 ViT · Sentinel-2 (2023)**
            
            Classi: Olivo · Vigneto · Agrumi · Frutteto · Grano · Legumi · Ortaggi · Incolto
            ---
            """
        )

        with gr.Row():
            # === INPUT ===
            with gr.Column(scale=1):
                gr.Markdown("### 📍 Input")

                with gr.Tab("Punto"):
                    preset_dd = gr.Dropdown(
                        choices=list(Config.POI.keys()),
                        label="Punto predefinito",
                        value=None,
                    )
                    lat_in = gr.Number(label="Latitudine", value=37.5, precision=6,
                                       info=f"[{Config.LAT_MIN} – {Config.LAT_MAX}]")
                    lon_in = gr.Number(label="Longitudine", value=14.2, precision=6,
                                       info=f"[{Config.LON_MIN} – {Config.LON_MAX}]")
                    name_in = gr.Textbox(label="Nome analisi", value="Query_Custom")
                    area_dd = gr.Dropdown(
                        choices=list(Config.BBOX_SIZE_OPTIONS.keys()),
                        label="Dimensione area",
                        value="Media (~11 km)",
                    )
                    btn_point = gr.Button("🚀 Analizza (punto)", variant="primary", size="lg")

                with gr.Tab("Bounding Box"):
                    gr.Markdown("Inserisci i 4 vertici del rettangolo geografico.")
                    with gr.Row():
                        bb_s = gr.Number(label="Sud (Min Lat)", value=36.70, precision=6)
                        bb_w = gr.Number(label="Ovest (Min Lon)", value=14.98, precision=6)
                    with gr.Row():
                        bb_n = gr.Number(label="Nord (Max Lat)", value=36.75, precision=6)
                        bb_e = gr.Number(label="Est (Max Lon)", value=15.05, precision=6)
                    bb_name = gr.Textbox(label="Nome analisi", value="BBox_Custom")
                    btn_bbox = gr.Button("🚀 Analizza (bbox)", variant="primary", size="lg")

                gr.Markdown(
                    f"""
                    ---
                    ℹ️ **Info**
                    - Risoluzione: 10 m/pixel  
                    - Dati: Sentinel-2, 4 stagioni 2023
                    - Copertura: Sicilia + Malta  
                    - Overlap inference: 32 px (no artefatti bordo)
                    - Margine auto: ~1.7 km (no immagini tagliate)
                    """
                )

            # === OUTPUT ===
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Risultati")

                with gr.Tab("Visualizzazione"):
                    with gr.Row():
                        out_rgb = gr.Image(label="Satellite RGB", type="pil")
                        out_seg = gr.Image(label="Segmentazione", type="pil")
                    out_overlay = gr.Image(label="Overlay", type="pil")

                with gr.Tab("Statistiche"):
                    out_stats = gr.Markdown("_Avvia un'analisi per visualizzare le statistiche._")

                with gr.Tab("Legenda"):
                    gr.Markdown(build_legend())

        # --- Eventi ---
        preset_dd.change(load_preset, [preset_dd], [lat_in, lon_in, name_in])

        btn_point.click(
            run_point,
            inputs=[lat_in, lon_in, name_in, preset_dd, area_dd],
            outputs=[out_rgb, out_seg, out_overlay, out_stats],
        )
        btn_bbox.click(
            run_bbox,
            inputs=[bb_s, bb_w, bb_n, bb_e, bb_name],
            outputs=[out_rgb, out_seg, out_overlay, out_stats],
        )

    return app


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    if not MODEL_LOADED:
        print("❌ Impossibile caricare il modello.")
        sys.exit(1)

    app = create_interface()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)