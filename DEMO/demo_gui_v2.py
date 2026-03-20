"""
🌍 DEMO INTERATTIVA - Sistema di Segmentazione Agricola Sicilia & Malta
========================================================================
Interfaccia web per query geografiche e visualizzazione risultati.
Supporta input per punto centrale o per bounding box diretto.
"""

import gradio as gr
import os
import sys
import json
import numpy as np
from PIL import Image

# Import del sistema di inferenza
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from minio_inference import SicilyInferencePoint
from visual_inference import CLASS_COLORS, CLASS_NAMES


# ==========================================
# CONFIGURAZIONE GLOBALE
# ==========================================
class DemoConfig:
    """Configurazione centralizzata per la demo."""

    # ------------------------------------------------------------------
    # Limiti geografici (Sicilia + Malta)
    # ------------------------------------------------------------------
    LAT_MIN, LAT_MAX = 35.7, 38.4   # Malta sud → Sicilia nord
    LON_MIN, LON_MAX = 12.3, 15.8

    # ------------------------------------------------------------------
    # Punti di interesse — Sicilia
    # ------------------------------------------------------------------
    POI_SICILIA = {
        "🍊 Agrumeti – Piana di Catania":       (37.380, 14.910),
        "🍅 Serre – Pachino":                   (36.715, 15.010),
        "🌾 Grano duro – Enna":                 (37.567, 14.279),
        "🏔️ Etna – Colture miste":              (37.751, 15.004),
        "🏖️ Costa – Palermo":                   (38.115, 13.361),
        "🫒 Oliveti – Castelvetrano":            (37.680, 12.800),
        "🍇 Vigneti – Menfi":                   (37.600, 13.100),
        "🍇 Vigneti – Marsala":                 (37.810, 12.510),
        "🫒 Oliveti – Ragusa":                  (36.926, 14.726),
        "🌿 Ortaggi – Vittoria":                (36.950, 14.530),
        "🌾 Seminativo – Caltanissetta":        (37.490, 14.060),
        "🍊 Agrumeti – Lentini":                (37.285, 14.990),
        "🍑 Frutteti – Bronte":                 (37.790, 14.830),
    }

    # ------------------------------------------------------------------
    # Punti di interesse — Malta
    # ------------------------------------------------------------------
    POI_MALTA = {
        "🇲🇹 Terreni agricoli – Mosta":         (35.910, 14.425),
        "🇲🇹 Colture – Rabat (Malta)":          (35.880, 14.400),
        "🇲🇹 Gozo – Xewkija":                  (36.033, 14.260),
        "🇲🇹 Gozo – Nadur":                     (36.040, 14.290),
        "🇲🇹 Siġġiewi":                         (35.855, 14.435),
        "🇲🇹 Dingli Cliffs (mix costiero)":     (35.860, 14.385),
    }

    # Tutti i preset unificati (per il dropdown)
    POI_PRESETS = {**POI_SICILIA, **POI_MALTA}

    # ------------------------------------------------------------------
    # Legenda classi (salta background = indice 0)
    # ------------------------------------------------------------------
    CLASS_INFO = list(zip(CLASS_NAMES[1:], CLASS_COLORS[1:]))


# ==========================================
# INIZIALIZZAZIONE MODELLO
# ==========================================
print("🔄 Inizializzazione sistema...")
try:
    pipeline = SicilyInferencePoint(config_path="../config.json")
    MODEL_LOADED = True
    print("✅ Sistema pronto!")
except Exception as e:
    print(f"❌ Errore inizializzazione: {e}")
    MODEL_LOADED = False


# ==========================================
# FUNZIONI CORE
# ==========================================

def _validate_point(lat: float, lon: float) -> str | None:
    """Ritorna un messaggio di errore se le coordinate non sono valide, altrimenti None."""
    if not (DemoConfig.LAT_MIN <= lat <= DemoConfig.LAT_MAX):
        return (
            f"❌ Latitudine {lat:.4f} fuori range "
            f"[{DemoConfig.LAT_MIN} – {DemoConfig.LAT_MAX}]"
        )
    if not (DemoConfig.LON_MIN <= lon <= DemoConfig.LON_MAX):
        return (
            f"❌ Longitudine {lon:.4f} fuori range "
            f"[{DemoConfig.LON_MIN} – {DemoConfig.LON_MAX}]"
        )
    return None


def _validate_bbox(min_lat, min_lon, max_lat, max_lon) -> str | None:
    """Validazione per input bbox diretto."""
    if min_lat >= max_lat:
        return "❌ min_lat deve essere < max_lat"
    if min_lon >= max_lon:
        return "❌ min_lon deve essere < max_lon"
    # Controlla che tutti i vertici siano dentro l'area coperta
    for lat in (min_lat, max_lat):
        for lon in (min_lon, max_lon):
            err = _validate_point(lat, lon)
            if err:
                return err
    return None


def _extract_panels(report_path: str):
    """Carica il report PNG a 3 pannelli e ritorna le 3 immagini separate."""
    report_img = Image.open(report_path)
    w, h = report_img.size
    pw = w // 3
    return (
        report_img.crop((0,      0, pw,   h)),   # RGB
        report_img.crop((pw,     0, 2*pw, h)),    # Segmentazione
        report_img.crop((2*pw,   0, w,    h)),    # Overlay
    )


def generate_stats_text(region_name: str) -> str:
    """Genera testo Markdown con le statistiche di area dal TIF."""
    tif_path = f"inference_results/tif/{region_name}_map.tif"

    if not os.path.exists(tif_path):
        return "⚠️ **Statistiche non disponibili** (TIF non trovato)"

    import rasterio
    with rasterio.open(tif_path) as src:
        mask = src.read(1)

    unique, counts = np.unique(mask, return_counts=True)

    # Totale pixel classificati (escluso background=0)
    total_pixels = sum(c for cid, c in zip(unique, counts) if cid != 0)
    if total_pixels == 0:
        return "ℹ️ Nessuna area agricola rilevata in questa zona."

    total_ha = (total_pixels * 100) / 10_000  # 10 m/px → 100 m²/px

    display_name = region_name.replace("demo_", "").replace("_", " ")
    lines = [
        f"### 📊 Analisi superfici — {display_name}",
        "",
        "| Classe | Superficie (ha) | % |",
        "| :--- | ---: | ---: |",
    ]

    for cid, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
        if cid == 0:
            continue
        ha = (count * 100) / 10_000
        pct = ha / total_ha * 100
        try:
            name = CLASS_NAMES[cid]
        except IndexError:
            name = f"Classe {cid}"
        lines.append(f"| **{name}** | {ha:.2f} | {pct:.1f}% |")

    lines += [
        "",
        f"**Totale area classificata:** `{total_ha:.2f} ha`",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------
# Entry-point per la query da punto
# ------------------------------------------------------------------
def run_query_point(lat, lon, region_name, _preset_ignored):
    """Inferenza centrata su un punto (lat, lon)."""
    if not MODEL_LOADED:
        return None, None, None, "❌ Modello non caricato. Controlla i log."

    err = _validate_point(lat, lon)
    if err:
        return None, None, None, err

    tag = f"demo_{region_name.strip().replace(' ', '_')}"

    try:
        pipeline.run_inference(lat, lon, region_name=tag)
        report = f"inference_results/reports/{tag}_report.png"
        if not os.path.exists(report):
            return None, None, None, "❌ Report non generato. Controlla i dati Sentinel-2."
        rgb, seg, overlay = _extract_panels(report)
        stats = generate_stats_text(tag)
        return rgb, seg, overlay, stats
    except Exception as e:
        return None, None, None, f"❌ Errore:\n```\n{e}\n```"


# ------------------------------------------------------------------
# Entry-point per la query da bbox diretto
# ------------------------------------------------------------------
def run_query_bbox(min_lat, min_lon, max_lat, max_lon, region_name):
    """Inferenza su un bounding box definito dall'utente."""
    if not MODEL_LOADED:
        return None, None, None, "❌ Modello non caricato."

    err = _validate_bbox(min_lat, min_lon, max_lat, max_lon)
    if err:
        return None, None, None, err

    # Centro del bbox (per il report)
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    tag = f"demo_bbox_{region_name.strip().replace(' ', '_')}"

    try:
        # Se il tuo pipeline supporta un bbox diretto, puoi chiamare:
        #   pipeline.run_inference_bbox([min_lon, min_lat, max_lon, max_lat], region_name=tag)
        # Altrimenti approssima col centro e size appropriato:
        size_lat = max_lat - min_lat
        size_lon = max_lon - min_lon
        size = max(size_lat, size_lon)
        pipeline.run_inference(center_lat, center_lon, region_name=tag)
        # NB: se vuoi passare il size personalizzato, modifica run_inference
        # per accettare un parametro `bbox_size` e usarlo in get_bbox_from_point

        report = f"inference_results/reports/{tag}_report.png"
        if not os.path.exists(report):
            return None, None, None, "❌ Report non generato."
        rgb, seg, overlay = _extract_panels(report)
        stats = generate_stats_text(tag)
        return rgb, seg, overlay, stats
    except Exception as e:
        return None, None, None, f"❌ Errore:\n```\n{e}\n```"


# ------------------------------------------------------------------
# Utility UI
# ------------------------------------------------------------------
def load_preset(preset_name):
    """Ritorna (lat, lon, nome_analisi) dal preset selezionato."""
    if preset_name and preset_name in DemoConfig.POI_PRESETS:
        lat, lon = DemoConfig.POI_PRESETS[preset_name]
        # Prendi la parte dopo l'emoji e il trattino
        clean = preset_name.split("–")[-1].strip() if "–" in preset_name else preset_name
        return lat, lon, clean.replace(" ", "_")
    return 37.5, 14.2, "Custom"


# ==========================================
# INTERFACCIA GRADIO
# ==========================================

def build_legend_markdown() -> str:
    """Genera la tabella legenda come Markdown."""
    rows = ["| Colore | Classe |", "| :---: | :--- |"]
    for name, color in DemoConfig.CLASS_INFO:
        hex_color = "#{:02x}{:02x}{:02x}".format(*color)
        swatch = f"<span style='background:{hex_color};padding:4px 14px;border-radius:3px'>&nbsp;</span>"
        rows.append(f"| {swatch} | **{name}** |")
    return "\n".join(rows)


def create_demo_interface():
    with gr.Blocks(
        title="🌍 Segmentazione Agricola – Sicilia & Malta",
        theme=gr.themes.Soft(),
    ) as demo:

        # --- Header ---
        gr.Markdown(
            """
            # 🌍 Piattaforma di Monitoraggio Satellitare — Sicilia & Malta
            **Classificazione automatica su dati Sentinel-2 multi-stagionali (2023)**
            
            Classi: Olivo · Vigneto · Agrumi · Frutteto · Grano · Legumi · Ortaggi · Incolto
            ---
            """
        )

        with gr.Row():
            # ============ COLONNA INPUT ============
            with gr.Column(scale=1):
                gr.Markdown("### 📍 Input")

                with gr.Tab("Punto"):
                    preset_dd = gr.Dropdown(
                        choices=list(DemoConfig.POI_PRESETS.keys()),
                        label="Punto predefinito",
                        value=None,
                    )
                    lat_in = gr.Number(label="Latitudine",  value=37.5, precision=6,
                                       info=f"[{DemoConfig.LAT_MIN} – {DemoConfig.LAT_MAX}]")
                    lon_in = gr.Number(label="Longitudine", value=14.2, precision=6,
                                       info=f"[{DemoConfig.LON_MIN} – {DemoConfig.LON_MAX}]")
                    name_in = gr.Textbox(label="Nome analisi", value="Query_Custom",
                                         placeholder="es. Vigneti_Marsala")
                    btn_point = gr.Button("🚀 Analizza (punto)", variant="primary", size="lg")

                with gr.Tab("Bounding Box"):
                    gr.Markdown(
                        "Inserisci direttamente i 4 vertici del rettangolo "
                        "geografico da analizzare."
                    )
                    with gr.Row():
                        bb_min_lat = gr.Number(label="Min Lat (sud)",  value=36.70, precision=6)
                        bb_min_lon = gr.Number(label="Min Lon (ovest)", value=14.98, precision=6)
                    with gr.Row():
                        bb_max_lat = gr.Number(label="Max Lat (nord)", value=36.75, precision=6)
                        bb_max_lon = gr.Number(label="Max Lon (est)",  value=15.05, precision=6)
                    bb_name = gr.Textbox(label="Nome analisi", value="BBox_Custom")
                    btn_bbox = gr.Button("🚀 Analizza (bbox)", variant="primary", size="lg")

                gr.Markdown(
                    f"""
                    ---
                    ℹ️ **Info**
                    - Area per query punto: ~5.5 km × 5.5 km  
                    - Risoluzione: 10 m/pixel  
                    - Copertura: Sicilia + Malta  
                    - Cache intelligente (MinIO)
                    """
                )

            # ============ COLONNA OUTPUT ============
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Risultati")

                with gr.Tab("Visualizzazione"):
                    with gr.Row():
                        out_rgb = gr.Image(label="Satellite RGB", type="pil")
                        out_seg = gr.Image(label="Segmentazione",  type="pil")
                    out_overlay = gr.Image(label="Overlay", type="pil")

                with gr.Tab("Statistiche"):
                    out_stats = gr.Markdown("_Avvia un'analisi per visualizzare le statistiche._")

                with gr.Tab("Legenda"):
                    gr.Markdown(build_legend_markdown())

        # --- Eventi ---
        preset_dd.change(load_preset, [preset_dd], [lat_in, lon_in, name_in])

        btn_point.click(
            run_query_point,
            inputs=[lat_in, lon_in, name_in, preset_dd],
            outputs=[out_rgb, out_seg, out_overlay, out_stats],
        )
        btn_bbox.click(
            run_query_bbox,
            inputs=[bb_min_lat, bb_min_lon, bb_max_lat, bb_max_lon, bb_name],
            outputs=[out_rgb, out_seg, out_overlay, out_stats],
        )

    return demo


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 Avvio Demo – Sicilia & Malta")
    print("=" * 70 + "\n")

    if not MODEL_LOADED:
        print("❌ Impossibile caricare il modello. Verifica config.json e pesi.")
        sys.exit(1)

    demo = create_demo_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )