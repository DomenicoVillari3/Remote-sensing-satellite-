"""
🌍 API REST - Riconoscimento Colture (Solo Inferenza)
======================================================
Nessun database. Solo modello + analisi + immagini.

Avvio:  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
Docs:   http://localhost:8000/docs

Endpoint:
  POST /api/v1/analysis/run                         → Lancia analisi asincrona
  GET  /api/v1/analysis/{task_id}/status             → Polling stato + risultati
  GET  /api/v1/analysis/{task_id}/images/{type}      → Scarica immagini PNG
  GET  /api/v1/classes                               → Catalogo classi + mapping
  GET  /api/v1/ndvi/{task_id}/timeseries             → Serie temporale NDVI (4 stagioni)
  GET  /api/v1/health                                → Stato sistema
"""

import os
import json
import uuid
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ==========================================
# IMPORT PROGETTO
# ==========================================
from train import PrithviSegmentation4090
from terratorch.registry import BACKBONE_REGISTRY
from visual_inference import CLASS_COLORS, CLASS_NAMES, colorize_mask_rgb

# ==========================================
# CONFIG
# ==========================================
CONFIG_PATH = "config.json"
RESULTS_DIR = Path("api_results")
RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "images").mkdir(exist_ok=True)

PIXEL_AREA_M2 = 100
M2_TO_HECTARES = 10_000
BAND_RED = 2   # indice nel cubo [blue, green, red, nir08, swir16, swir22]
BAND_NIR = 3

# Mapping nomi colture comuni → classi del modello.
# Serve al frontend per sapere come tradurre le dichiarazioni.
DECLARATION_TO_MODEL_CLASS = {
    "Olivo": "Olivo", "Vite": "Vite", "Agrumi": "Agrumi",
    "Frutteto": "Frutteto", "Grano": "Grano", "Legumi": "Legumi",
    "Ortaggi": "Ortaggi", "Incolto": "Incolto",
    "Mais": "Grano", "Granturco": "Grano", "Soia": "Legumi",
    "Girasole": "Grano", "Orzo": "Grano", "Avena": "Grano",
    "Pomodoro": "Ortaggi", "Zucchina": "Ortaggi", "Melanzana": "Ortaggi",
    "Arancio": "Agrumi", "Limone": "Agrumi", "Mandarino": "Agrumi",
    "Mandorlo": "Frutteto", "Pistacchio": "Frutteto", "Pesco": "Frutteto",
    "Vigneto": "Vite", "Uva": "Vite",
    "Fagiolo": "Legumi", "Cece": "Legumi", "Lenticchia": "Legumi", "Fava": "Legumi",
}


# ==========================================
# PYDANTIC MODELS
# ==========================================

class PointInput(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    region_name: Optional[str] = None

class BBoxInput(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    region_name: Optional[str] = None

class AnalysisRequest(BaseModel):
    """Invia `point` OPPURE `bbox`."""
    point: Optional[PointInput] = None
    bbox: Optional[BBoxInput] = None


# ==========================================
# MODELLO SINGLETON
# ==========================================

class ModelService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def initialize(self, config_path: str = CONFIG_PATH):
        if self._initialized:
            return
        with open(config_path) as f:
            self.config = json.load(f)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chip_size = 224

        model_path = self.config["paths"]["model_save_path"]
        print(f"🔄 [API] Caricamento modello: {model_path}")
        backbone = BACKBONE_REGISTRY.build(self.config["project_meta"]["backbone_model"], pretrained=False)
        self.model = PrithviSegmentation4090(backbone, self.config["data_specs"]["num_classes"])
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state.items()})
        self.model.to(self.device).eval()

        self.means = torch.tensor(self.config["data_specs"]["normalization"]["means"]).view(1,1,6,1,1).to(self.device).float()
        self.stds  = torch.tensor(self.config["data_specs"]["normalization"]["stds"]).view(1,1,6,1,1).to(self.device).float()
        self._initialized = True
        print(f"✅ [API] Modello pronto su {self.device}")

    @property
    def is_ready(self): return self._initialized

    def predict_chip(self, chip_tensor: torch.Tensor):
        """(pred_mask, confidence_map, probs_full) per un chip 224x224."""
        with torch.no_grad():
            logits = self.model(chip_tensor)           # (1, C, H, W)
            probs = torch.softmax(logits, dim=1)       # softmax → probabilità
            conf, pred = probs.max(dim=1)              # classe + confidence
        return pred.squeeze(0).cpu().numpy(), conf.squeeze(0).cpu().numpy(), probs.squeeze(0).cpu().numpy()


# ==========================================
# CALCOLI
# ==========================================

def calculate_ndvi(cube: np.ndarray) -> dict:
    """NDVI per ciascuna delle 4 stagioni + media (riferimento estate)."""
    seasons = ['winter', 'spring', 'summer', 'autumn']
    per_season = {}
    for t, s in enumerate(seasons):
        if t >= cube.shape[0]: break
        red = cube[t, BAND_RED].astype(np.float64)
        nir = cube[t, BAND_NIR].astype(np.float64)
        d = nir + red
        ndvi = np.where(d > 0, (nir - red) / d, 0.0)
        valid = (d > 0) & (ndvi > -0.1)
        per_season[s] = round(float(ndvi[valid].mean()), 3) if valid.any() else 0.0
    mean = per_season.get('summer', float(np.mean(list(per_season.values()))))
    return {"ndvi_mean": round(mean, 3), "ndvi_per_season": per_season}


def calculate_class_distribution(pred_mask, confidence_map, total_pixels):
    """Distribuzione %, ettari, confidence per classe."""
    distribution = []
    confidence_per_class = {}
    for cid in range(len(CLASS_NAMES)):
        m = (pred_mask == cid)
        count = int(m.sum())
        if count == 0: continue
        pct = round(count / total_pixels * 100, 1)
        ha  = round(count * PIXEL_AREA_M2 / M2_TO_HECTARES, 2)
        c   = round(float(confidence_map[m].mean()) * 100, 1)
        distribution.append({
            "class_name": CLASS_NAMES[cid], "class_id": cid,
            "percentage": pct, "hectares": ha, "color_rgb": CLASS_COLORS[cid],
        })
        confidence_per_class[CLASS_NAMES[cid]] = c
    distribution.sort(key=lambda x: x["percentage"], reverse=True)
    non_bg = [d for d in distribution if d["class_id"] != 0]
    dominant = non_bg[0]["class_name"] if non_bg else "Sfondo"
    return distribution, confidence_per_class, dominant


def generate_images(cube, pred_mask, confidence_map, task_id) -> dict:
    """Genera 5 PNG e ritorna dict con URL per ciascuno."""
    out = RESULTS_DIR / "images"
    H, W = pred_mask.shape
    urls = {}

    # RGB normalizzato (estate)
    rgb = cube[2, [2,1,0]].transpose(1,2,0)
    p2, p98 = np.percentile(rgb, (2,98))
    rgb_n = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)

    # NDVI map
    red = cube[2, BAND_RED].astype(np.float64)
    nir = cube[2, BAND_NIR].astype(np.float64)
    d = nir + red
    ndvi_map = np.where(d > 0, (nir - red) / d, 0)

    # Overlay RGBA
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    for i, c in enumerate(CLASS_COLORS):
        if i == 0: continue
        m = pred_mask == i
        overlay[m, :3] = [v/255 for v in c]
        overlay[m, 3] = 0.45

    specs = {
        "rgb":           (rgb_n,                  {},                            "Satellite RGB (Estate)"),
        "segmentation":  (colorize_mask_rgb(pred_mask), {},                     "Colture Rilevate"),
        "overlay":       (rgb_n,                  {"overlay": overlay},          "Overlay Satellite + Colture"),
        "confidence":    (confidence_map,         {"cmap":"RdYlGn","vmin":0.3,"vmax":1.0}, "Confidence Modello"),
        "ndvi":          (ndvi_map,               {"cmap":"RdYlGn","vmin":-0.2,"vmax":0.9},"NDVI Vigore Vegetativo (Estate)"),
    }

    for name, (data, kwargs, title) in specs.items():
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(data, **{k:v for k,v in kwargs.items() if k != "overlay"})
        if "overlay" in kwargs:
            ax.imshow(kwargs["overlay"])
        if name == "segmentation":
            ax.legend(
                handles=[mpatches.Patch(color=[v/255 for v in CLASS_COLORS[i]], label=CLASS_NAMES[i]) for i in range(1, len(CLASS_NAMES))],
                loc='lower right', fontsize=7
            )
        if name in ("confidence", "ndvi"):
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.axis('off')
        fig.savefig(out / f"{task_id}_{name}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        urls[name] = f"/api/v1/analysis/{task_id}/images/{name}"

    return urls


# ==========================================
# DOWNLOAD SENTINEL-2
# ==========================================

def download_cube(lat: float, lon: float, size: float = 0.02) -> np.ndarray:
    import pystac_client, stackstac
    bbox = [lon-size, lat-size, lon+size, lat+size]
    periods = {
        'winter': "2023-01-01/2023-02-28", 'spring': "2023-04-15/2023-05-30",
        'summer': "2023-07-01/2023-08-15",  'autumn': "2023-10-01/2023-11-15",
    }
    assets = ["blue","green","red","nir08","swir16","swir22"]
    catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
    items = []
    for season in ['winter','spring','summer','autumn']:
        res = catalog.search(collections=["sentinel-2-l2a"], bbox=bbox,
                             datetime=periods[season], query={"eo:cloud_cover":{"lt":20}})
        found = res.item_collection()
        if not found: raise ValueError(f"Dati mancanti per: {season}")
        items.append(min(found, key=lambda x: x.properties['eo:cloud_cover']))
    return stackstac.stack(items, assets=assets, bounds_latlon=bbox,
                           resolution=10, epsg=32633, fill_value=0,
                           rescale=False).astype("float32").compute()


# ==========================================
# PIPELINE (eseguita in background thread)
# ==========================================

tasks: dict[str, dict]   = {}   # stato dei job
results: dict[str, dict] = {}   # risultati completati

executor = ThreadPoolExecutor(max_workers=1)  # 1 job alla volta (GPU singola)


def _run_pipeline(task_id: str, req: dict):
    try:
        # ── Download ──
        tasks[task_id]["status"]   = "downloading"
        tasks[task_id]["progress"] = 10

        pt = req.get("point")
        bx = req.get("bbox")
        if pt:
            name = pt.get("region_name") or f"pt_{pt['lat']:.3f}_{pt['lon']:.3f}"
            cube = download_cube(pt["lat"], pt["lon"])
        else:
            clat = (bx["min_lat"]+bx["max_lat"])/2
            clon = (bx["min_lon"]+bx["max_lon"])/2
            sz   = max((bx["max_lon"]-bx["min_lon"])/2, (bx["max_lat"]-bx["min_lat"])/2)
            name = bx.get("region_name") or f"bbox_{clat:.3f}"
            cube = download_cube(clat, clon, sz)

        # ── Inferenza ──
        tasks[task_id]["status"]   = "processing"
        tasks[task_id]["progress"] = 40

        svc = ModelService()
        T, C, H, W = cube.shape
        cs = svc.chip_size
        ph = (cs - H % cs) % cs
        pw = (cs - W % cs) % cs
        padded = np.pad(cube, ((0,0),(0,0),(0,ph),(0,pw)), mode='constant')
        nH, nW = padded.shape[2], padded.shape[3]

        pred_map = np.zeros((nH, nW), dtype=np.uint8)
        conf_map = np.zeros((nH, nW), dtype=np.float32)

        total_chips = (nH // cs) * (nW // cs)
        done = 0
        for y in range(0, nH, cs):
            for x in range(0, nW, cs):
                chip = padded[:, :, y:y+cs, x:x+cs]
                inp = torch.from_numpy(chip).unsqueeze(0).to(svc.device).float()
                inp = (inp - svc.means) / (svc.stds + 1e-6)
                p, c, _ = svc.predict_chip(inp)
                pred_map[y:y+cs, x:x+cs] = p
                conf_map[y:y+cs, x:x+cs] = c
                done += 1
                tasks[task_id]["progress"] = 40 + int(done / total_chips * 40)

        # Unpad + filtro acqua
        pred_map = pred_map[:H, :W]
        conf_map = conf_map[:H, :W]
        cube = cube[:, :, :H, :W]
        water = cube[2, BAND_NIR] < 400
        pred_map[water] = 0
        conf_map[water] = 0

        tasks[task_id]["progress"] = 85

        # ── Calcoli ──
        total_px = int((pred_map > 0).sum()) or (H * W)
        ndvi       = calculate_ndvi(cube)
        dist, cpc, dominant = calculate_class_distribution(pred_map, conf_map, total_px)
        vc = conf_map[pred_map > 0]
        conf_mean = round(float(vc.mean()) * 100, 1) if len(vc) else 0.0
        total_ha  = round(total_px * PIXEL_AREA_M2 / M2_TO_HECTARES, 2)

        # ── Immagini ──
        img_urls = generate_images(cube, pred_map, conf_map, task_id)

        tasks[task_id]["progress"] = 95

        # ── Risultato ──
        results[task_id] = {
            "task_id":              task_id,
            "status":               "completed",
            "timestamp":            datetime.now().isoformat(),
            "region_name":          name,
            "total_hectares":       total_ha,
            "ndvi_mean":            ndvi["ndvi_mean"],
            "ndvi_per_season":      ndvi["ndvi_per_season"],
            "confidence_mean":      conf_mean,
            "dominant_class":       dominant,
            "class_distribution":   dist,
            "confidence_per_class": cpc,
            "images":               img_urls,
        }

        tasks[task_id]["status"]   = "completed"
        tasks[task_id]["progress"] = 100

    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"]  = str(e)
        print(f"❌ [API] Task {task_id}: {e}")


# ##############################################
#
#                 FASTAPI APP
#
# ##############################################

app = FastAPI(
    title="🌍 SmartFood — API Riconoscimento Colture",
    description="Analisi satellitare: segmentazione, NDVI, confidence. Nessun database richiesto.",
    version="2.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup():
    ModelService().initialize(CONFIG_PATH)


# ══════════════════════════════════════════
# 1 — POST /api/v1/analysis/run
# ══════════════════════════════════════════
@app.post("/api/v1/analysis/run", tags=["Analisi"],
          summary="🚀 Lancia analisi satellitare (asincrono)")
async def run_analysis(request: AnalysisRequest):
    """
    Riceve `point` (lat/lon) oppure `bbox` (4 coordinate).
    Ritorna **subito** un `task_id`.
    Il frontend fa polling su `/analysis/{task_id}/status`.

    **Pipeline interna:**
    1. Download cubo Sentinel-2 multi-temporale (4 stagioni × 6 bande)
    2. Inferenza Prithvi → segmentazione pixel-per-pixel
    3. Softmax sui logits → **confidence** per classe e per pixel
    4. Calcolo **NDVI** = (NIR − Red) / (NIR + Red) per ciascuna stagione
    5. Generazione immagini: RGB, segmentazione, overlay, confidence, NDVI
    6. Aggregazione statistiche: distribuzione classi (%), ettari, confidence media
    """
    svc = ModelService()
    if not svc.is_ready:
        raise HTTPException(503, "Modello non ancora caricato")
    if request.point is None and request.bbox is None:
        raise HTTPException(400, "Specificare 'point' oppure 'bbox'")

    task_id = str(uuid.uuid4())[:12]
    tasks[task_id] = {
        "task_id": task_id, "status": "queued", "progress": 0,
        "created_at": datetime.now().isoformat(), "error": None,
    }

    req_data = {
        "point": request.point.model_dump() if request.point else None,
        "bbox":  request.bbox.model_dump()  if request.bbox  else None,
    }
    executor.submit(_run_pipeline, task_id, req_data)

    return {"task_id": task_id, "status": "queued"}


# ══════════════════════════════════════════
# 2 — GET /api/v1/analysis/{task_id}/status
# ══════════════════════════════════════════
@app.get("/api/v1/analysis/{task_id}/status", tags=["Analisi"],
         summary="📊 Stato analisi + risultati quando completata")
async def get_status(task_id: str):
    """
    Il frontend chiama questo ogni 3-5 secondi.

    **Stati:** `queued` → `downloading` → `processing` → `completed` | `failed`

    Il campo `progress` (0-100) alimenta la progress bar.
    Quando `status == "completed"`, il campo `result` contiene il JSON completo
    con distribuzione classi, confidence, NDVI, URL immagini.
    """
    if task_id not in tasks:
        raise HTTPException(404, "Task non trovato")

    t = tasks[task_id]
    resp = {
        "task_id":    task_id,
        "status":     t["status"],
        "progress":   t.get("progress", 0),
        "created_at": t.get("created_at"),
    }
    if t["status"] == "completed" and task_id in results:
        resp["result"] = results[task_id]
    if t["status"] == "failed":
        resp["error"] = t.get("error", "Errore sconosciuto")
    return resp


# ══════════════════════════════════════════
# 3 — GET /api/v1/analysis/{task_id}/images/{type}
# ══════════════════════════════════════════
@app.get("/api/v1/analysis/{task_id}/images/{image_type}", tags=["Analisi"],
         summary="🖼️ Scarica immagine risultato (PNG)")
async def get_image(task_id: str, image_type: str):
    """
    **Tipi:** `rgb` · `segmentation` · `overlay` · `confidence` · `ndvi`

    Ritorna il file PNG direttamente (content-type: image/png).
    Il frontend lo usa come `<img src="...">`.
    """
    valid = ["rgb", "segmentation", "overlay", "confidence", "ndvi"]
    if image_type not in valid:
        raise HTTPException(400, f"Tipo non valido. Opzioni: {valid}")
    path = RESULTS_DIR / "images" / f"{task_id}_{image_type}.png"
    if not path.exists():
        raise HTTPException(404, "Immagine non trovata (analisi ancora in corso?)")
    return FileResponse(path, media_type="image/png")


# ══════════════════════════════════════════
# 4 — GET /api/v1/classes
# ══════════════════════════════════════════
@app.get("/api/v1/classes", tags=["Metadata"],
         summary="🎨 Catalogo classi del modello + mapping dichiarazioni")
async def get_classes():
    """
    Ritorna le 9 classi del modello con colori RGB/hex, più il dizionario
    di mapping tra nomi usati dai produttori e classi del modello.

    Il frontend usa questo per:
    - Costruire la **legenda** colori sulla mappa
    - Popolare i **dropdown** di selezione coltura
    - Tradurre "Mais" → "Grano" nel confronto dichiarato vs rilevato
    """
    return {
        "num_classes": len(CLASS_NAMES),
        "classes": [
            {
                "id": i,
                "name": CLASS_NAMES[i],
                "color_rgb": CLASS_COLORS[i],
                "color_hex": '#{:02x}{:02x}{:02x}'.format(*CLASS_COLORS[i]),
            }
            for i in range(len(CLASS_NAMES))
        ],
        "declaration_mapping": DECLARATION_TO_MODEL_CLASS,
        "supported_declarations": sorted(DECLARATION_TO_MODEL_CLASS.keys()),
    }


# ══════════════════════════════════════════
# 5 — GET /api/v1/ndvi/{task_id}/timeseries
# ══════════════════════════════════════════
@app.get("/api/v1/ndvi/{task_id}/timeseries", tags=["NDVI"],
         summary="📈 Serie temporale NDVI (4 stagioni)")
async def get_ndvi_timeseries(task_id: str):
    """
    Ritorna l'NDVI calcolato per ciascuna delle 4 stagioni dell'analisi indicata.

    Il frontend può disegnare un **grafico a linea** con il trend del vigore
    vegetativo nell'anno:
    Inverno (basso) → Primavera (crescita) → Estate (picco) → Autunno (declino)

    Include anche una classificazione qualitativa del vigore:
    `ottimo` (≥0.7) · `buono` (≥0.5) · `moderato` (≥0.3) · `scarso` (<0.3)
    """
    if task_id not in results:
        if task_id in tasks and tasks[task_id]["status"] != "completed":
            raise HTTPException(202, "Analisi ancora in corso, riprova dopo il completamento")
        raise HTTPException(404, "Analisi non trovata")

    r = results[task_id]
    ndvi_seasons = r.get("ndvi_per_season", {})
    ndvi_mean = r.get("ndvi_mean", 0)

    labels = {"winter": "Gen-Feb", "spring": "Apr-Mag", "summer": "Lug-Ago", "autumn": "Ott-Nov"}
    timeseries = [
        {"season": s, "month": labels[s], "ndvi": ndvi_seasons[s]}
        for s in ["winter", "spring", "summer", "autumn"]
        if s in ndvi_seasons
    ]

    if ndvi_mean >= 0.7:   vigor = "ottimo"
    elif ndvi_mean >= 0.5: vigor = "buono"
    elif ndvi_mean >= 0.3: vigor = "moderato"
    else:                  vigor = "scarso"

    return {
        "task_id":        task_id,
        "region_name":    r.get("region_name"),
        "dominant_class": r.get("dominant_class"),
        "timeseries":     timeseries,
        "ndvi_mean":      ndvi_mean,
        "vigor_status":   vigor,
    }


# ══════════════════════════════════════════
# EXTRA — Health Check
# ══════════════════════════════════════════
@app.get("/api/v1/health", tags=["Metadata"], summary="💚 Stato sistema")
async def health():
    svc = ModelService()
    gpu = {}
    if torch.cuda.is_available():
        gpu = {
            "name":       torch.cuda.get_device_name(0),
            "vram_used":  round(torch.cuda.memory_allocated(0) / 1e9, 2),
            "vram_total": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2),
        }
    active = sum(1 for t in tasks.values() if t["status"] in ("queued","downloading","processing"))
    return {
        "status": "healthy" if svc.is_ready else "loading",
        "model_loaded": svc.is_ready,
        "device": str(svc.device) if svc.is_ready else "unknown",
        "gpu": gpu, "active_tasks": active, "completed_tasks": len(results),
    }


# ==========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)