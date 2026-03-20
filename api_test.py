"""
🧪 TEST COMPLETO - Tutti e 7 gli Endpoint
==========================================
python test_api.py --mock         # Test logica offline
python test_api.py --live         # Test su server reale
python test_api.py --mock --live  # Entrambi
"""

import argparse
import json
import time
import sys
import numpy as np
from datetime import datetime

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}✅ PASS{RESET} - {msg}")
def fail(msg): print(f"  {RED}❌ FAIL{RESET} - {msg}")
def info(msg): print(f"  {CYAN}ℹ️  {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}⚠️  {msg}{RESET}")

P = 0  # passed
F = 0  # failed

def check(condition, msg_ok, msg_fail=""):
    global P, F
    if condition:
        ok(msg_ok); P += 1
    else:
        fail(msg_fail or msg_ok); F += 1


# =============================================
# MOCK TESTS
# =============================================
def run_mock_tests():
    global P, F
    P, F = 0, 0
    print(f"\n{BOLD}{'='*65}")
    print(f"  🧪 MOCK TESTS — Logica di calcolo (no server, no GPU)")
    print(f"{'='*65}{RESET}\n")

    sys.path.insert(0, '.')

    # --- Test NDVI ---
    print(f"{BOLD}Test 1: NDVI{RESET}")
    from api import calculate_ndvi
    cube = np.random.uniform(200, 3000, (4, 6, 10, 10)).astype(np.float32)
    cube[2, 2, :, :] = 1000  # Red
    cube[2, 3, :, :] = 3000  # NIR → NDVI = 0.5
    ndvi = calculate_ndvi(cube)
    check(abs(ndvi["ndvi_per_season"]["summer"] - 0.5) < 0.01,
          f"NDVI estate = {ndvi['ndvi_per_season']['summer']} (atteso ~0.5)")
    check(len(ndvi["ndvi_per_season"]) == 4, "4 stagioni presenti")
    check(all(-1 <= v <= 1 for v in ndvi["ndvi_per_season"].values()), "Valori in range [-1,1]")

    # --- Test NDVI con maschera ---
    print(f"\n{BOLD}Test 2: NDVI per appezzamento (maschera){RESET}")
    from api import calculate_ndvi_for_mask
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    ndvi_masked = calculate_ndvi_for_mask(cube, mask)
    check("ndvi_mean" in ndvi_masked, f"NDVI masked = {ndvi_masked['ndvi_mean']}")
    check(abs(ndvi_masked["ndvi_per_season"]["summer"] - 0.5) < 0.01, "NDVI masked coerente")

    # --- Test Distribuzione classi ---
    print(f"\n{BOLD}Test 3: Distribuzione classi + Confidence{RESET}")
    from api import calculate_class_distribution, CLASS_NAMES
    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[:60, :] = 5   # Grano
    pred[60:85, :] = 1  # Olivox
    pred[85:, :] = 8   # Incolto
    conf = np.random.uniform(0.8, 0.95, (100, 100)).astype(np.float32)
    dist, conf_cls, dominant = calculate_class_distribution(pred, conf, 10000)
    check(dominant == "Grano", f"Dominante = '{dominant}'")
    grano = next((d for d in dist if d["class_name"] == "Grano"), None)
    check(grano and abs(grano["percentage"] - 60) < 1, f"Grano ~60%: {grano['percentage']}%")
    check(abs(sum(d["percentage"] for d in dist) - 100) < 1, "Somma ~100%")
    check("Grano" in conf_cls and conf_cls["Grano"] > 70, f"Confidence Grano: {conf_cls.get('Grano')}%")

    # --- Test Mapping dichiarazioni ---
    print(f"\n{BOLD}Test 4: Mapping dichiarazioni → classi modello{RESET}")
    from api import DECLARATION_TO_MODEL_CLASS
    check(DECLARATION_TO_MODEL_CLASS["Mais"] == "Grano", "Mais → Grano")
    check(DECLARATION_TO_MODEL_CLASS["Soia"] == "Legumi", "Soia → Legumi")
    check(DECLARATION_TO_MODEL_CLASS["Arancio"] == "Agrumi", "Arancio → Agrumi")
    check(DECLARATION_TO_MODEL_CLASS["Pistacchio"] == "Frutteto", "Pistacchio → Frutteto")

    # --- Test Geometry to pixel mask ---
    print(f"\n{BOLD}Test 5: Conversione GeoJSON → pixel mask{RESET}")
    from api import geometry_to_pixel_mask
    geo = {"type": "Polygon", "coordinates": [[[14.0, 37.0], [14.5, 37.0], [14.5, 37.5], [14.0, 37.5], [14.0, 37.0]]]}
    bbox = [14.0, 37.0, 14.5, 37.5]
    mask = geometry_to_pixel_mask(geo, bbox, (100, 100))
    check(mask.shape == (100, 100), f"Shape maschera: {mask.shape}")
    check(mask.sum() > 0, f"Pixel attivi: {mask.sum()}")
    # Il poligono copre tutta l'area → ~100% dei pixel
    check(mask.sum() > 9000, f"Copertura ~totale: {mask.sum()}/10000")

    # --- Test Pydantic models ---
    print(f"\n{BOLD}Test 6: Validazione Pydantic{RESET}")
    from api import AnalysisRequest, PointInput, BBoxInput, DeclarationUpdate
    from pydantic import ValidationError
    req = AnalysisRequest(point=PointInput(lat=37.38, lon=14.91), user_id="g.agricola")
    check(req.point.lat == 37.38 and req.user_id == "g.agricola", "Request con user_id")
    req2 = AnalysisRequest(bbox=BBoxInput(min_lon=12.4, min_lat=37.0, max_lon=12.6, max_lat=37.2))
    check(req2.bbox is not None, "Request con bbox")
    try:
        PointInput(lat=95, lon=14)
        check(False, "", "Lat 95 non rifiutata")
    except ValidationError:
        check(True, "Lat 95 correttamente rifiutata")
    upd = DeclarationUpdate(declared_crop="Soia")
    check(upd.declared_crop == "Soia", "DeclarationUpdate valida")

    # --- Test Verifica colturale ---
    print(f"\n{BOLD}Test 7: Logica verifica dichiarato vs rilevato{RESET}")
    from api import verify_plot_against_prediction
    plot = {
        "id": "test-001", "user_id": "test", "name": "Test Plot",
        "declared_crop": "Mais", "hectares": 10,
        "geometry": {"type": "Polygon", "coordinates": [[[14.0,37.0],[14.5,37.0],[14.5,37.5],[14.0,37.5],[14.0,37.0]]]},
    }
    pred_mask = np.full((100, 100), 5, dtype=np.uint8)   # Tutto Grano (classe 5)
    conf_map = np.full((100, 100), 0.92, dtype=np.float32)
    cube_test = np.random.uniform(200, 3000, (4, 6, 100, 100)).astype(np.float32)
    cube_test[:, 3, :, :] = 3000  # NIR alto
    cube_test[:, 2, :, :] = 1000  # Red
    area_bbox = [14.0, 37.0, 14.5, 37.5]
    result = verify_plot_against_prediction(plot, pred_mask, conf_map, cube_test, area_bbox)
    # Mais → mappato a "Grano", rilevato = "Grano" → verificato
    check(result.stato == "verificato", f"Mais dichiarato, Grano rilevato → stato='{result.stato}' (atteso: verificato)")
    check(result.rilevato == "Grano", f"Rilevato: '{result.rilevato}'")
    check(result.confidence > 80, f"Confidence: {result.confidence}%")
    check(result.ndvi is not None and result.ndvi > 0, f"NDVI: {result.ndvi}")
    check(result.distribuzione_classi is not None, "Distribuzione classi presente")

    # Ora testa una discordanza
    plot2 = {**plot, "declared_crop": "Vite"}
    result2 = verify_plot_against_prediction(plot2, pred_mask, conf_map, cube_test, area_bbox)
    check(result2.stato == "discordanza", f"Vite dichiarata, Grano rilevato → stato='{result2.stato}' (atteso: discordanza)")

    # --- Riepilogo ---
    print(f"\n{BOLD}{'='*65}")
    print(f"  MOCK: {P}/{P+F} passati", end="")
    if F == 0: print(f"  {GREEN}🎉 TUTTI OK{RESET}")
    else: print(f"  {RED}⚠️ {F} falliti{RESET}")
    print(f"{'='*65}{RESET}")
    return F == 0


# =============================================
# LIVE TESTS
# =============================================
def run_live_tests(base_url="http://localhost:8000"):
    global P, F
    P, F = 0, 0
    import requests

    print(f"\n{BOLD}{'='*65}")
    print(f"  🌐 LIVE TESTS — Server: {base_url}")
    print(f"{'='*65}{RESET}\n")

    # L1: Health
    print(f"{BOLD}L1: Health Check{RESET}")
    try:
        r = requests.get(f"{base_url}/api/v1/health", timeout=5)
        check(r.status_code == 200, f"Server OK — modello: {r.json().get('model_loaded')}")
    except requests.ConnectionError:
        fail(f"Server non raggiungibile"); warn("Avvia: uvicorn api:app --port 8000"); return False

    # L2: GET /classes (Endpoint 6)
    print(f"\n{BOLD}L2: Endpoint 6 — GET /classes{RESET}")
    r = requests.get(f"{base_url}/api/v1/classes")
    data = r.json()
    check(r.status_code == 200 and data["num_classes"] == 9, f"9 classi ricevute")
    check("declaration_mapping" in data, "Mapping dichiarazioni presente")
    check(data["declaration_mapping"].get("Mais") == "Grano", "Mapping Mais→Grano confermato")
    info(f"Dichiarazioni supportate: {len(data.get('supported_declarations', []))}")

    # L3: GET /plots (Endpoint 4)
    print(f"\n{BOLD}L3: Endpoint 4 — GET /plots/g.agricola{RESET}")
    r = requests.get(f"{base_url}/api/v1/plots/g.agricola")
    check(r.status_code == 200, "Appezzamenti utente trovati")
    plots = r.json()
    check(plots["count"] == 3, f"3 appezzamenti demo: {plots['count']}")
    for p in plots["plots"][:2]:
        info(f"  {p['name']}: {p['declared_crop']} ({p['hectares']} ha) — stato: {p['status']}")

    # L4: POST /analysis/run (Endpoint 1) — ASINCRONO
    print(f"\n{BOLD}L4: Endpoint 1 — POST /analysis/run (asincrono){RESET}")
    payload = {
        "point": {"lat": 37.380, "lon": 14.910, "region_name": "Test_API_Catania"},
        "user_id": "g.agricola"
    }
    r = requests.post(f"{base_url}/api/v1/analysis/run", json=payload, timeout=10)
    check(r.status_code == 200, "Task creato")
    task_id = r.json()["task_id"]
    check(r.json()["status"] == "queued", f"Status iniziale: queued (task: {task_id})")

    # L5: Polling status (Endpoint 2)
    print(f"\n{BOLD}L5: Endpoint 2 — Polling /analysis/{task_id}/status{RESET}")
    info("⏳ Attendo completamento (max 5 minuti)...")
    start = time.time()
    final_status = None
    while time.time() - start < 300:
        r = requests.get(f"{base_url}/api/v1/analysis/{task_id}/status")
        if r.status_code == 200:
            s = r.json()
            status = s["status"]
            progress = s.get("progress", 0)
            elapsed = int(time.time() - start)
            print(f"\r    [{elapsed}s] Status: {status} | Progress: {progress}%    ", end="", flush=True)
            if status == "completed":
                final_status = s
                break
            elif status == "failed":
                print()
                fail(f"Task fallito: {s.get('error')}")
                break
        time.sleep(5)
    print()

    if final_status and final_status["status"] == "completed":
        elapsed = int(time.time() - start)
        check(True, f"Analisi completata in {elapsed}s")

        result = final_status.get("result", {})

        # Verifica struttura risposta
        check("summary" in result or "class_distribution" in result, "Struttura response corretta")
        check("images" in result and result["images"], "URLs immagini presenti")
        info(f"Classe dominante: {result.get('dominant_class')}")
        info(f"Confidence media: {result.get('confidence_mean')}%")
        info(f"NDVI medio: {result.get('ndvi_mean')}")
        info(f"Ettari totali: {result.get('total_hectares')} ha")

        if result.get("appezzamenti"):
            check(True, f"{len(result['appezzamenti'])} appezzamenti analizzati")
            for app in result["appezzamenti"]:
                stato_icon = "✅" if app["stato"] == "verificato" else "⚠️" if app["stato"] == "discordanza" else "❓"
                info(f"  {stato_icon} {app['nome']}: dichiarato={app['dichiarato']}, rilevato={app.get('rilevato')}, conf={app.get('confidence')}%, NDVI={app.get('ndvi')}")

        if result.get("summary"):
            s = result["summary"]
            info(f"  📊 Verificate: {s['colture_verificate']} | Discordanze: {s['discordanze']} | Conf media: {s['confidence_media']}%")

        # L6: Immagini (Endpoint 3)
        print(f"\n{BOLD}L6: Endpoint 3 — Download immagini{RESET}")
        for img_type in ["rgb", "segmentation", "overlay", "ndvi", "confidence"]:
            r = requests.get(f"{base_url}/api/v1/analysis/{task_id}/images/{img_type}")
            check(r.status_code == 200 and 'image/png' in r.headers.get('content-type', ''),
                  f"Immagine '{img_type}': {len(r.content)//1024} KB")

    else:
        fail("Analisi non completata entro timeout")

    # L7: PUT declaration (Endpoint 5)
    print(f"\n{BOLD}L7: Endpoint 5 — PUT /plots/plot-est-003/declaration{RESET}")
    r = requests.put(
        f"{base_url}/api/v1/plots/plot-est-003/declaration",
        json={"declared_crop": "Soia"}
    )
    check(r.status_code == 200, "Dichiarazione aggiornata")
    if r.status_code == 200:
        data = r.json()
        info(f"  Vecchia: {data.get('old_declared')} → Nuova: {data.get('new_declared')}")
        info(f"  Mappata a: {data.get('mapped_to_model_class')}")
        info(f"  Nuovo stato: {data.get('new_status')}")

    # L8: NDVI Timeseries (Endpoint 7)
    print(f"\n{BOLD}L8: Endpoint 7 — GET /ndvi/plot-nord-001/timeseries{RESET}")
    r = requests.get(f"{base_url}/api/v1/ndvi/plot-nord-001/timeseries")
    if r.status_code == 200:
        data = r.json()
        check(True, f"NDVI timeseries ricevuta")
        info(f"  Vigore: {data.get('vigor_status')}")
        for ts in data.get("timeseries", []):
            bar = "█" * int(ts["ndvi"] * 20)
            info(f"  {ts['month']:<10} NDVI={ts['ndvi']:.3f} {bar}")
    elif r.status_code == 404:
        warn("Appezzamento non ancora analizzato (atteso se l'analisi non copriva quell'area)")
        P += 1  # OK comunque
    else:
        fail(f"Errore: {r.status_code}")

    # L9: Errori
    print(f"\n{BOLD}L9: Gestione errori{RESET}")
    r = requests.post(f"{base_url}/api/v1/analysis/run", json={})
    check(r.status_code in [400, 422], f"Request vuota → {r.status_code}")
    r = requests.get(f"{base_url}/api/v1/analysis/nonexistent/status")
    check(r.status_code == 404, "Task inesistente → 404")
    r = requests.get(f"{base_url}/api/v1/plots/utente_che_non_esiste")
    check(r.status_code == 404, "Utente senza plot → 404")
    r = requests.put(f"{base_url}/api/v1/plots/plot-inesistente/declaration", json={"declared_crop": "Mais"})
    check(r.status_code == 404, "Plot inesistente → 404")

    # Riepilogo
    print(f"\n{BOLD}{'='*65}")
    print(f"  LIVE: {P}/{P+F} passati", end="")
    if F == 0: print(f"  {GREEN}🎉 TUTTI OK{RESET}")
    else: print(f"  {RED}⚠️ {F} falliti{RESET}")
    print(f"{'='*65}{RESET}")
    return F == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    if not args.mock and not args.live:
        print("Usa: python test_api.py --mock | --live | --mock --live")
        sys.exit(1)
    success = True
    if args.mock:  success = run_mock_tests() and success
    if args.live:  success = run_live_tests(args.url) and success
    sys.exit(0 if success else 1)