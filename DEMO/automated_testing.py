"""
🧪 SISTEMA DI TESTING AUTOMATICO
=================================
Valutazione completa del modello su dataset di test diversificato
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# Import sistema di inferenza
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from inference_point_COMPLETE import SicilyInferencePoint
from visual_inference import CLASS_COLORS, CLASS_NAMES

# ==========================================
# CONFIGURAZIONE TEST
# ==========================================

class TestSuite:
    """
    Suite completa di test geografici per validare il modello
    """
    
    # Dataset di test: coppie (lat, lon, tipo_atteso, descrizione)
    TEST_CASES = [
        # ===== AGRICOLTURA INTENSIVA =====
        {
            "name": "Agrumeti_Lentini",
            "coords": (37.285, 14.915),
            "expected_classes": ["Agrumi", "Frutteto"],
            "description": "Zona intensiva di agrumeti e frutteti temperati",
            "category": "agriculture_intensive"
        },
        {
            "name": "Serre_Pachino",
            "coords": (36.715, 15.010),
            "expected_classes": ["Ortaggi"],
            "description": "Serre intensive per pomodori e ortaggi",
            "category": "agriculture_intensive"
        },
        {
            "name": "Vigneti_Marsala",
            "coords": (37.810, 12.510),
            "expected_classes": ["Vite"],
            "description": "Zona vinicola storica di Marsala",
            "category": "agriculture_intensive"
        },
        
        # ===== COLTURE PERMANENTI =====
        {
            "name": "Oliveti_Ragusa",
            "coords": (36.926, 14.726),
            "expected_classes": ["Olivo"],
            "description": "Piantagioni di olivi secolari",
            "category": "permanent_crops"
        },
        {
            "name": "Oliveti_Modica",
            "coords": (36.856, 14.763),
            "expected_classes": ["Olivo", "Frutteto"],
            "description": "Mix olivi e mandorli",
            "category": "permanent_crops"
        },
        
        # ===== CEREALI =====
        {
            "name": "Grano_Enna",
            "coords": (37.567, 14.279),
            "expected_classes": ["Grano"],
            "description": "Altipiano cerealicolo ennese",
            "category": "cereals"
        },
        {
            "name": "Grano_Caltanissetta",
            "coords": (37.490, 14.043),
            "expected_classes": ["Grano", "Incolto"],
            "description": "Cereali con rotazione maggese",
            "category": "cereals"
        },
        
        # ===== ZONE COSTIERE (Mix) =====
        {
            "name": "Costa_Catania",
            "coords": (37.507, 15.083),
            "expected_classes": ["Agrumi", "Ortaggi"],
            "description": "Piana di Catania - mix irriguo",
            "category": "coastal_mixed"
        },
        {
            "name": "Costa_Siracusa",
            "coords": (37.075, 15.287),
            "expected_classes": ["Agrumi", "Olivo"],
            "description": "Fascia costiera sud-orientale",
            "category": "coastal_mixed"
        },
        
        # ===== ZONE MONTANE/MARGINALI =====
        {
            "name": "Etna_Est",
            "coords": (37.751, 15.004),
            "expected_classes": ["Vite", "Frutteto", "Incolto"],
            "description": "Pendici orientali Etna (vigneti e boschi)",
            "category": "mountain"
        },
        {
            "name": "Interno_Palermo",
            "coords": (37.912, 13.543),
            "expected_classes": ["Grano", "Olivo", "Incolto"],
            "description": "Entroterra collinare palermitano",
            "category": "mountain"
        },
        
        # ===== ZONE ACQUATICHE (Stress Test) =====
        {
            "name": "Stretto_Messina",
            "coords": (38.190, 15.562),
            "expected_classes": [],  # Dovrebbe essere filtrato
            "description": "Area marittima - test filtro acqua",
            "category": "water"
        },
        
        # ===== ZONE URBANE (Edge Case) =====
        {
            "name": "Catania_Urbano",
            "coords": (37.502, 15.087),
            "expected_classes": ["Incolto"],  # Aree verdi urbane
            "description": "Zona urbana - test robustezza",
            "category": "urban"
        },
    ]

# ==========================================
# FUNZIONI TESTING
# ==========================================

def run_single_test(pipeline, test_case):
    """
    Esegue un singolo test case
    
    Returns:
        dict: Risultati del test
    """
    name = test_case["name"]
    lat, lon = test_case["coords"]
    expected = test_case["expected_classes"]
    
    print(f"\n{'='*70}")
    print(f"🧪 TEST: {name}")
    print(f"📍 Coordinate: ({lat}, {lon})")
    print(f"🎯 Classi attese: {', '.join(expected) if expected else 'Nessuna (acqua)'}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        # Esegui inferenza
        pipeline.run_inference(lat, lon, region_name=f"test_{name}")
        
        elapsed = time.time() - start_time
        
        # Analizza risultati
        tif_path = f"inference_results/tif/test_{name}_map.tif"
        
        if not os.path.exists(tif_path):
            return {
                "status": "failed",
                "error": "GeoTIFF non generato",
                "time": elapsed
            }
        
        # Carica predizioni
        import rasterio
        with rasterio.open(tif_path) as src:
            mask = src.read(1)
        
        # Statistiche
        unique, counts = np.unique(mask, return_counts=True)
        
        detected_classes = []
        class_coverage = {}
        
        total_pixels = mask.size
        
        for cls_id, count in zip(unique, counts):
            if cls_id == 0:
                continue
            
            class_name = CLASS_NAMES[cls_id]
            coverage_pct = (count / total_pixels) * 100
            
            class_coverage[class_name] = {
                "pixels": int(count),
                "hectares": float((count * 100) / 10000),
                "coverage_pct": float(coverage_pct)
            }
            
            # Considera rilevata se copre almeno 1% dell'area
            if coverage_pct >= 1.0:
                detected_classes.append(class_name)
        
        # Verifica successo (se classi attese presenti)
        if expected:
            success = any(exp in detected_classes for exp in expected)
        else:
            # Caso acqua: successo se nessuna classe rilevata
            success = len(detected_classes) == 0
        
        return {
            "status": "success" if success else "mismatch",
            "detected_classes": detected_classes,
            "class_coverage": class_coverage,
            "expected_classes": expected,
            "time": elapsed,
            "success": success
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "time": time.time() - start_time
        }

def generate_test_report(results, output_dir="test_results"):
    """
    Genera report completo in formato HTML e JSON
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ===== REPORT JSON =====
    json_path = f"{output_dir}/test_report_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # ===== REPORT HTML =====
    html_path = f"{output_dir}/test_report_{timestamp}.html"
    
    # Statistiche aggregate
    total_tests = len(results)
    successful = sum(1 for r in results if r.get("success", False))
    failed = sum(1 for r in results if r["status"] == "error")
    mismatches = sum(1 for r in results if r["status"] == "mismatch")
    
    avg_time = np.mean([r["time"] for r in results])
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Report - Segmentazione Agricola</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                       color: white; padding: 30px; border-radius: 10px; }}
            .summary {{ background: white; padding: 20px; margin: 20px 0; 
                       border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .test-case {{ background: white; margin: 15px 0; padding: 20px; 
                         border-radius: 8px; border-left: 4px solid #667eea; }}
            .success {{ border-left-color: #10b981; }}
            .failed {{ border-left-color: #ef4444; }}
            .mismatch {{ border-left-color: #f59e0b; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background: #f8f9fa; font-weight: 600; }}
            .badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px; 
                     font-size: 12px; font-weight: 600; }}
            .badge-success {{ background: #d1fae5; color: #065f46; }}
            .badge-error {{ background: #fee2e2; color: #991b1b; }}
            .badge-warning {{ background: #fef3c7; color: #92400e; }}
            .chart {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧪 Report di Testing - Sistema di Segmentazione Agricola</h1>
            <p>Generato: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <div class="summary">
            <h2>📊 Sommario Esecuzione</h2>
            <table>
                <tr>
                    <th>Metrica</th>
                    <th>Valore</th>
                </tr>
                <tr>
                    <td><strong>Test Totali</strong></td>
                    <td>{total_tests}</td>
                </tr>
                <tr>
                    <td><strong>✅ Successi</strong></td>
                    <td><span class="badge badge-success">{successful} ({successful/total_tests*100:.1f}%)</span></td>
                </tr>
                <tr>
                    <td><strong>⚠️ Mismatch</strong></td>
                    <td><span class="badge badge-warning">{mismatches} ({mismatches/total_tests*100:.1f}%)</span></td>
                </tr>
                <tr>
                    <td><strong>❌ Errori</strong></td>
                    <td><span class="badge badge-error">{failed} ({failed/total_tests*100:.1f}%)</span></td>
                </tr>
                <tr>
                    <td><strong>⏱️ Tempo Medio</strong></td>
                    <td>{avg_time:.2f} secondi</td>
                </tr>
            </table>
        </div>
        
        <h2>📋 Dettaglio Test Cases</h2>
    """
    
    # Dettaglio per ogni test
    for i, result in enumerate(results):
        test_case = TestSuite.TEST_CASES[i]
        
        status_class = "success" if result.get("success") else "failed" if result["status"] == "error" else "mismatch"
        status_icon = "✅" if result.get("success") else "❌" if result["status"] == "error" else "⚠️"
        
        html_content += f"""
        <div class="test-case {status_class}">
            <h3>{status_icon} {test_case['name']}</h3>
            <p><strong>Categoria:</strong> {test_case['category']}</p>
            <p><strong>Descrizione:</strong> {test_case['description']}</p>
            <p><strong>Coordinate:</strong> ({test_case['coords'][0]}, {test_case['coords'][1]})</p>
            <p><strong>Tempo:</strong> {result['time']:.2f}s</p>
            
            <table>
                <tr>
                    <th>Classi Attese</th>
                    <th>Classi Rilevate</th>
                </tr>
                <tr>
                    <td>{', '.join(test_case['expected_classes']) if test_case['expected_classes'] else 'Nessuna'}</td>
                    <td>{', '.join(result.get('detected_classes', [])) if result.get('detected_classes') else 'Nessuna'}</td>
                </tr>
            </table>
        """
        
        if "class_coverage" in result:
            html_content += "<h4>Copertura Classi:</h4><table><tr><th>Classe</th><th>Ettari</th><th>Copertura %</th></tr>"
            for cls, stats in result["class_coverage"].items():
                html_content += f"<tr><td>{cls}</td><td>{stats['hectares']:.2f}</td><td>{stats['coverage_pct']:.1f}%</td></tr>"
            html_content += "</table>"
        
        if "error" in result:
            html_content += f"<p style='color: red;'><strong>Errore:</strong> {result['error']}</p>"
        
        html_content += "</div>"
    
    html_content += """
    </body>
    </html>
    """
    
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    return json_path, html_path

def plot_test_summary(results, output_path="test_results/summary_charts.png"):
    """
    Genera grafici riassuntivi
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("📊 Analisi Risultati Testing", fontsize=16, fontweight='bold')
    
    # 1. Success Rate per Categoria
    categories = defaultdict(lambda: {"total": 0, "success": 0})
    for i, result in enumerate(results):
        cat = TestSuite.TEST_CASES[i]["category"]
        categories[cat]["total"] += 1
        if result.get("success"):
            categories[cat]["success"] += 1
    
    cat_names = list(categories.keys())
    success_rates = [categories[c]["success"]/categories[c]["total"]*100 for c in cat_names]
    
    axes[0, 0].barh(cat_names, success_rates, color='#667eea')
    axes[0, 0].set_xlabel("Success Rate (%)")
    axes[0, 0].set_title("Success Rate per Categoria")
    axes[0, 0].set_xlim(0, 100)
    
    # 2. Distribuzione Tempi
    times = [r["time"] for r in results]
    axes[0, 1].hist(times, bins=10, color='#764ba2', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.mean(times), color='red', linestyle='--', label=f'Media: {np.mean(times):.2f}s')
    axes[0, 1].set_xlabel("Tempo (s)")
    axes[0, 1].set_ylabel("Frequenza")
    axes[0, 1].set_title("Distribuzione Tempi di Inferenza")
    axes[0, 1].legend()
    
    # 3. Classi Più Rilevate
    class_counts = defaultdict(int)
    for result in results:
        for cls in result.get("detected_classes", []):
            class_counts[cls] += 1
    
    if class_counts:
        top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        classes, counts = zip(*top_classes)
        
        axes[1, 0].bar(classes, counts, color='#10b981', alpha=0.7)
        axes[1, 0].set_xlabel("Classe")
        axes[1, 0].set_ylabel("Numero di Rilevamenti")
        axes[1, 0].set_title("Classi Più Frequentemente Rilevate")
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Status Distribution
    status_counts = defaultdict(int)
    for r in results:
        status_counts[r["status"]] += 1
    
    labels = list(status_counts.keys())
    sizes = list(status_counts.values())
    colors = {'success': '#10b981', 'mismatch': '#f59e0b', 'error': '#ef4444'}
    pie_colors = [colors.get(l, '#gray') for l in labels]
    
    axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%', colors=pie_colors, startangle=90)
    axes[1, 1].set_title("Distribuzione Status Test")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Grafici salvati: {output_path}")

# ==========================================
# MAIN TESTING
# ==========================================

def run_full_test_suite():
    """
    Esegue l'intera suite di test
    """
    print("\n" + "="*70)
    print("🧪 AVVIO TESTING AUTOMATICO")
    print("="*70)
    print(f"\n📋 Test da eseguire: {len(TestSuite.TEST_CASES)}")
    print(f"⏱️  Tempo stimato: ~{len(TestSuite.TEST_CASES) * 30 / 60:.1f} minuti\n")
    
    # Inizializza pipeline
    print("🔄 Caricamento modello...")
    pipeline = SicilyInferencePoint(config_path='../config.json')
    print("✅ Modello pronto!\n")
    
    # Esegui tutti i test
    results = []
    
    for i, test_case in enumerate(TestSuite.TEST_CASES):
        print(f"\n[{i+1}/{len(TestSuite.TEST_CASES)}]", end=" ")
        result = run_single_test(pipeline, test_case)
        results.append(result)
        
        # Status immediato
        if result.get("success"):
            print("✅ TEST PASSED")
        elif result["status"] == "error":
            print(f"❌ TEST FAILED: {result.get('error', 'Unknown')}")
        else:
            print("⚠️  TEST MISMATCH (classi rilevate diverse da attese)")
    
    # Genera report
    print("\n" + "="*70)
    print("📊 GENERAZIONE REPORT...")
    print("="*70 + "\n")
    
    json_path, html_path = generate_test_report(results)
    print(f"✅ Report JSON: {json_path}")
    print(f"✅ Report HTML: {html_path}")
    
    # Genera grafici
    plot_test_summary(results)
    
    # Riepilogo finale
    successful = sum(1 for r in results if r.get("success", False))
    print("\n" + "="*70)
    print("🎯 RIEPILOGO FINALE")
    print("="*70)
    print(f"✅ Test Passati: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    print(f"📁 Risultati salvati in: test_results/")
    print("\n🌍 Apri il report HTML per visualizzazione completa!")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_full_test_suite()