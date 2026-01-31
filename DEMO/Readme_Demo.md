# 🌍 Demo & Testing - Sistema Segmentazione Agricola Sicilia

## 🚀 Quick Start (3 Passi)

### **1. Installa Dipendenze**

```bash
pip install gradio pillow --break-system-packages
```

### **2. Avvia il Launcher**

```bash
python launcher.py
```

### **3. Scegli Modalità**

- **1️⃣ Demo Interattiva** → Interfaccia web per query singole
- **2️⃣ Testing Automatico** → Validazione su 13 scenari

---

## 📁 File Principali

```
.
├── launcher.py                    # 🚀 Start here!
├── demo_gui.py                    # Interfaccia Gradio
├── automated_testing.py           # Suite di test completa
├── inference_point_COMPLETE.py    # Engine di inferenza
├── download_minio_IMPROVED.py     # Download preventivo (opzionale)
└── GUIDA_DEMO_TESTING.md         # Documentazione completa
```

---

## 🖥️ Demo Interattiva

### **Cosa fa:**
- Interfaccia web user-friendly
- 7 punti predefiniti (agrumi, vigneti, serre, etc.)
- Query personalizzate su coordinate
- Visualizzazione 3-panel (RGB + Segmentazione + Overlay)
- Statistiche aree in ettari

### **Come usarla:**

```bash
python demo_gui.py
# Apri: http://localhost:7860
```

**Screenshot UI:**
```
┌─────────────────────────────────────────────┐
│  📍 Query Geografica                        │
│  ┌─────────────────────────────┐            │
│  │ 🎯 Vigneti Marsala          │ ▼          │
│  └─────────────────────────────┘            │
│                                              │
│  Latitudine:  37.810                        │
│  Longitudine: 12.510                        │
│  Nome: Vigneti_Marsala                      │
│                                              │
│  🚀 [Avvia Analisi]                         │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  📊 Risultati                               │
│  ┌───────────┬───────────┬───────────┐      │
│  │ RGB       │ Predizione│ Overlay   │      │
│  │ [Satellite│ [Segmented│ [Combined]│      │
│  │  Image]   │   Map]    │   View]   │      │
│  └───────────┴───────────┴───────────┘      │
│                                              │
│  📈 Statistiche:                            │
│  🟢 Vite: 45.2 ha (68.5%)                   │
│  🟢 Olivo: 12.1 ha (18.3%)                  │
│  🟢 Incolto: 8.7 ha (13.2%)                 │
└─────────────────────────────────────────────┘
```

---

## 🧪 Testing Automatico

### **Cosa fa:**
- 13 test su scenari diversificati:
  - ✅ Agrumeti intensivi
  - ✅ Vigneti DOC
  - ✅ Oliveti secolari
  - ✅ Serre orticole
  - ✅ Cerealicoltura
  - ✅ Zone montane
  - ✅ Stress test (mare, urbano)
- Genera report HTML con metriche
- Grafici riassuntivi

### **Come eseguirlo:**

```bash
python automated_testing.py
# Durata: ~10-15 minuti
```

### **Output:**

```
test_results/
├── test_report_20260131_143025.html  # 📄 Report visuale
├── test_report_20260131_143025.json  # 📊 Dati strutturati
├── summary_charts.png                # 📈 Grafici
└── inference_results/                # 🗺️ Mappe per ogni test
    ├── reports/ (13 PNG)
    └── tif/ (13 GeoTIFF)
```

### **Metriche Chiave (Report HTML):**

| Metrica | Target | Significato |
|---------|--------|-------------|
| Success Rate | >80% | % test con classi attese rilevate |
| Tempo Medio | <60s | Performance inferenza |
| Mismatch | <15% | Classi diverse da attese (accettabile) |
| Errori | <5% | Crash / Download fallito |

---

## 📊 Scenari di Test

### **Categoria: Agriculture Intensive**
1. **Agrumeti Lentini** → Agrumi + Frutteti
2. **Serre Pachino** → Ortaggi intensivi
3. **Vigneti Marsala** → Viticoltura DOC

### **Categoria: Permanent Crops**
4. **Oliveti Ragusa** → Olivi secolari
5. **Oliveti Modica** → Mix olivi/mandorli

### **Categoria: Cereals**
6. **Grano Enna** → Cerealicoltura
7. **Grano Caltanissetta** → Rotazione maggese

### **Categoria: Coastal Mixed**
8. **Costa Catania** → Piana irrigua
9. **Costa Siracusa** → Mix agrumi/olivi

### **Categoria: Mountain**
10. **Etna Est** → Vigneti vulcanici
11. **Interno Palermo** → Colline

### **Categoria: Stress Test**
12. **Stretto Messina** → Mare (deve essere filtrato)
13. **Catania Urbano** → Città (edge case)

---

## 🎯 Use Cases

### **Per Demo a Stakeholder Non Tecnici**

1. Usa la **Demo GUI**
2. Mostra 3-4 casi "wow":
   - 🍷 **Vigneti Marsala** (mono-coltura pulita)
   - 🍅 **Serre Pachino** (tecnologia intensiva)
   - 🍊 **Agrumeti Catania** (diversità colturale)
3. Evidenzia:
   - ✅ **Velocità**: 30-60s vs giorni di survey
   - ✅ **Precisione**: 10m/pixel (1 ettaro visibile)
   - ✅ **Copertura**: Tutta la Sicilia

### **Per Validazione Tecnica**

1. Esegui il **Testing Automatico**
2. Analizza il **Report HTML**:
   - Success rate per categoria
   - Confusion tra classi simili (es: Frutteto/Agrumi)
   - Performance temporale
3. Verifica **Grafici Riassuntivi**:
   - Bias verso certe classi?
   - Outlier di tempo?

---

## ⚙️ Configurazione Avanzata

### **Aumentare Area di Analisi**

In `inference_point_COMPLETE.py` linea 179:
```python
bbox_download = self.get_bbox_from_point(lat, lon, size=0.15)  # Da 0.1 (11km) a 0.15 (16km)
```

### **Cambiare Soglia Filtro Mare**

In `inference_point_COMPLETE.py` linea 278:
```python
water_mask = nir_summer < 500  # Da 400 a 500 (più conservativo)
```

### **Disabilitare Cache MinIO (Solo STAC)**

In `demo_gui.py` o `automated_testing.py`, commenta:
```python
# key, bbox_tile = self.find_tile_containing_point(lat, lon)
key, bbox_tile = None, None  # Forza download on-the-fly
```

---

## 🔧 Troubleshooting

### **"Modello non caricato"**
→ Verifica che `config.json` e `best_model.pth` siano accessibili

### **"Download fallito"**
→ Problemi di rete o area senza copertura Sentinel-2

### **Demo lenta/freeze**
→ Normale per download on-the-fly (60s). Usa pre-cache MinIO.

### **Mare ancora colorato**
→ Aumenta soglia filtro NIR (vedi sopra)

---

## 📖 Documentazione Completa

Per dettagli su interpretazione risultati, metriche, best practices:

```bash
# Apri guida completa
cat GUIDA_DEMO_TESTING.md
# O usa launcher opzione 4
```

---

## 🌟 Highlights

- **Zero codice visibile** → Perfetto per demo non tecniche
- **Testing automatico** → Validazione oggettiva
- **Report HTML professionali** → Pronto per presentazioni
- **Cache intelligente** → Prestazioni ottimali
- **13 scenari diversificati** → Coverage completo

---

**Sviluppato con:** PyTorch • Prithvi • Sentinel-2 • Gradio  
**Durata demo singola:** 30-60 secondi  
**Durata testing completo:** 10-15 minuti  

🚀 **Buona demo!**