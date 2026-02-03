import streamlit as st
import folium
from streamlit_folium import st_folium
import os
import json
from PIL import Image

# Importiamo la tua classe e le tue funzioni originali dal tuo file inference.py
from inference import SicilyInferencePipeline, get_bbox_from_point

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Dashboard Segmentazione Territoriale", layout="wide")

# Gestione dello stato per rendere il segnaposto (marker) dinamico
if 'lat' not in st.session_state:
    st.session_state.lat = 37.51015991536745
if 'lon' not in st.session_state:
    st.session_state.lon = 14.262956081139988

# Carichiamo la pipeline una sola volta
@st.cache_resource
def get_pipeline():
    return SicilyInferencePipeline(config_path='config.json')

pipeline = get_pipeline()

# --- INTERFACCIA UTENTE ---
st.title("🛰️ Dashboard Segmentazione Territoriale")

col_map, col_res = st.columns([1, 1.2])

with col_map:
    st.subheader("1. Selezione Area")
    # Mappa Satellitare Google Hybrid (Satellite + Etichette)
    m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=12)
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Il segnaposto segue le coordinate salvate
    folium.Marker([st.session_state.lat, st.session_state.lon], tooltip="Centro Analisi").add_to(m)
    
    # Visualizzazione mappa e cattura del click
    map_output = st_folium(m, width=600, height=500, key="map_input")

    # Se l'utente clicca sulla mappa, aggiorniamo le coordinate e ricarichiamo la pagina
    if map_output['last_clicked']:
        click_lat = map_output['last_clicked']['lat']
        click_lon = map_output['last_clicked']['lng']
        if click_lat != st.session_state.lat or click_lon != st.session_state.lon:
            st.session_state.lat = click_lat
            st.session_state.lon = click_lon
            st.rerun()

with col_res:
    st.subheader("2. Report e Statistiche")
    
    # Sidebar per i parametri
    st.sidebar.header("Impostazioni")
    st.session_state.lat = st.sidebar.number_input("Latitudine", value=st.session_state.lat, format="%.10f")
    st.session_state.lon = st.sidebar.number_input("Longitudine", value=st.session_state.lon, format="%.10f")
    area_size = st.sidebar.slider("Dimensione Area (BBox size)", 0.05, 0.40, 0.20, step=0.05)
    
    st.write(f"📍 Punto selezionato: `{st.session_state.lat:.6f}, {st.session_state.lon:.6f}`")
    
    if st.button("🚀 AVVIA SEGMENTAZIONE TERRITORIALE", use_container_width=True):
        # 1. Calcolo del BBox con la tua funzione originale
        bbox = get_bbox_from_point(st.session_state.lat, st.session_state.lon, size=area_size)
        
        # Nome del file di output (verrà salvato in tif/ e reports/)
        output_filename = "web_analysis_output.tif"
        base_name = "web_analysis_output"
        
        with st.spinner("⏳ Elaborazione in corso... Recupero dati Sentinel-2 e segmentazione AI"):
            try:
                # 2. Chiamata alla tua funzione originale (Senza progress_callback)
                pipeline.run_inference(bbox=bbox, output_name=output_filename, output_img=True)
                
                # 3. Visualizzazione dei risultati generati dal tuo codice
                path_panel = f"reports/{base_name}_panel.png"
                path_stats = f"reports/{base_name}_stats.json"

                # Mostra l'immagine a 3 pannelli (Satellite, Predizione, Overlay)
                if os.path.exists(path_panel):
                    st.success("✅ Analisi completata!")
                    st.image(Image.open(path_panel), caption="Report di Segmentazione", use_container_width=True)
                
                # Carica e mostra le statistiche salvate nel JSON in verticale
                if os.path.exists(path_stats):
                    with open(path_stats, 'r') as f:
                        stats = json.load(f)
                    
                    st.subheader("📊 Superfici rilevate")
                    # Loop per mostrare i dati uno sotto l'altro (Verticale)
                    for label, value in stats.items():
                        st.metric(label=label, value=f"{value:.2f} ha")
                    
                    # Grafico a barre per un riepilogo visivo
                    st.bar_chart(stats)

            except Exception as e:
                st.error(f"❌ Errore durante l'elaborazione: {e}")

    else:
        st.info("💡 Fai clic sulla mappa per spostare il segnaposto o inserisci le coordinate a sinistra.")