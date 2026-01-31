#!/usr/bin/env python3
"""
🚀 LAUNCHER - Demo & Testing Sistema Segmentazione Agricola
============================================================
Script interattivo per avviare demo o testing
"""

import os
import sys
import subprocess

def print_banner():
    """Stampa banner iniziale"""
    print("\n" + "="*70)
    print("  🌍 SISTEMA SEGMENTAZIONE AGRICOLA SICILIA")
    print("  Deep Learning per Analisi Territoriale")
    print("="*70 + "\n")

def check_dependencies():
    """Verifica dipendenze necessarie"""
    print("🔍 Verifica dipendenze...")
    
    required_packages = {
        'gradio': 'Interfaccia web',
        'torch': 'Deep Learning',
        'rasterio': 'Geospatial',
        'boto3': 'MinIO client',
        'matplotlib': 'Visualizzazione'
    }
    
    missing = []
    
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {package:<15} ({description})")
        except ImportError:
            print(f"  ❌ {package:<15} MANCANTE")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Dipendenze mancanti: {', '.join(missing)}")
        print("\nInstalla con:")
        print(f"pip install {' '.join(missing)} --break-system-packages\n")
        return False
    
    print("\n✅ Tutte le dipendenze sono presenti!\n")
    return True

def check_model():
    """Verifica presenza modello e config"""
    print("🔍 Verifica configurazione modello...")
    
    config_paths = ['../config.json', 'config.json']
    config_found = any(os.path.exists(p) for p in config_paths)
    
    model_paths = ['../prithvi_4090_best.pth', 'models/prithvi_4090_best.pth', 'prithvi_4090_best.pth']
    model_found = any(os.path.exists(p) for p in model_paths)
    
    if config_found:
        print("  ✅ config.json trovato")
    else:
        print("  ❌ config.json NON trovato")
    
    if model_found:
        print("  ✅ best_model.pth trovato")
    else:
        print("  ❌ best_model.pth NON trovato")
    
    if not (config_found and model_found):
        print("\n⚠️  ATTENZIONE: File essenziali mancanti!")
        print("   Assicurati che config.json e best_model.pth siano accessibili.\n")
        return False
    
    print("\n✅ Configurazione valida!\n")
    return True

def check_minio():
    """Verifica se MinIO è raggiungibile"""
    print("🔍 Verifica MinIO...")
    
    try:
        import boto3
        s3_client = boto3.client(
            's3',
            endpoint_url='http://localhost:9000',
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin'
        )
        
        # Prova a listare i bucket
        s3_client.list_buckets()
        print("  ✅ MinIO raggiungibile (cache attiva)\n")
        return True
        
    except Exception as e:
        print(f"  ⚠️  MinIO non raggiungibile: {e}")
        print("  → Il sistema userà download on-the-fly (più lento)\n")
        return False

def show_menu():
    """Mostra menu principale"""
    print("="*70)
    print("  COSA VUOI FARE?")
    print("="*70)
    print("\n1️⃣  🖥️  Avvia DEMO INTERATTIVA (Interfaccia Web)")
    print("     → Perfetto per presentazioni e query singole")
    print("     → Interfaccia grafica user-friendly")
    print()
    print("2️⃣  🧪 Esegui TESTING AUTOMATICO (13 scenari)")
    print("     → Validazione completa del modello")
    print("     → Genera report HTML con metriche")
    print("     → Durata: ~10-15 minuti")
    print()
    print("3️⃣  📥 Pre-Popola CACHE MinIO (Download preventivo)")
    print("     → Scarica dati per tutta la Sicilia")
    print("     → Query future saranno istantanee")
    print("     → Durata: ~2-4 ore, esegui overnight")
    print()
    print("4️⃣  📖 Apri GUIDA COMPLETA")
    print()
    print("5️⃣  ❌ Esci")
    print()
    print("="*70)

def launch_demo():
    """Avvia demo GUI"""
    print("\n🚀 Avvio Demo Interattiva...")
    print("-" * 70)
    print("L'interfaccia si aprirà su: http://localhost:7860")
    print("Premi CTRL+C per terminare")
    print("-" * 70 + "\n")
    
    try:
        subprocess.run([sys.executable, "demo_gui.py"])
    except KeyboardInterrupt:
        print("\n\n✅ Demo terminata.")

def launch_testing():
    """Avvia testing automatico"""
    print("\n🧪 Avvio Testing Automatico...")
    print("-" * 70)
    print("Saranno eseguiti 13 test su scenari diversificati")
    print("I risultati saranno salvati in test_results/")
    print("-" * 70 + "\n")
    
    proceed = input("⚠️  ATTENZIONE: Il testing richiede 10-15 minuti. Procedere? (y/n): ")
    
    if proceed.lower() in ['y', 'yes', 's', 'si', 'sì']:
        try:
            subprocess.run([sys.executable, "automated_testing.py"])
        except KeyboardInterrupt:
            print("\n\n⚠️ Testing interrotto.")
    else:
        print("❌ Testing annullato.")

def launch_downloader():
    """Avvia downloader MinIO"""
    print("\n📥 Avvio Download Preventivo...")
    print("-" * 70)
    print("Questo processo scaricherà dati per tutta la Sicilia")
    print("Tempo stimato: 2-4 ore")
    print("Spazio richiesto: ~50-100 GB")
    print("-" * 70 + "\n")
    
    proceed = input("⚠️  Procedere con il download? (y/n): ")
    
    if proceed.lower() in ['y', 'yes', 's', 'si', 'sì']:
        try:
            subprocess.run([sys.executable, "download_minio_IMPROVED.py"])
        except KeyboardInterrupt:
            print("\n\n⚠️ Download interrotto (parziale salvato).")
    else:
        print("❌ Download annullato.")

def open_guide():
    """Apri guida completa"""
    print("\n📖 Apertura Guida Completa...")
    
    guide_path = "GUIDA_DEMO_TESTING.md"
    
    if os.path.exists(guide_path):
        # Prova ad aprire con il visualizzatore markdown predefinito
        try:
            if sys.platform == 'darwin':  # macOS
                subprocess.run(['open', guide_path])
            elif sys.platform == 'win32':  # Windows
                os.startfile(guide_path)
            else:  # Linux
                subprocess.run(['xdg-open', guide_path])
            print("✅ Guida aperta nel visualizzatore predefinito.")
        except:
            print(f"📄 Guida disponibile in: {os.path.abspath(guide_path)}")
            print("   Aprila manualmente con un editor markdown.")
    else:
        print("❌ Guida non trovata. Assicurati che GUIDA_DEMO_TESTING.md sia nella directory corrente.")

def main():
    """Funzione principale"""
    print_banner()
    
    # Verifiche preliminari
    if not check_dependencies():
        sys.exit(1)
    
    if not check_model():
        print("⚠️  Puoi comunque continuare, ma alcune funzionalità potrebbero non funzionare.\n")
        proceed = input("Continuare comunque? (y/n): ")
        if proceed.lower() not in ['y', 'yes', 's', 'si', 'sì']:
            sys.exit(1)
    
    check_minio()
    
    # Loop menu principale
    while True:
        show_menu()
        
        choice = input("Inserisci scelta (1-5): ").strip()
        
        if choice == '1':
            launch_demo()
        elif choice == '2':
            launch_testing()
        elif choice == '3':
            launch_downloader()
        elif choice == '4':
            open_guide()
        elif choice == '5':
            print("\n👋 Arrivederci!\n")
            break
        else:
            print("\n❌ Scelta non valida. Riprova.\n")
        
        # Pausa tra operazioni
        if choice in ['1', '2', '3']:
            input("\n\nPremi INVIO per tornare al menu...")
            print("\n" * 2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Programma terminato dall'utente.\n")
        sys.exit(0)
