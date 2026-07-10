import os
import re
import sys
import time
import random
import requests
import subprocess
import signal
from datetime import datetime
from pathlib import Path

# ============================================
# CONFIGURAZIONE
# ============================================

# Policy di inserimento disponibili:  
# - "casuale":  ordine completamente random
# - "corretto": depth 0 → depth 1 → depth 2 → ...  → depth n (tutti gli alberi per ogni depth)
# - "inverso": depth n → depth n-1 → ...  → depth 0
# - "incrociato": depth n → depth 0 → depth n-1 → depth 1 → depth n-2 → depth 2 → ... 
# - "breadth_first_per_albero": completa un albero intero prima di passare al successivo
# - "round_robin": un file da ogni albero a depth 0, poi un file da ogni albero a depth 1, ecc. 
# - "worst_case": figli prima dei genitori (massimizza conflitti di dipendenza)

POLICY = "breadth_first_per_albero"

# Path della directory contenente i file safetensors
DATASET_PATH = "/home/cristian/projects/dataset/roBERTa"

# Path del repository Model_Graph
REPO_PATH = os.path.expanduser("/home/cristian/projects/Model_Graph")

# Endpoint API
API_URL = "http://localhost:5002/api/models"

# Timeout per l'attesa del backend (secondi)
BACKEND_TIMEOUT = 200

# Intervallo di polling per verificare se il backend è pronto (secondi)
POLL_INTERVAL = 2

# Tempo massimo di attesa per un singolo upload (secondi) - Nessun timeout per file giganti
UPLOAD_TIMEOUT = None 

# ============================================
# CONFIGURAZIONE SINCRONIZZAZIONE
# ============================================

# Tempo di attesa dopo ogni upload (secondi)
SLEEP_AFTER_UPLOAD = 2.0

# Tempo di attesa aggiuntivo dopo upload a depth 0 (root models)
SLEEP_AFTER_ROOT = 5.0

# Abilita verifica che il modello sia effettivamente presente dopo l'upload
VERIFY_UPLOAD = True

# Numero massimo di tentativi per la verifica
VERIFY_MAX_RETRIES = 5

# Intervallo tra i tentativi di verifica (secondi)
VERIFY_INTERVAL = 1.0


# ============================================
# FUNZIONI DI UTILITÀ
# ============================================

def log(message):
    """Stampa un messaggio con timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def log_separator():
    """Stampa un separatore visivo."""
    print("=" * 70, flush=True)


# ============================================
# CATALOGAZIONE FILE (AGGIORNATA PER SHARD)
# ============================================

# ============================================
# CATALOGAZIONE FILE (AGGIORNATA PER SHARD E BIN)
# ============================================

def explore_dataset(dataset_path):
    """
    Esplora il dataset. Gestisce:
    - File .safetensors e .bin singoli
    - Cartelle contenenti multipli .safetensors o .bin (Shard)
    """
    catalog = {}
    dataset = Path(dataset_path)
    
    if not dataset.exists():
        log(f"ERRORE: Directory non trovata: {dataset_path}")
        sys.exit(1)
    
    # Identifica se il path è già un singolo albero o un contenitore di alberi
    is_single_tree = any(
        item.name.startswith("depth_") for item in dataset.iterdir() if item.is_dir()
    )
    
    tree_dirs = [dataset] if is_single_tree else [d for d in sorted(dataset.iterdir()) if d.is_dir()]
    
    for tree_dir in tree_dirs:
        tree_name = tree_dir.name
        catalog[tree_name] = {}
        
        # Helper interno per scansionare una specifica directory (root o depth_N)
        def scan_level(directory, depth_level):
            if depth_level not in catalog[tree_name]:
                catalog[tree_name][depth_level] = []
                
            for item in sorted(directory.iterdir()):
                # CASO 1: File singolo .safetensors o .bin
                if item.is_file() and item.suffix in [".safetensors", ".bin"]:
                    catalog[tree_name][depth_level].append({
                        "name": item.stem, # Nome senza estensione (.safetensors o .bin)
                        "paths": [str(item)],
                        "type": "single"
                    })
                
                # CASO 2: Cartella (Modello Sharded)
                # Ignoriamo le cartelle strutturali 'depth_N'
                elif item.is_dir() and not item.name.startswith("depth_"):
                    # Cerchiamo sia file .safetensors che .bin all'interno della cartella
                    model_files = sorted(list(item.glob("*.safetensors")) + list(item.glob("*.bin")))
                    if model_files:
                        catalog[tree_name][depth_level].append({
                            "name": item.name, # Nome della cartella
                            "paths": [str(f) for f in model_files],
                            "type": "sharded"
                        })

        # Scansiona la root dell'albero (depth 0)
        scan_level(tree_dir, 0)
        
        # Scansiona le sottocartelle depth_N
        for item in tree_dir.iterdir():
            if item.is_dir() and item.name.startswith("depth_"):
                try:
                    depth = int(item.name.split("_")[1])
                    scan_level(item, depth)
                except (IndexError, ValueError):
                    pass
                    
    return catalog

def get_max_depth(catalog):
    """Ritorna la profondità massima presente nel catalogo."""
    max_depth = 0
    for tree_name, depths in catalog.items():
        if depths:
            max_depth = max(max_depth, max(depths.keys()))
    return max_depth


def get_all_files_at_depth(catalog, depth):
    """Ritorna tutti i modelli a una specifica profondità."""
    models = []
    for tree_name in sorted(catalog.keys()):
        if depth in catalog[tree_name]: 
            for model_info in catalog[tree_name][depth]:
                models.append({
                    **model_info,
                    "tree": tree_name,
                    "depth": depth
                })
    return models

# ============================================
# POLICY DI ORDINAMENTO
# ============================================

def apply_policy(catalog, policy):
    """Applica la policy di ordinamento."""
    max_depth = get_max_depth(catalog)
    ordered_files = []
    
    if policy == "casuale":
        for tree_name, depths in catalog.items():
            for depth, files in depths.items():
                for file_info in files:
                    ordered_files.append({**file_info, "tree": tree_name, "depth": depth})
        random.shuffle(ordered_files)
    
    elif policy == "corretto":
        for depth in range(max_depth + 1):
            ordered_files.extend(get_all_files_at_depth(catalog, depth))
    
    elif policy == "inverso" or policy == "worst_case":
        for depth in range(max_depth, -1, -1):
            files_at_depth = get_all_files_at_depth(catalog, depth)
            if policy == "worst_case":
                random.shuffle(files_at_depth)
            ordered_files.extend(files_at_depth)
            
    elif policy == "incrociato": 
        low, high = 0, max_depth
        turn_high = True
        while low <= high:
            if turn_high:
                # Estrae i file alla profondità massima attuale
                current_depth_files = get_all_files_at_depth(catalog, high)
                # Li mescola in ordine casuale
                random.shuffle(current_depth_files)
                # Li aggiunge alla lista finale
                ordered_files.extend(current_depth_files)
                high -= 1
            else:
                # Estrae i file alla profondità minima attuale
                current_depth_files = get_all_files_at_depth(catalog, low)
                # Li mescola in ordine casuale (anche se è 1 solo, come la radice, non crea problemi)
                random.shuffle(current_depth_files)
                # Li aggiunge alla lista finale
                ordered_files.extend(current_depth_files)
                low += 1
            turn_high = not turn_high
            
    elif policy == "breadth_first_per_albero":
        for tree_name in sorted(catalog.keys()):
            for depth in sorted(catalog[tree_name].keys()):
                for file_info in catalog[tree_name][depth]:
                    ordered_files.append({**file_info, "tree": tree_name, "depth": depth})
                    
    elif policy == "round_robin":
        for depth in range(max_depth + 1):
            tree_names = sorted(catalog.keys())
            iterators = {t: iter(catalog[t][depth]) for t in tree_names if depth in catalog[t]}
            while iterators:
                exhausted = []
                for t in list(iterators.keys()):
                    try:
                        ordered_files.append({**next(iterators[t]), "tree": t, "depth": depth})
                    except StopIteration:
                        exhausted.append(t)
                for t in exhausted: 
                    del iterators[t]
    else:
        log(f"ERRORE: Policy '{policy}' non riconosciuta")
        sys.exit(1)
    
    return ordered_files

# ============================================
# GESTIONE PROCESSI E SINCRONIZZAZIONE
# ============================================

def start_tool(repo_path):
    log(f"Avvio del tool da: {repo_path}")
    os.chdir(repo_path)
    process = subprocess.Popen(
        ["./run.sh"],
        stdout=None, # Lascia che stampi a schermo
        stderr=None, 
        preexec_fn=os.setsid
    )
    log(f"Tool avviato (PID: {process.pid})")
    return process

def wait_for_backend(timeout=BACKEND_TIMEOUT):
    log(f"Attendo che il backend sia pronto su {API_URL}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            if requests.get(API_URL, timeout=5).status_code in [200, 404, 405]:
                log("Backend pronto!")
                return True
        except: pass
        time.sleep(POLL_INTERVAL)
    return False

def stop_tool(process):
    if process:
        log("Arresto del tool...")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=10)
        except:
            try: os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except: pass

def verify_model_exists(model_id):
    for _ in range(VERIFY_MAX_RETRIES):
        try:
            res = requests.get(f"{API_URL}/{model_id}", timeout=10)
            if res.status_code == 200 and ("status" in res.json() or "id" in res.json()):
                return True
        except: pass
        time.sleep(VERIFY_INTERVAL)
    return False

def wait_after_upload(file_info):
    if SLEEP_AFTER_UPLOAD > 0:
        time.sleep(SLEEP_AFTER_UPLOAD)
    if file_info['depth'] == 0 and SLEEP_AFTER_ROOT > SLEEP_AFTER_UPLOAD:
        time.sleep(SLEEP_AFTER_ROOT - SLEEP_AFTER_UPLOAD)

# ============================================
# UPLOAD (AGGIORNATO PER SHARD)
# ============================================

def upload_file(model_info):
    """
    Esegue l'upload di un modello. Gestisce l'apertura simultanea 
    di molteplici shard per l'invio multipart in un'unica richiesta.
    """
    start_time = time.time()
    
    paths = model_info["paths"]
    model_name = model_info["name"]
    is_foundation = (model_info['depth'] == 0) or bool(re.search(r"D0", model_name))
    
    opened_files = []
    
    try:
        # Prepara la lista 'files' per la richiesta multipart
        # Formato: [('file', (nome_file, oggetto_file, mime_type)), ...]
        files_payload = []
        for p in paths:
            f = open(p, "rb")
            opened_files.append(f)
            files_payload.append(
                ("file", (os.path.basename(p), f, "application/octet-stream"))
            )

        data = {
            "name": model_name,
            "description": f"Tree: {model_info['tree']} | Depth: {model_info['depth']} | Shards: {len(paths)}",
            #"is_foundation_model": str(is_foundation).lower() # Flask si aspetta spesso stringhe
        }
        
        response = requests.post(
            API_URL,
            files=files_payload,
            data=data,
            timeout=UPLOAD_TIMEOUT
        )
        
        duration = time.time() - start_time
        
        # Chiude i file immediatamente dopo l'invio
        for f in opened_files: f.close()
        opened_files.clear()

        if response.status_code in [200, 201]:
            result = response.json()
            m_id = result.get('model', {}).get('id', None) or result.get('id')
            return True, f"OK - ID: {m_id}", duration, m_id
        else:
            err = response.json().get("error", response.text) if response.headers.get('content-type') == 'application/json' else response.text
            return False, f"ERRORE ({response.status_code}): {err}", duration, None
            
    except Exception as e:
        for f in opened_files: f.close()
        return False, f"ECCEZIONE: {str(e)}", time.time() - start_time, None


# ============================================
# MAIN
# ============================================

def main():
    log_separator()
    log("UPLOAD AUTOMATICO SAFETENSORS - Model_Graph")
    log_separator()
    
    log("Esplorazione del dataset in corso...")
    catalog = explore_dataset(DATASET_PATH)
    
    total_trees = len(catalog)
    total_models = sum(len(models) for depths in catalog.values() for models in depths.values())
    max_depth = get_max_depth(catalog)
    
    log(f"Trovati {total_models} MODELLI in {total_trees} alberi (profondità max: {max_depth})")
    
    ordered_models = apply_policy(catalog, POLICY)
    log_separator()
    
    process = None
    try:
        process = start_tool(REPO_PATH)
        if not wait_for_backend():
            stop_tool(process)
            sys.exit(1)
        
        log_separator()
        log("Inizio upload...")
        
        success_count = error_count = verified_count = total_duration = 0
        
        for i, model_info in enumerate(ordered_models, 1):
            # Print personalizzato a seconda se è shardato o no
            shard_info = f"({len(model_info['paths'])} shard)" if model_info['type'] == 'sharded' else "(1 file)"
            
            log(f"[{i}/{len(ordered_models)}] Uploading: {model_info['name']} {shard_info}")
            log(f"         Albero: {model_info['tree']}, Depth: {model_info['depth']}")
            
            success, message, duration, model_id = upload_file(model_info)
            total_duration += duration
            
            if success:
                success_count += 1
                log(f"         ✓ {message} ({duration:.2f}s)")
                
                if VERIFY_UPLOAD and model_id: 
                    if verify_model_exists(model_id):
                        verified_count += 1
                    else:
                        log(f"         ⚠ Verifica DB fallita, ma upload OK")
                wait_after_upload(model_info)
            else:
                error_count += 1
                log(f"         ❌ {message} ({duration:.2f}s)")
            print() 
            
        log_separator()
        log(f"Completato! Successi: {success_count}, Falliti: {error_count}, Tempo: {total_duration:.2f}s")
        log_separator()
        
    except KeyboardInterrupt:
        log("\nInterruzione manuale")
    finally:
        stop_tool(process)

if __name__ == "__main__":
    main()