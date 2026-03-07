import os
import re
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

POLICY = "breadth_first_per_albero" # Modifica con la policy desiderata

# Path della directory contenente i file safetensors
DATASET_PATH = "/home/trabbo/Documents/Universita/BigData/Models_for_Project/MoTHer"  # Modifica con il path corretto del tuo dataset

# Path del repository Model_Graph
REPO_PATH = os.path.expanduser("~/Documents/GitHub/Model_Graph")

# Endpoint API
API_URL = "http://localhost:5001/api/models"

# Timeout per l'attesa del backend (secondi)
BACKEND_TIMEOUT = 200

# Intervallo di polling per verificare se il backend è pronto (secondi)
POLL_INTERVAL = 2

# Tempo massimo di attesa per un singolo upload (secondi)
UPLOAD_TIMEOUT = 500
UPLOAD_TIMEOUT = 500

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
    print(f"[{timestamp}] {message}")


def log_separator():
    """Stampa un separatore visivo."""
    print("=" * 70)


# ============================================
# CATALOGAZIONE FILE
# ============================================

def explore_dataset(dataset_path):
    """
    Esplora la directory del dataset e cataloga tutti i file safetensors
    per albero e profondità.
    
    Supporta sia:  
    - Path alla cartella contenente più alberi (es.  dataset_model_heritage/)
    - Path diretto a un singolo albero (es.  dataset_model_heritage/Tree-1-Mother/)
    """
    catalog = {}
    dataset = Path(dataset_path)
    
    if not dataset.exists():
        log(f"ERRORE: Directory non trovata: {dataset_path}")
        sys.exit(1)
    
    # Verifica se il path è già un albero (contiene file .safetensors o cartelle depth_N)
    is_single_tree = False
    for item in dataset.iterdir():
        if (item.is_file() and item.suffix == ".safetensors") or \
           (item.is_dir() and item.name.startswith("depth_")):
            is_single_tree = True
            break
    
    if is_single_tree:
        # Il path passato È già un albero
        tree_dirs = [dataset]
    else:
        # Il path contiene più alberi come sottocartelle
        tree_dirs = [d for d in sorted(dataset.iterdir()) if d.is_dir()]
    
    # Itera su ogni albero
    for tree_dir in tree_dirs:
        tree_name = tree_dir.name
        catalog[tree_name] = {}
        
        # Cerca file safetensors nella root dell'albero (depth 0)
        for item in tree_dir.iterdir():
            if item.is_file() and item.suffix == ".safetensors":
                if 0 not in catalog[tree_name]:
                    catalog[tree_name][0] = []
                catalog[tree_name][0].append({
                    "filename": item.name,
                    "path": str(item)
                })
        
        # Cerca cartelle depth_N
        for item in tree_dir.iterdir():
            if item.is_dir() and item.name.startswith("depth_"):
                try:
                    depth = int(item.name.split("_")[1])
                except (IndexError, ValueError):
                    continue
                
                # Esplora ricorsivamente la cartella depth_N
                for safetensor_file in item.rglob("*.safetensors"):
                    if depth not in catalog[tree_name]: 
                        catalog[tree_name][depth] = []
                    catalog[tree_name][depth].append({
                        "filename": safetensor_file.name,
                        "path": str(safetensor_file)
                    })
    
    return catalog


def get_max_depth(catalog):
    """Ritorna la profondità massima presente nel catalogo."""
    max_depth = 0
    for tree_name, depths in catalog.items():
        for depth in depths.keys():
            max_depth = max(max_depth, depth)
    return max_depth


def get_all_files_at_depth(catalog, depth):
    """Ritorna tutti i file a una specifica profondità (da tutti gli alberi)."""
    files = []
    for tree_name in sorted(catalog.keys()):
        if depth in catalog[tree_name]: 
            for file_info in catalog[tree_name][depth]:
                files.append({
                    **file_info,
                    "tree": tree_name,
                    "depth": depth
                })
    return files

# ============================================
# POLICY DI ORDINAMENTO
# ============================================

def apply_policy(catalog, policy):
    """Applica la policy di ordinamento e ritorna la lista ordinata dei file."""
    
    max_depth = get_max_depth(catalog)
    ordered_files = []
    
    if policy == "casuale":
        # Raccoglie tutti i file e li mescola
        for tree_name, depths in catalog.items():
            for depth, files in depths.items():
                for file_info in files:
                    ordered_files.append({
                        **file_info,
                        "tree": tree_name,
                        "depth":  depth
                    })
        random.shuffle(ordered_files)
    
    elif policy == "corretto":
        # depth 0 → depth 1 → ...  → depth n
        for depth in range(max_depth + 1):
            ordered_files.extend(get_all_files_at_depth(catalog, depth))
    
    elif policy == "inverso":
        # depth n → depth n-1 → ... → depth 0
        for depth in range(max_depth, -1, -1):
            ordered_files.extend(get_all_files_at_depth(catalog, depth))
    
    elif policy == "incrociato": 
        # depth n → depth 0 → depth n-1 → depth 1 → depth n-2 → depth 2 → ...
        low = 0
        high = max_depth
        turn_high = True
        
        while low <= high:
            if turn_high:
                current_files = get_all_files_at_depth(catalog, high)
                random.shuffle(current_files)  # Mescola i file a questa profondità
                ordered_files.extend(current_files)
                high -= 1
            else:
                current_files = get_all_files_at_depth(catalog, low)
                random.shuffle(current_files)  # Mescola i file a questa profondità
                ordered_files.extend(current_files)
                low += 1
            turn_high = not turn_high
    
    elif policy == "breadth_first_per_albero":
        # Completa un albero intero prima di passare al successivo
        for tree_name in sorted(catalog.keys()):
            for depth in sorted(catalog[tree_name].keys()):
                for file_info in catalog[tree_name][depth]:
                    ordered_files.append({
                        **file_info,
                        "tree": tree_name,
                        "depth": depth
                    })
    
    elif policy == "round_robin":
        # Un file da ogni albero a depth 0, poi un file da ogni albero a depth 1, ecc.
        for depth in range(max_depth + 1):
            tree_names = sorted(catalog.keys())
            # Crea iteratori per ogni albero a questa profondità
            iterators = {}
            for tree_name in tree_names:
                if depth in catalog[tree_name]:
                    iterators[tree_name] = iter(catalog[tree_name][depth])
            
            # Round robin tra gli alberi
            while iterators:
                exhausted = []
                for tree_name in list(iterators.keys()):
                    try:
                        file_info = next(iterators[tree_name])
                        ordered_files. append({
                            **file_info,
                            "tree":  tree_name,
                            "depth": depth
                        })
                    except StopIteration:
                        exhausted.append(tree_name)
                
                for tree_name in exhausted: 
                    del iterators[tree_name]
    
    elif policy == "worst_case":
        # Figli prima dei genitori (depth n → depth 0)
        # Uguale a "inverso" ma con ordine casuale all'interno di ogni depth
        for depth in range(max_depth, -1, -1):
            files_at_depth = get_all_files_at_depth(catalog, depth)
            random.shuffle(files_at_depth)
            ordered_files.extend(files_at_depth)
    
    else:
        log(f"ERRORE: Policy '{policy}' non riconosciuta")
        sys.exit(1)
    
    return ordered_files


# ============================================
# GESTIONE PROCESSI
# ============================================

def start_tool(repo_path):
    """Avvia il tool Model_Graph e ritorna il processo."""
    log(f"Avvio del tool da: {repo_path}")
    
    os.chdir(repo_path)
    
    # Avvia run.
    # sh in background
    process = subprocess.Popen(
        ["./run.sh"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid  # Crea un nuovo gruppo di processi
    )
    
    log(f"Tool avviato (PID: {process.pid})")
    return process


def wait_for_backend(timeout=BACKEND_TIMEOUT):
    """Attende che il backend sia pronto."""
    log(f"Attendo che il backend sia pronto su {API_URL}...")
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(API_URL, timeout=5)
            if response.status_code in [200, 404, 405]:  # Backend risponde
                log("Backend pronto!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.Timeout:
            pass
        
        time.sleep(POLL_INTERVAL)
    
    log(f"ERRORE: Backend non disponibile dopo {timeout} secondi")
    return False


def stop_tool(process):
    """Ferma il tool Model_Graph."""
    if process:
        log("Arresto del tool...")
        try:
            # Termina l'intero gruppo di processi
            os.killpg(os.getpgid(process. pid), signal.SIGTERM)
            process.wait(timeout=10)
            log("Tool arrestato correttamente")
        except Exception as e:
            log(f"Errore durante l'arresto: {e}")
            try:
                os.killpg(os.getpgid(process. pid), signal.SIGKILL)
            except:
                pass


# ============================================
# VERIFICA E SINCRONIZZAZIONE
# ============================================

def verify_model_exists(model_id):
    """
    Verifica che un modello sia effettivamente presente e processato.
    Ritorna True se il modello esiste ed è pronto, False altrimenti.
    """
    for attempt in range(VERIFY_MAX_RETRIES):
        try:
            response = requests.get(
                f"{API_URL}/{model_id}",
                timeout=10
            )
            if response.status_code == 200:
                model_data = response.json()
                # Verifica che il modello sia completamente processato
                # (adatta questa logica in base alla tua API)
                if model_data.get("status") == "ready" or "id" in model_data:
                    return True
        except Exception: 
            pass
        
        time.sleep(VERIFY_INTERVAL)
    
    return False


def wait_after_upload(file_info):
    """
    Applica le pause necessarie dopo un upload in base alla depth. 
    """
    # Sleep base dopo ogni upload
    if SLEEP_AFTER_UPLOAD > 0:
        log(f"         Attesa {SLEEP_AFTER_UPLOAD}s...")
        time.sleep(SLEEP_AFTER_UPLOAD)
    
    # Sleep aggiuntivo per i modelli root (depth 0)
    if file_info['depth'] == 0 and SLEEP_AFTER_ROOT > SLEEP_AFTER_UPLOAD:
        extra_sleep = SLEEP_AFTER_ROOT - SLEEP_AFTER_UPLOAD
        log(f"         Attesa extra per root:  {extra_sleep}s...")
        time.sleep(extra_sleep)


# ============================================
# UPLOAD
# ============================================

def upload_file(file_info):
    """
    Esegue l'upload di un singolo file safetensors. 
    Ritorna (success, message, duration, model_id).
    """
    start_time = time.time()
    
    file_path = file_info["path"]
    filename = file_info["filename"]
    
    # Usa il nome del file (senza estensione) come nome del modello
    model_name = os.path.splitext(filename)[0]
    
    try:
        with open(file_path, "rb") as f:
            files = {"file": (filename, f, "application/octet-stream")}
            data = {
                "name": model_name,
                "description": f"Uploaded from tree:  {file_info['tree']}, depth: {file_info['depth']}",
                "is_foundation_model": bool(re.search(r"D0", model_name))
            }
            
            response = requests.post(
                API_URL,
                files=files,
                data=data,
                timeout=UPLOAD_TIMEOUT
            )
        
        duration = time.time() - start_time
        
        if response.status_code in [200, 201]:
            result = response.json()
            model_id = result.get('model', {}).get('id', None)
            return True, f"OK - Model ID: {model_id}", duration, model_id
        else:
            error_msg = response.json().get("error", response.text)
            return False, f"ERRORE ({response.status_code}): {error_msg}", duration, None
    
    except requests.exceptions. Timeout:
        duration = time.time() - start_time
        return False, "ERRORE: Timeout durante l'upload", duration, None
    
    except Exception as e:
        duration = time.time() - start_time
        return False, f"ERRORE: {str(e)}", duration, None


# ============================================
# MAIN
# ============================================

def main():
    log_separator()
    log("UPLOAD AUTOMATICO SAFETENSORS - Model_Graph")
    log_separator()
    
    log(f"Policy selezionata: {POLICY}")
    log(f"Dataset path: {DATASET_PATH}")
    log(f"Repository path: {REPO_PATH}")
    log(f"Sleep dopo upload: {SLEEP_AFTER_UPLOAD}s")
    log(f"Sleep dopo root: {SLEEP_AFTER_ROOT}s")
    log(f"Verifica upload: {'Sì' if VERIFY_UPLOAD else 'No'}")
    log_separator()
    
    # 1. Esplora e cataloga i file
    log("Esplorazione del dataset...")
    catalog = explore_dataset(DATASET_PATH)
    
    total_trees = len(catalog)
    total_files = sum(
        len(files)
        for depths in catalog.values()
        for files in depths.values()
    )
    max_depth = get_max_depth(catalog)
    
    log(f"Trovati {total_files} file safetensors in {total_trees} alberi (profondità max: {max_depth})")
    
    # Mostra riepilogo per albero
    for tree_name in sorted(catalog.keys()):
        depths_info = ", ".join(
            f"D{d}:{len(files)}"
            for d, files in sorted(catalog[tree_name].items())
        )
        log(f"  - {tree_name}: {depths_info}")
    
    log_separator()
    
    # 2. Applica la policy di ordinamento
    log(f"Applicazione policy '{POLICY}'...")
    ordered_files = apply_policy(catalog, POLICY)
    log(f"Ordine di upload determinato per {len(ordered_files)} file")
    log_separator()
    
    # 3. Avvia il tool
    process = None
    try:
        process = start_tool(REPO_PATH)
        
        # 4. Attendi che il backend sia pronto
        if not wait_for_backend():
            log("Impossibile procedere: backend non disponibile")
            stop_tool(process)
            sys.exit(1)
        
        log_separator()
        
        # 5. Esegui gli upload
        log("Inizio upload dei file...")
        log_separator()
        
        success_count = 0
        error_count = 0
        verified_count = 0
        total_duration = 0
        
        for i, file_info in enumerate(ordered_files, 1):
            log(f"[{i}/{len(ordered_files)}] Upload: {file_info['filename']}")
            log(f"         Albero: {file_info['tree']}, Depth: {file_info['depth']}")
            
            success, message, duration, model_id = upload_file(file_info)
            total_duration += duration
            
            if success:
                success_count += 1
                log(f"         {message} ({duration:.2f}s)")
                
                # Verifica che il modello sia effettivamente pronto
                if VERIFY_UPLOAD and model_id: 
                    log(f"         Verifica completamento...")
                    if verify_model_exists(model_id):
                        log(f"         ✓ Modello verificato e pronto")
                        verified_count += 1
                    else:
                        log(f"         ⚠ Verifica fallita, continuo comunque")
                
                # Applica le pause di sincronizzazione
                wait_after_upload(file_info)
            
            else:
                error_count += 1
                log(f"         {message} ({duration:.2f}s)")
            
            print()  # Riga vuota per leggibilità
        
        # 6. Riepilogo finale
        log_separator()
        log("RIEPILOGO FINALE")
        log_separator()
        log(f"Policy utilizzata: {POLICY}")
        log(f"File totali: {len(ordered_files)}")
        log(f"Upload riusciti: {success_count}")
        log(f"Upload verificati: {verified_count}")
        log(f"Upload falliti: {error_count}")
        log(f"Tempo totale: {total_duration:.2f}s")
        if ordered_files:
            log(f"Tempo medio per upload: {total_duration/len(ordered_files):.2f}s")
        log_separator()
    
    except KeyboardInterrupt:
        log("\nInterruzione manuale rilevata")
    
    finally:
        # 7. Chiudi il tool
        stop_tool(process)
        log("Script terminato")


if __name__ == "__main__":
    main()