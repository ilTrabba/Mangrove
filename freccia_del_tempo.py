import torch
import numpy as np
from safetensors.torch import load_file
import os
import re
import json
import time  # <-- Aggiunto per misurare i tempi

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================

def normalize_key(key):
    """Pulisce i nomi dei layer dai prefissi comuni per facilitare il matching."""
    key = key.replace('model.', '').replace('vit.', '').replace('transformer.', '')
    match = re.search(r'(layers|blocks|h|encoder\.layer|decoder\.layer|layer)\.(\d+)\.(.*)', key)
    if match: return f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
    return key

def is_integrator_layer(key, shape, exclude_value=False):
    """
    Filtro geometrico con blacklist estesa per l'Attention e CNN.
    """
    # 1. Filtro Geometrico Duro: Solo matrici (2D) e tensori convoluzionali (4D)
    if len(shape) not in [2, 4]: 
        return False
        
    key_lower = key.lower()
    
    # 2. Blacklist base (Normalizzazioni, bias, embedding, layer temporali)
    exclusions = ["norm", "embed", "lora", "bias", "time"]
    
    # 3. Blacklist per Query e Key (SEMPRE BLOCCATI)
    # Copre sia i layer di Stable Diffusion (to_q) che dei Transformer classici (q_proj, query)
    attention_q = ["to_q", "q_proj", ".q.", "query"]
    attention_k = ["to_k", "k_proj", ".k.", "key"]
    exclusions.extend(attention_q)
    exclusions.extend(attention_k)
    
    # 4. Blacklist per Value (controllata dal parametro)
    if exclude_value:
        attention_v = ["to_v", "v_proj", ".v.", "value"]
        exclusions.extend(attention_v)
        
    # Applica la blacklist
    if any(bad_word in key_lower for bad_word in exclusions):
        return False
        
    # Se arriva qui, il layer è valido (es. o_proj, up_proj, down_proj, gate_proj, fc1, conv)
    return True

def load_chunk(path):
    """Carica i pesi gestendo sia Safetensors che PyTorch Checkpoints storici."""
    try:
        if path.endswith(".safetensors"): 
            return load_file(path, device="cpu")
            
        # 1. Disattiviamo weights_only per permettere il caricamento dei .ckpt di Lightning
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        
        # 2. Se è un file .ckpt di Stable Diffusion, i pesi sono dentro la chiave "state_dict"
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            return ckpt["state_dict"]
            
        return ckpt
    except Exception as e: 
        print(f"Errore nel caricamento di {path}: {e}")
        return None

# ==============================================================================
# ENGINE V2.6: RATIO-AWARE GEOMETRIC LINEAGE
# ==============================================================================

class LineageEngineV2_6:
    def __init__(self, name_a, name_b):
        self.name_a = name_a
        self.name_b = name_b
        
        # Accumulatori Voti
        self.votes_a = 0.0 
        self.votes_b = 0.0 
        
        # Statistiche Diagnostiche
        self.mode_additive = 0   # Casi con Ratio > 1.1 (Vince Delta)
        self.mode_subtractive = 0 # Casi con Ratio < 1.1 (Vince Spettro)
        
        self.n_layers = 0

    def analyze_layer(self, w_a, w_b):
        w_a, w_b = w_a.float(), w_b.float()
        
        # --- FLATTEN PER LE CONVOLUZIONI (4D -> 2D) ---
        # L'algoritmo SVD richiede matrici 2D. 
        # Rimodelliamo i kernel CNN [Out, In, H, W] in [Out, In * H * W]
        if w_a.ndim == 4:
            w_a = w_a.view(w_a.shape[0], -1)
            w_b = w_b.view(w_b.shape[0], -1)
            
        delta = w_b - w_a
        
        if torch.norm(delta) < 1e-6: return

        # --- SVD COMPLETA (U + S) ---
        try:
            U_a, S_a, _ = torch.linalg.svd(w_a, full_matrices=False)
            U_b, S_b, _ = torch.linalg.svd(w_b, full_matrices=False)
            U_a, U_b = U_a[:, :10], U_b[:, :10]
        except: return

        # 1. Energie Delta
        en_a = torch.norm(torch.matmul(U_a.T, delta))**2
        en_b = torch.norm(torch.matmul(U_b.T, delta))**2
        
        # 2. Concentrazione Spettrale (Max Singular Value / Sum)
        conc_a = S_a[0] / (torch.sum(S_a) + 1e-9)
        conc_b = S_b[0] / (torch.sum(S_b) + 1e-9)
        
        # --- DECISION LOGIC V2.6 ---
        max_e = max(en_a, en_b)
        min_e = min(en_a, en_b) + 1e-9
        ratio = max_e / min_e
        
        if ratio > 1.01: 
            # MODE: ADDITIVE (Trust Delta)
            if en_b > en_a: self.votes_a += 1 
            else: self.votes_b += 1
            self.mode_additive += 1
        else:
            # MODE: SUBTRACTIVE / REFINING (Trust Spectrum)
            if conc_b > conc_a: self.votes_a += 1 
            else: self.votes_b += 1
            self.mode_subtractive += 1
            
        self.n_layers += 1

    def get_verdict(self):
        if self.n_layers == 0: return {"error": "0 Layers Analizzati"}
        
        is_a_father = self.votes_a > self.votes_b
        
        # Calcolo confidenza
        total = self.votes_a + self.votes_b + 1e-9
        ratio = self.votes_a / total
        conf = abs(ratio - 0.5) * 2 * 100
        
        father = self.name_a if is_a_father else self.name_b
        son = self.name_b if is_a_father else self.name_a
        
        stats = f"AdditiveLayers({self.mode_additive}) SubtractiveLayers({self.mode_subtractive})"
        
        return {
            "father": father,
            "son": son,
            "conf": min(conf, 99.9),
            "stats": stats,
            "error": None
        }

# ==============================================================================
# GESTIONE MODELLI E CARTELLE
# ==============================================================================

def find_model_path(tree_dir, model_name, depth=None):
    """
    Trova il path del modello supportando estensioni multiple e strutture piatte.
    """
    valid_exts = ('.safetensors', '.bin', '.ckpt', '.pt', '.pth')
    
    # 1. Cerca nella root directory (copre i modelli root E i dataset con cartella piatta)
    root_file = os.path.join(tree_dir, model_name)
    if os.path.isfile(root_file) and root_file.endswith(valid_exts):
        return root_file
        
    root_dir = os.path.join(tree_dir, model_name)
    if os.path.isdir(root_dir) and not model_name.startswith("depth_"):
        files = [f for f in os.listdir(root_dir) if f.endswith(valid_exts)]
        if files:
            return root_dir

    # 2. Cerca nelle cartelle depth_i
    depth_dirs = [f'depth_{depth}'] if (depth is not None and depth > 0) else ['depth_1', 'depth_2', 'depth_3', 'depth_4']
    
    for depth_dir in depth_dirs:
        depth_path = os.path.join(tree_dir, depth_dir)
        if not os.path.exists(depth_path):
            continue
            
        # Caso A: Cartella multi-file
        model_dir = os.path.join(depth_path, model_name)
        if os.path.isdir(model_dir):
            files = [f for f in os.listdir(model_dir) if f.endswith(valid_exts)]
            if files:
                return model_dir
        
        # Caso B: File singolo
        model_file = os.path.join(depth_path, model_name)
        if os.path.isfile(model_file) and model_file.endswith(valid_exts):
            return model_file
            
    return None

def load_model_weights(model_path):
    """Carica i pesi da un file singolo o fonde una directory shardata."""
    valid_exts = ('.safetensors', '.bin', '.ckpt', '.pt', '.pth')
    
    if os.path.isfile(model_path):
        return load_chunk(model_path)
    
    elif os.path.isdir(model_path):
        files = [f for f in os.listdir(model_path) if f.endswith(valid_exts)]
        if not files:
            return None
        
        combined_state_dict = {}
        for file in files:
            file_path = os.path.join(model_path, file)
            chunk = load_chunk(file_path)
            if chunk is not None:
                combined_state_dict.update(chunk)
        
        return combined_state_dict if combined_state_dict else None
    
    return None

def get_model_name(model_path):
    return os.path.basename(model_path)

# ==============================================================================
# RUNNER
# ==============================================================================

def compare_models_v2_6(path_a, path_b, exclude_value=False):
    sd_a = load_model_weights(path_a)
    sd_b = load_model_weights(path_b)
    if sd_a is None or sd_b is None: return {"error": "Load Fail"}

    map_a = {normalize_key(k): v for k, v in sd_a.items()}
    map_b = {normalize_key(k): v for k, v in sd_b.items()}
    
    name_a = get_model_name(path_a)
    name_b = get_model_name(path_b)
    
    engine = LineageEngineV2_6(name_a, name_b)
    keys = set(map_a.keys()) & set(map_b.keys())
    
    valid = 0
    
    # ⏱️ INIZIO CRONOMETRO MATEMATICO
    start_time = time.time()
    
    for k in keys:
        wa, wb = map_a[k], map_b[k]
        # Passiamo il parametro exclude_value al filtro geometrico
        if is_integrator_layer(k, wa.shape, exclude_value) and wa.shape == wb.shape:
            engine.analyze_layer(wa, wb)
            valid += 1
            
    if engine.n_layers == 0: 
        return {"error": "0 Layers Analizzati", "time": time.time() - start_time}
    
    verdict = engine.get_verdict()
    
    # ⏱️ FINE CRONOMETRO MATEMATICO
    calc_time = time.time() - start_time
    
    if not verdict.get("error"):
        verdict["stats"] += f" LayersUsed({valid})"
    
    verdict["time"] = calc_time
    return verdict

# ==============================================================================
# MAIN - ANALISI CON GROUND TRUTH JSON E DOPPIO TEST
# ==============================================================================

def extract_pairs_from_tree(tree_data):
    """Estrae tutte le coppie padre-figlio dal formato ground truth."""
    pairs = []
    root = tree_data.get("root")
    
    if "d1" in tree_data:
        d1_data = tree_data["d1"]
        if isinstance(d1_data, list):
            for child in d1_data: pairs.append((root, child, 0, 1))
        elif isinstance(d1_data, dict):
            for parent, children in d1_data.items():
                for child in children: pairs.append((parent, child, 0, 1))
    
    if "d2" in tree_data:
        for parent, children in tree_data["d2"].items():
            for child in children: pairs.append((parent, child, 1, 2))
            
    if "d3" in tree_data:
        for parent, children in tree_data["d3"].items():
            for child in children: pairs.append((parent, child, 2, 3))
            
    if "d4" in tree_data:
        for parent, children in tree_data["d4"].items():
            for child in children: pairs.append((parent, child, 3, 4))
            
    return pairs


def run_tree_analysis(tree_dir, ground_truth_json, tree_id=None):
    tree_name = os.path.basename(tree_dir)
    print(f"\n[⚔️] GEOMETRIC LINEAGE V2.6: {tree_name}")
    
    with open(ground_truth_json, 'r') as f:
        gt_data = json.load(f)
    
    if "TREES" in gt_data:
        trees = gt_data["TREES"]
    elif isinstance(gt_data, dict) and any(isinstance(k, int) or (isinstance(k, str) and k.isdigit()) for k in gt_data.keys()):
        trees = gt_data
    else:
        trees = {0: gt_data}
    
    normalized_trees = {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in trees.items()}
    trees = normalized_trees
    
    if tree_id is not None:
        if tree_id not in trees:
            print(f"Errore: Tree {tree_id} non trovato nel JSON. Alberi disponibili: {list(trees.keys())}")
            return
        trees = {tree_id: trees[tree_id]}
    
    for tid, tree_data in trees.items():
        print(f"\n--- Tree {tid} ---")
        gt_pairs = extract_pairs_from_tree(tree_data)
        
        print(f"\n{'PADRE (GT)':<20} | {'PRED: P -> F':<40} | {'CONF':<5} | {'LOGIC MODE'}")
        print("-" * 130)
        
        for parent_name, child_name, parent_depth, child_depth in gt_pairs:
            parent_path = find_model_path(tree_dir, parent_name, parent_depth)
            child_path = find_model_path(tree_dir, child_name, child_depth)
            
            if parent_path is None or child_path is None:
                err_name = parent_name if parent_path is None else child_name
                err_depth = parent_depth if parent_path is None else child_depth
                print(f"{parent_name[:20]:<20} | ERROR | Model '{err_name}' not found at depth {err_depth}")
                print("-" * 130)
                continue
            
            # --- TEST 1: Con i layer VALUE inclusi (exclude_value=False) ---
            res_with_v = compare_models_v2_6(child_path, parent_path, exclude_value=False)
            if not res_with_v.get("error"):
                status_v = "✅" if (res_with_v['father'] == parent_name or res_with_v['father'] == os.path.basename(parent_name)) else "❌"
                lineage_v = f"{res_with_v['father'][:18]}->{res_with_v['son'][:18]}"
                print(f"{parent_name[:20]:<20} (With V) | {lineage_v:<40} | {res_with_v['conf']:.0f}%  | {status_v} {res_with_v['stats']}")
                print(f"{'':<29} | ⏱️  Tempo elaborazione matematica: {res_with_v['time']:.2f}s")
            else:
                print(f"{parent_name[:20]:<20} (With V) | ERROR | {res_with_v['error']}")

            # --- TEST 2: Senza i layer VALUE (exclude_value=True) ---
            res_without_v = compare_models_v2_6(child_path, parent_path, exclude_value=True)
            if not res_without_v.get("error"):
                status_no_v = "✅" if (res_without_v['father'] == parent_name or res_without_v['father'] == os.path.basename(parent_name)) else "❌"
                lineage_no_v = f"{res_without_v['father'][:18]}->{res_without_v['son'][:18]}"
                print(f"{parent_name[:20]:<20} (No V)   | {lineage_no_v:<40} | {res_without_v['conf']:.0f}%  | {status_no_v} {res_without_v['stats']}")
                print(f"{'':<29} | ⏱️  Tempo elaborazione matematica: {res_without_v['time']:.2f}s")
            else:
                print(f"{parent_name[:20]:<20} (No V)   | ERROR | {res_without_v['error']}")
                
            print("-" * 130)

if __name__ == "__main__":
    # ==============================================================================
    # CONFIGURAZIONE AMBIENTE
    # ==============================================================================
    
    tree_directory = "/home/cristian/projects/dataset/WHISPER-SMALL"  # Metti qui la cartella in cui ci sono i file .ckpt o .safetensors
    ground_truth_file = "~/projects/Model_Graph/ground_truth.json"  
    tree_id = 11  # Stable diffusion è l'albero 3 nel tuo file
    
    # ==============================================================================
    
    tree_directory = os.path.expanduser(tree_directory)
    ground_truth_file = os.path.expanduser(ground_truth_file)
    
    if os.path.exists(tree_directory) and os.path.exists(ground_truth_file):
        run_tree_analysis(tree_directory, ground_truth_file, tree_id)
    else:
        print(f"Errore: Verifica che esistano:")
        print(f"  - Tree directory: {tree_directory}")
        print(f"  - Ground truth JSON: {ground_truth_file}")