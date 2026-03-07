#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import torch
import numpy as np
import itertools
from safetensors import safe_open

# ==============================================================================
# 1. FUNZIONI DI RICERCA E FILTRAGGIO
# ==============================================================================

def identify_layer(key, shape):
    """
    Identifica se il tensore è una matrice rilevante, la sua profondità e il tipo.
    Ritorna: (is_valid: bool, layer_idx: int, matrix_type: str)
    """
    if len(shape) != 2: 
        return False, -1, None
    
    k = key.lower()
    
    # Esclusioni base
    if any(x in k for x in ["norm", "embed", "lora", "bias", "bn", "wte", "wpe", "classifier", "lm_head", "pooler"]): 
        return False, -1, None

    # Estrai l'indice del layer
    match = re.search(r'(?:layers|blocks|h)\.(\d+)\.', k)
    layer_idx = int(match.group(1)) if match else -1
    
    if layer_idx == -1:
        return False, -1, None

    # Focus sulle matrici di SCRITTURA
    write_targets = ["o_proj", "wo", "down_proj", "fc2", "to_out"]
    
    if any(t in k for t in write_targets):
        return True, layer_idx, "write"
        
    return False, -1, None


def find_models(base_dir, mode_id):
    """
    Scansiona la cartella in base al MODE_ID.
    MODE_ID 1: File singoli (.safetensors)
    MODE_ID 2: Cartelle (ogni cartella è un modello sharded)
    """
    models = {}
    for root, dirs, files in os.walk(base_dir):
        safe_files = sorted([f for f in files if f.endswith(".safetensors")])
        if not safe_files: 
            continue
            
        if mode_id == 1:
            for f in safe_files:
                models[f.replace(".safetensors", "")] = [os.path.join(root, f)]
        elif mode_id == 2:
            models[os.path.basename(root)] = [os.path.join(root, f) for f in safe_files]
            
    return models


def index_model_tensors(file_paths):
    """
    Mappa {nome_tensore: percorso_file} per un singolo modello.
    """
    tensor_index = {}
    for path in file_paths:
        try:
            with safe_open(path, framework="pt", device="cpu") as st:
                for key in st.keys():
                    tensor_index[key] = path
        except Exception as e:
            print(f"⚠️ Errore indicizzazione {path}: {e}")
    return tensor_index

# ==============================================================================
# 2. MOTORE SVD SU DELTA W
# ==============================================================================

@torch.no_grad()
def analyze_delta_spectrum(base_paths, instruct_paths):
    layer_data = {"write": {}, "read": {}}
    max_layer_idx = 0

    # Creiamo un dizionario per i tensori del modello base
    base_tensors = {}
    for path in base_paths:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                base_tensors[key] = f.get_tensor(key).float()

    # Ora iteriamo sul modello Instruct e calcoliamo Delta W
    for path in instruct_paths:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key not in base_tensors:
                    continue
                    
                slice_obj = f.get_slice(key)
                shape = tuple(int(x) for x in slice_obj.get_shape())
                
                is_valid, l_idx, m_type = identify_layer(key, shape)
                if not is_valid or min(shape) < 16:
                    continue
                
                # Calcoliamo DELTA W
                tensor_instruct = f.get_tensor(key).float()
                tensor_delta = tensor_instruct - base_tensors[key]
                
                # SVD su DELTA W!
                S = torch.linalg.svdvals(tensor_delta)
                Energy = S ** 2
                
                e_top3 = Energy[:3].sum().item()
                e_top16 = Energy[:16].sum().item()
                
                if e_top16 > 1e-9:
                    ratio = e_top3 / e_top16
                    if l_idx not in layer_data[m_type]:
                        layer_data[m_type][l_idx] = []
                    layer_data[m_type][l_idx].append(ratio)
                    
                    if l_idx > max_layer_idx:
                        max_layer_idx = l_idx
                        
    # Calcolo Terzili (uguale a prima)
    terzile_size = (max_layer_idx + 1) / 3
    early, mid, late = [], [], []
    
    for l_idx, ratios in layer_data["write"].items():
        avg_ratio = np.mean(ratios)
        if l_idx < terzile_size: early.append(avg_ratio)
        elif l_idx < terzile_size * 2: mid.append(avg_ratio)
        else: late.append(avg_ratio)
            
    return {
        "early": np.mean(early) * 100 if early else 0,
        "mid": np.mean(mid) * 100 if mid else 0,
        "late": np.mean(late) * 100 if late else 0,
        "layers_found": max_layer_idx + 1
    }

# ==============================================================================
# 3. RUNNER PRINCIPALE E COMBINAZIONI
# ==============================================================================

if __name__ == "__main__":
    
    # --------------------------------------------------------------------------
    # ⚙️ IMPOSTAZIONI UTENTE
    # --------------------------------------------------------------------------
    BASE_DIR = "/home/trabbo/Downloads/Qwen2.5-0.5B-family"
    MODE_ID = 1  # 1 = File singoli, 2 = Cartelle/Shard
    # --------------------------------------------------------------------------
    
    print(f"\n🔍 SCANSIONE MODELLI IN: {BASE_DIR}")
    models = find_models(BASE_DIR, MODE_ID)
    model_names = sorted(models.keys())
    
    if len(model_names) < 2:
        print("❌ Servono almeno 2 modelli per fare un confronto (Delta W)!")
        exit()
        
    # Genera tutte le combinazioni possibili di coppie (A-B, A-C, B-C, ecc.)
    pairs = list(itertools.combinations(model_names, 2))
    
    print(f"✅ Trovati {len(model_names)} modelli. Verranno analizzate {len(pairs)} coppie.")
    print("   Analisi focalizzata sulle matrici di SCRITTURA (o_proj, down_proj)")
    print("   Metrica: % Energia SVD di Delta W (TOP-3 / TOP-16)\n")
    
    print(f"{'COPPIA (A <-> B)':<45} | {'LAYERS':<6} | {'EARLY (1/3)':<12} | {'MID (2/3)':<12} | {'LATE (3/3)':<12}")
    print("-" * 97)
    
    for name_A, name_B in pairs:
        # Formattazione per abbreviare nomi troppo lunghi
        short_A = name_A[:20] + ".." if len(name_A) > 22 else name_A
        short_B = name_B[:20] + ".." if len(name_B) > 22 else name_B
        pair_name = f"{short_A} <-> {short_B}"
        
        metrics = analyze_delta_spectrum(models[name_A], models[name_B])
        
        if metrics is None:
            print(f"{pair_name:<45} | {'ERR':<6} | Nessuna matrice in comune trovata")
        else:
            print(f"{pair_name:<45} | {metrics['layers_found']:<6} | "
                  f"{metrics['early']:>9.2f} % | {metrics['mid']:>9.2f} % | {metrics['late']:>9.2f} %")
            
    print("-" * 97)
    print("\n💡 GUIDA ALL'INTERPRETAZIONE:")
    print("   - Se metti a confronto [Base <-> Instruct_SFT], vedrai percentuali stabili sui layer.")
    print("   - Se metti a confronto [Base <-> Instruct_RLHF/Censurato], la colonna LATE schizzerà verso l'alto (>60-70%).")
    print("   - Se confronti due modelli simili tra loro (es. SFT_1 <-> SFT_2), le percentuali saranno basse.")