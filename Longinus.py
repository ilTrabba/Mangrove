import torch
import numpy as np
from safetensors.torch import load_file
import os
import re
import gc
import sys

# ==============================================================================
# 1. CONFIGURAZIONE & FILTRI
# ==============================================================================

def normalize_key(key):
    key = key.replace('base_model.model.', '')
    key = key.replace('model.', '')
    key = key.replace('vit.', '')         
    key = key.replace('transformer.', '')
    
    match = re.search(r'(layers|blocks|h|encoder\.layer|decoder\.layer|layer)\.(\d+)\.(.*)', key)
    if match: 
        return f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
    return key

def is_integrator_layer(key, shape):
    # Filtro dimensionale: solo matrici 2D (Linear layers)
    if len(shape) != 2: return False
    key = key.lower()
    
    # Esclusioni generiche (Norm, Embedding, Bias, LoRA adapter)
    if any(x in key for x in ["norm", "embed", "lora", "bias", "bn", "wte", "wpe"]): return False
    
    # --- LISTA TARGET SICURA (Solo Dense + Value) ---
    targets = [
        # --- Layer Densi Generici ---
        "dense", "fc1", "fc2", "fc", "mlp", 
        "dense_h_to_4h", "dense_4h_to_h", # NeoX / Falcon
        
        # --- LLaMA / Mistral / Qwen (MLP & Output) ---
        "down_proj", "up_proj", "gate_proj", # MLP
        "o_proj",                            # Output
        
        # --- Attention VALUE (Solo Value, NO Query/Key) ---
        "value",  
        "v_proj", 
        "to_v",
        
        # --- Output Projections specifiche ---
        "out_proj",  # GPT-2
        "c_proj",    # GPT-2
        "attn.proj", # ViT/BERT output
        "wo",        # T5 output
        
        # --- T5 MLP Parts ---
        "wi", "wi_0", "wi_1",
    ]
    
    return any(t in key for t in targets)

def load_chunk(path):
    try:
        if path.endswith(".safetensors"): return load_file(path, device="cpu")
        return torch.load(path, map_location="cpu", weights_only=True)
    except: return None

# ==============================================================================
# 2. ENGINE V2.6 (Analisi Geometrica Lineage)
# ==============================================================================

class LineageEngineV2_6:
    def __init__(self, name_a, name_b):
        self.name_a = name_a
        self.name_b = name_b
        self.votes_a = 0.0 
        self.votes_b = 0.0 
        self.mode_additive = 0   
        self.mode_subtractive = 0 
        self.n_layers = 0

    def analyze_layer(self, w_a, w_b):
        w_a, w_b = w_a.float(), w_b.float()
        delta = w_b - w_a
        
        if torch.norm(delta) < 1e-6: return

        try:
            # SVD sui primi 10 componenti
            U_a, S_a, _ = torch.linalg.svd(w_a, full_matrices=False)
            U_b, S_b, _ = torch.linalg.svd(w_b, full_matrices=False)
            U_a, U_b = U_a[:, :10], U_b[:, :10]
        except: return

        # Proiezione del Delta sugli spazi dei pesi
        en_a = torch.norm(torch.matmul(U_a.T, delta))**2
        en_b = torch.norm(torch.matmul(U_b.T, delta))**2
        
        conc_a = S_a[0] / (torch.sum(S_a) + 1e-9)
        conc_b = S_b[0] / (torch.sum(S_b) + 1e-9)
        
        max_e = max(en_a, en_b)
        min_e = min(en_a, en_b) + 1e-9
        ratio = max_e / min_e
        
        if ratio > 1.01: 
            if en_b > en_a: self.votes_a += 1 
            else: self.votes_b += 1
            self.mode_additive += 1
        else:
            if conc_b > conc_a: self.votes_a += 1 
            else: self.votes_b += 1
            self.mode_subtractive += 1
            
        self.n_layers += 1

    def get_verdict(self):
        if self.n_layers == 0: return None
        is_a_father = self.votes_a > self.votes_b
        total = self.votes_a + self.votes_b + 1e-9
        ratio = self.votes_a / total
        conf = abs(ratio - 0.5) * 2 * 100
        
        father = self.name_a if is_a_father else self.name_b
        son = self.name_b if is_a_father else self.name_a
        
        stats = f"Add({self.mode_additive}) Sub({self.mode_subtractive}) Layers({self.n_layers})"
        return {"father": father, "son": son, "conf": min(conf, 99.9), "stats": stats, "error": None}

# ==============================================================================
# 3. INTERFACCIA CONFRONTO
# ==============================================================================

def compare_models_v2_6(path_a, path_b):
    sd_a = load_chunk(path_a)
    sd_b = load_chunk(path_b)
    if sd_a is None or sd_b is None: return {"error": "Load Fail"}

    map_a = {normalize_key(k): v for k, v in sd_a.items()}
    map_b = {normalize_key(k): v for k, v in sd_b.items()}
    
    keys = set(map_a.keys()) & set(map_b.keys())
    
    if len(keys) == 0:
        return {"error": "Key Mismatch (0 Common Keys)"}

    engine = LineageEngineV2_6(os.path.basename(path_a), os.path.basename(path_b))
    
    skipped_identical = 0
    
    for k in keys:
        wa, wb = map_a[k], map_b[k]
        # Applica i filtri (Is Dense/Value?)
        if is_integrator_layer(k, wa.shape) and wa.shape == wb.shape:
            if torch.norm(wa.float() - wb.float()) < 1e-6:
                skipped_identical += 1
                continue
            engine.analyze_layer(wa, wb)
            
    if engine.n_layers == 0: 
        if skipped_identical > 0: return {"error": "All Target Layers are Identical"}
        return {"error": "0 Valid Layers (Check Filter Logic)"}
        
    return engine.get_verdict()

# ==============================================================================
# 4. NAVIGATORE GERARCHICO (Tree Analysis)
# ==============================================================================

def find_model_in_dir(path):
    # Se il path è già un file valido, ritornalo
    if os.path.isfile(path) and path.endswith(('.safetensors', '.bin', '.pt', '.pth')):
        return path
    
    # Se è una directory, cerca il file più grande
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if f.endswith(('.safetensors', '.bin'))]
        if not files: return None
        full_paths = [os.path.join(path, f) for f in files]
        return max(full_paths, key=os.path.getsize)
    
    return None

def print_result(label, res, indent="      "):
    """
    Stampa il risultato.
    label: Descrizione del contesto (es. "L2 Nome (vs Padre)")
    res: Il dizionario dei risultati dell'engine
    """
    if res.get("error"):
        print(f"{indent}❌ {label} | ERROR: {res['error']}")
        return
    
    # Qui stampiamo la direzione rilevata: Padre -> Figlio
    print(f"{indent}✅ {label} | Rilevato: {res['father'][:20]} -> {res['son'][:20]} | Conf: {res['conf']:.1f}% | {res['stats']}")

def run_tree_analysis(root_dir):
    tree_name = os.path.basename(root_dir)
    print(f"\n\n{'='*100}")
    print(f"🌳 ANALISI ALBERO: {tree_name}")
    print(f"{'='*100}")
    
    # --- LIVELLO 0: RADICE ---
    root_model_path = find_model_in_dir(root_dir)
    if not root_model_path: 
        print(f"❌ Errore: Nessun modello radice trovato in {root_dir}")
        return
    
    root_name = os.path.basename(root_model_path)
    print(f"📍 RADICE: {root_name}")

    path_d1 = os.path.join(root_dir, "depth_1")
    path_d2 = os.path.join(root_dir, "depth_2")

    if not os.path.exists(path_d1):
        print("⚠️ Nessuna cartella 'depth_1' trovata.")
        return

    # Trova le cartelle Child_X in depth_1 (i Padri)
    l1_folders = sorted(
        [d for d in os.listdir(path_d1) if os.path.isdir(os.path.join(path_d1, d)) and "Child" in d],
        key=lambda x: x.lower()
    )

    if not l1_folders:
        print("⚠️ Nessuna cartella 'Child' in depth_1.")
        return

    for child_name in l1_folders:
        print(f"\n📂 RAMO: {child_name}")
        
        # =====================================================================
        # LIVELLO 1: Il Padre (depth_1/Child_X) vs RADICE
        # =====================================================================
        l1_full_path = os.path.join(path_d1, child_name)
        model_l1 = find_model_in_dir(l1_full_path)
        
        if not model_l1:
            print(f"   ⚠️ Nessun modello trovato in {l1_full_path}")
            continue

        l1_name = os.path.basename(model_l1)
        
        # Confronto L1 vs ROOT
        res_l1 = compare_models_v2_6(model_l1, root_model_path)
        
        # Stampa con riferimento al confronto
        print_result(f"L1 (vs {root_name})", res_l1, indent="   ")

        # =====================================================================
        # LIVELLO 2: I Nipoti (depth_2/Child_X/) vs PADRE (L1)
        # =====================================================================
        l2_container_path = os.path.join(path_d2, child_name)
        
        if os.path.exists(l2_container_path):
            # Prende tutto il contenuto (cartelle o file)
            l2_items = sorted(os.listdir(l2_container_path))
            
            found_grandchildren = False
            for item_name in l2_items:
                item_path = os.path.join(l2_container_path, item_name)
                
                # Trova modello (gestisce sia cartella che file diretto)
                model_l2 = find_model_in_dir(item_path)

                if model_l2:
                    found_grandchildren = True
                    l2_name = os.path.basename(model_l2)
                    
                    # Confronto L2 vs L1 (Padre)
                    res_l2 = compare_models_v2_6(model_l2, model_l1)
                    
                    # Stampa: mostriamo chi stiamo confrontando
                    print_result(f"L2 {l2_name} (vs {l1_name})", res_l2, indent="      ")
            
            if not found_grandchildren:
                print(f"      ⚠️ Cartella {child_name} in depth_2 vuota o senza modelli.")
        else:
            print(f"      ⚠️ Nessun nipote trovato (manca {l2_container_path})")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Sostituisci con il tuo percorso base corretto
    base_dir = "/home/trabbo/Documents/Universita/BigData/Models_for_Project/MOTHER-LORA-V/"
    extentions = ["Tree_0", "Tree_1", "Tree_2", "Tree_3", "Tree_4"]
    
    print(f"🚀 Avvio Analisi Geometrica Lineage (Target: Dense + Value Only)...")
    
    for t in extentions:
        new_dir = os.path.join(base_dir, t)
        if os.path.exists(new_dir): 
            run_tree_analysis(new_dir)
        else:
            print(f"\n⚠️ Cartella non trovata: {new_dir}")