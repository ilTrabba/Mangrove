#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# ------------------------------------------------------------------------------
# ⚙️ INTERRUTTORI DI FILTRAGGIO (Accendi con True o Spegni con False)
# ------------------------------------------------------------------------------
USE_INTEGRATORS  = True   # Output che scrivono sul Residual Stream (o_proj, down_proj, etc.)
USE_INTERMEDIATE = True   # Layer di espansione interna (up_proj, gate_proj, fc1, intermediate)
USE_VALUE        = False  # Contenuto dell'Attention (v_proj, value)
USE_ROUTING      = False  # Query e Key dell'Attention (q_proj, k_proj)
# ------------------------------------------------------------------------------

def normalize_key(key):
    for p in ("base_model.model.", "model.", "transformer.", "vit.", "encoder."):
        if key.startswith(p):
            key = key[len(p):]
    match = re.search(r'(layers|blocks|h|encoder\.layer|decoder\.layer|layer)\.(\d+)\.(.*)', key)
    if match: 
        return f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
    return key

def is_target_layer(key, shape, mode="pure"):
    """
    mode='pure': Solo Integrators e Intermediate (Filtro Chirurgico Residual Stream)
    mode='lora_fallback': Allarga l'indagine a Value e Routing
    """
    if len(shape) != 2: return False
    key = key.lower()
    
    if any(x in key for x in ["norm", "embed", "lora", "bias", "bn", "wte", "wpe", "classifier", "pooler", "patch_embeddings"]): 
        return False
    
    # FASE 1: I veri Integrators e Intermediate
    targets = [
        "o_proj", "out_proj", "attention.output.dense", "down_proj", "dense_4h_to_h", "c_proj", "attn.proj", "wo",
        "up_proj", "gate_proj", "fc1", "fc2", "intermediate.dense", "wi", "wi_0", "wi_1", "mlp"
    ]
    
    # FASE 2: Se la Fase 1 fallisce, accendiamo l'attenzione
    if mode == "lora_fallback":
        targets.extend(["v_proj", "value", "to_v", "q_proj", "query", "to_q", "k_proj", "key", "to_k", "c_attn"])

    return any(t in key for t in targets)

def load_chunk(path):
    try:
        if path.endswith(".safetensors"): return load_file(path, device="cpu")
        return torch.load(path, map_location="cpu", weights_only=True)
    except: return None

# ==============================================================================
# 2. ENGINE V9 (Delta Entropy + Delta Alignment)
# ==============================================================================

class LineageEngineV9_DeltaEntropy:
    def __init__(self, name_a, name_b):
        self.name_a = name_a
        self.name_b = name_b
        
        self.host_scores = []       # Freccia dello Spazio (Chi ha la feature)
        self.entropies = []         # Entropia di Delta W (Firma del Regime)
        
        self.skipped_identical = 0
        self.processed = 0
        self.k = 16
        self.eps = 1e-9

    @torch.no_grad()
    def _get_U(self, W):
        mn = min(W.shape)
        if mn <= 1: return None
        try:
            U, _, _ = torch.linalg.svd(W, full_matrices=False)
            return U[:, :self.k]
        except Exception:
            return None

    def analyze_layer(self, w_a, w_b):
        w_a, w_b = w_a.float(), w_b.float()
        delta = w_b - w_a
        
        norm_delta = float(torch.norm(delta, p="fro")) + self.eps
        
        if norm_delta < 1e-6:
            self.skipped_identical += 1
            return

        # --- 1. FIRMA DEL REGIME (SVD su Delta W) ---
        try:
            _, S_delta, _ = torch.linalg.svd(delta, full_matrices=False)
            k_delta = min(self.k, S_delta.shape[0])
            S_k = S_delta[:k_delta]
            
            # Normalizziamo in distribuzione di probabilità
            p = S_k / (S_k.sum() + self.eps)
            # Calcolo Entropia di Shannon normalizzata (tra 0 e 1)
            H_delta = float(-torch.sum(p * torch.log(p + self.eps))) / np.log(k_delta + self.eps)
            self.entropies.append(H_delta)
        except Exception:
            pass # Se SVD fallisce, saltiamo l'entropia per questo layer

        # --- 2. IDENTIFICAZIONE HOST (SVD sui Modelli A e B) ---
        U_A = self._get_U(w_a)
        U_B = self._get_U(w_b)
        if U_A is None or U_B is None: return

        ck_A = float(torch.norm(U_A.T @ delta, p="fro")) / norm_delta
        ck_B = float(torch.norm(U_B.T @ delta, p="fro")) / norm_delta
        
        # > 0 indica che B ospita il Delta
        self.host_scores.append(ck_B - ck_A)
        self.processed += 1

    def get_verdict(self):
        if self.processed == 0:
            if self.skipped_identical > 0:
                return {"error": f"Trovate {self.skipped_identical} matrici identiche."}
            return {"error": "Nessuna matrice analizzata."}
            
        h_scores = np.array(self.host_scores)
        
        # 1. Chi ospita il Delta?
        is_B_host = h_scores.sum() > 0
        host = self.name_b if is_B_host else self.name_a
        
        # 2. Qual è il Regime? (Media dell'Entropia di Delta W)
        mean_entropy = np.mean(self.entropies) if self.entropies else 0.0
        
        # SOGLIE DA TARARE SUI TUOI DATI EMPIRICI:
        # > 0.85 = Rumore diffuso (Generalizzazione / CPT)
        # < 0.65 = Picco strutturato (Sottrattivo / RLHF)
        # Altrimenti = Additivo / Instruct SFT
        if mean_entropy > 0.85:
            regime = "GENERALIZZAZIONE"
        elif mean_entropy < 0.65:
            regime = "SOTTRATTIVO"
        else:
            regime = "ADDITIVO"
            
        # 3. Deduzione della Parentela (Incrocio logico)
        if regime in ["GENERALIZZAZIONE", "ADDITIVO"]:
            # La feature è stata aggiunta. L'Host è il Figlio.
            son = host
            father = self.name_a if son == self.name_b else self.name_b
        else:
            # La feature è stata tolta. L'Host è il Padre (la possedeva prima).
            father = host
            son = self.name_a if father == self.name_b else self.name_b
            
        conf = float((h_scores > 0).mean() * 100.0) if is_B_host else float((h_scores < 0).mean() * 100.0)
        
        stats = f"Mats({self.processed})"
        return {
            "father": father, 
            "son": son, 
            "conf": min(conf, 99.9), 
            "stats": stats, 
            "error": None, 
            "host": host, 
            "regime": regime,
            "entropy": mean_entropy
        }

# ==============================================================================
# 3. INTERFACCIA E CONFRONTO
# ==============================================================================

def compare_models(path_a, path_b):
    sd_a = load_chunk(path_a)
    sd_b = load_chunk(path_b)
    if sd_a is None or sd_b is None: return {"error": "Load Fail: File non trovato o corrotto"}

    map_a = {normalize_key(k): v for k, v in sd_a.items()}
    map_b = {normalize_key(k): v for k, v in sd_b.items()}
    
    keys = sorted(list(set(map_a.keys()) & set(map_b.keys())))
    if len(keys) == 0: return {"error": "Key Mismatch (Architetture diverse o nomi incompatibili)"}

    # FASE 1: Scansione PURA
    torch.manual_seed(42)
    np.random.seed(42)
    
    engine_pure = LineageEngineV9_DeltaEntropy(os.path.basename(path_a), os.path.basename(path_b))
    
    for k in keys:
        wa, wb = map_a.get(k), map_b.get(k)
        if wa is not None and wb is not None and is_target_layer(k, wa.shape, mode="pure") and wa.shape == wb.shape:
            engine_pure.analyze_layer(wa, wb)
            
    if engine_pure.processed > 0:
        res = engine_pure.get_verdict()
        res['stats'] = "[FULL] " + res['stats']
        del sd_a, sd_b, map_a, map_b
        gc.collect()
        return res

    # FASE 2: Scansione FALLBACK
    torch.manual_seed(42) 
    np.random.seed(42)
    
    engine_fallback = LineageEngineV9_DeltaEntropy(os.path.basename(path_a), os.path.basename(path_b))
    
    for k in keys:
        wa, wb = map_a.get(k), map_b.get(k)
        if wa is not None and wb is not None and is_target_layer(k, wa.shape, mode="lora_fallback") and wa.shape == wb.shape:
            engine_fallback.analyze_layer(wa, wb)
            
    res = engine_fallback.get_verdict()
    
    if res.get("error") and "Trovate" in res.get("error"):
        res['error'] = res['error'].replace("Trovate", "Fase 2 fallita. Trovate").replace("Nessun gradiente.", "Nessun gradiente.")
    elif not res.get("error"):
        res['stats'] = "[PEFT] " + res['stats']
        
    del sd_a, sd_b, map_a, map_b
    gc.collect()
    
    return res

# ==============================================================================
# 4. NAVIGAZIONE UI E STAMPE
# ==============================================================================

def find_checkpoint(path):
    if os.path.isfile(path) and path.endswith(('.safetensors', '.bin', '.pt', '.pth')): return path
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if f.endswith(('.safetensors', '.bin'))]
        if not files: return None
        full_paths = [os.path.join(path, f) for f in files]
        return max(full_paths, key=os.path.getsize)
    return None

def clean_name(name):
    return name.replace(".safetensors", "").replace(".bin", "")[:26]

def print_result_row(p_path, res):
    p_name_clean = clean_name(os.path.basename(p_path))
    
    if res.get("error"):
        print(f"{p_name_clean:<26} | ERROR: {res['error']}")
        return False
        
    f_name_clean = clean_name(res['father'])
    s_name_clean = clean_name(res['son'])
    host_clean = clean_name(res['host'])
    
    is_correct = (res['father'] == os.path.basename(p_path))
    status = "✅" if is_correct else "❌"
    lineage = f"{f_name_clean}->{s_name_clean}"
    
    entropy_str = f"H={res['entropy']:.3f}"
    
    print(f"{p_name_clean:<26} | {lineage:<46} | {res['conf']:>3.0f}% | {status} {res['regime']:<16} | {entropy_str:<8} | Host: {host_clean}")
    return is_correct

def print_active_layers():
    print(f"\n{'='*140}")
    print(f"⚙️  CONFIGURAZIONE LAYER ATTIVI:")
    print(f"   - INTEGRATORS  : {'✅' if USE_INTEGRATORS else '❌'} (es. o_proj, down_proj, out_proj)")
    print(f"   - INTERMEDIATE : {'✅' if USE_INTERMEDIATE else '❌'} (es. up_proj, gate_proj, fc1, wi)")
    print(f"   - VALUE        : {'✅' if USE_VALUE else '❌'} (es. v_proj, value)")
    print(f"   - ROUTING      : {'✅' if USE_ROUTING else '❌'} (es. q_proj, k_proj, query, key)")
    print(f"{'='*140}")

# --- OPZIONE 1: SINGOLA ---
def run_single_comparison(path_a, path_b):
    print(f"\n{'='*140}")
    print(f"🔍 CONFRONTO DIRETTO: A vs B")
    print(f"{'='*140}")
    
    file_a = find_checkpoint(path_a)
    file_b = find_checkpoint(path_b)
    
    if not file_a or not file_b:
        print("❌ Errore: Uno dei due percorsi non contiene file validi.")
        return

    name_a = clean_name(os.path.basename(file_a))
    name_b = clean_name(os.path.basename(file_b))

    print(f"{'PADRE (GT O IPOTESI)':<26} | {'PREDIZIONE: PADRE -> FIGLIO':<46} | {'CONF':<4} | {'STATO REGIME':<18} | {'ENTROPIA':<8} | {'DETTAGLI (HOST)'}")
    print("-" * 140)

    res = compare_models(file_a, file_b)
    
    if res.get("error"):
        print(f"ERROR: {res['error']}")
        return
        
    f_name_clean = clean_name(res['father'])
    s_name_clean = clean_name(res['son'])
    host_clean = clean_name(res['host'])
    lineage = f"{f_name_clean} -> {s_name_clean}"
    entropy_str = f"H={res['entropy']:.3f}"
    
    print(f"{name_a:<26} | {lineage:<46} | {res['conf']:>3.0f}% | - {res['regime']:<16} | {entropy_str:<8} | Host: {host_clean}")
    print(f"\n   Dettagli: {res['stats']}")
    print("-" * 140)

# --- OPZIONE 2: ALBERO STANDARD ---
def run_tree_analysis(root_dir):
    tree_name = os.path.basename(root_dir)
    print(f"\n{'='*140}")
    print(f"🌲 TREE ANALYSIS | STRUTTURA STANDARD: {tree_name}")
    print(f"{'='*140}")

    root_model_path = find_checkpoint(root_dir)
    if not root_model_path: 
        print(f"Radice non trovata in {root_dir}")
        return

    gt_pairs = []
    path_d1 = os.path.join(root_dir, "depth_1")
    path_d2 = os.path.join(root_dir, "depth_2")

    if not os.path.exists(path_d1):
        print("⚠️ Nessuna cartella 'depth_1' trovata.")
        return

    l1_folders = sorted(
        [d for d in os.listdir(path_d1) if os.path.isdir(os.path.join(path_d1, d)) and "Child" in d],
        key=lambda x: x.lower()
    )

    for child_name in l1_folders:
        l1_full_path = os.path.join(path_d1, child_name)
        model_l1 = find_checkpoint(l1_full_path)
        
        if not model_l1: continue
        gt_pairs.append((root_model_path, model_l1))

        l2_container_path = os.path.join(path_d2, child_name)
        if os.path.exists(l2_container_path):
            l2_items = sorted(os.listdir(l2_container_path))
            for item_name in l2_items:
                item_path = os.path.join(l2_container_path, item_name)
                model_l2 = find_checkpoint(item_path)
                if model_l2:
                    gt_pairs.append((model_l1, model_l2))

    if not gt_pairs:
        print("Nessuna coppia trovata da confrontare nell'albero.")
        return

    print(f"{'PADRE (GT)':<26} | {'PREDIZIONE: PADRE -> FIGLIO':<46} | {'CONF':<4} | {'STATO REGIME':<18} | {'ENTROPIA':<8} | {'DETTAGLI (HOST)'}")
    print("-" * 140)

    passed = 0
    for p_path, c_path in gt_pairs:
        res = compare_models(p_path, c_path) 
        if print_result_row(p_path, res): passed += 1
        
    acc = (passed / len(gt_pairs)) * 100 if gt_pairs else 0
    print("-" * 140)
    print(f"📊 ACCURACY ({tree_name}): {acc:.2f}%\n")

# --- OPZIONE 3: ALBERO ANNIDATO PROFONDO ---
def run_nested_tree_analysis(root_dir):
    tree_name = os.path.basename(root_dir)
    print(f"\n{'='*140}")
    print(f"🌳 NESTED TREE ANALYSIS | STRUTTURA PROFONDA: {tree_name}")
    print(f"{'='*140}")

    root_model_path = find_checkpoint(root_dir)
    if not root_model_path: 
        print(f"Radice non trovata in {root_dir}")
        return

    gt_pairs = []
    
    try:
        child_folders = sorted(
            [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("Child")],
            key=lambda x: x.lower()
        )
    except Exception as e:
        print(f"Errore nella lettura di {root_dir}: {e}")
        return

    for child_name in child_folders:
        child_dir = os.path.join(root_dir, child_name)
        model_l1 = find_checkpoint(child_dir)
        
        if not model_l1:
            continue

        gt_pairs.append((root_model_path, model_l1))

        current_parent_model = model_l1
        current_dir = child_dir
        current_depth = 2
        
        while True:
            next_dir = os.path.join(current_dir, f"depth_{current_depth}")
            if not os.path.exists(next_dir):
                break
                
            next_model = find_checkpoint(next_dir)
            if next_model:
                gt_pairs.append((current_parent_model, next_model))
                current_parent_model = next_model
            
            current_dir = next_dir
            current_depth += 1

    if not gt_pairs:
        print("Nessuna coppia trovata da confrontare nell'albero.")
        return

    print(f"{'PADRE (GT)':<26} | {'PREDIZIONE: PADRE -> FIGLIO':<46} | {'CONF':<4} | {'STATO REGIME':<18} | {'ENTROPIA':<8} | {'DETTAGLI (HOST)'}")
    print("-" * 140)

    passed = 0
    for p_path, c_path in gt_pairs:
        res = compare_models(p_path, c_path)
        if print_result_row(p_path, res): passed += 1
        
    acc = (passed / len(gt_pairs)) * 100 if gt_pairs else 0
    print("-" * 140)
    print(f"📊 ACCURACY ({tree_name}): {acc:.2f}%\n")


if __name__ == "__main__":
    
    print_active_layers()
    
    # =========================================================================
    # ⚙️ SCEGLI LA MODALITÀ DI ESECUZIONE 
    # =========================================================================
    
    MODE = "SINGLE"  
    
    if MODE == "SINGLE":
        modello_1 = "/home/trabbo/Downloads/Qwen2.5-0.5B-family/Qwen2.5-0.5B.safetensors"
        modello_2 = "/home/trabbo/Downloads/Qwen2.5-0.5B-family/depth_1/Child_1/Qwen2.5-0.5B-Instruct.safetensors"
        
        run_single_comparison(modello_1, modello_2)

    elif MODE == "TREE":
        base_dir = "/home/trabbo/Documents/Universita/BigData/Models_for_Project/MoTHer"
        trees = ["Tree_0", "Tree_1", "Tree_2", "Tree_3", "Tree_4"]
        
        for t in trees:
            tree_dir = os.path.join(base_dir, t)
            if os.path.exists(tree_dir): 
                run_tree_analysis(tree_dir)

    elif MODE == "NESTED_TREE":
        base_dir = "/home/trabbo/Documents/Universita/BigData/Models_for_Project/WHISPER-SMALL"
        trees = ["Tree_0","Tree_1","Tree_2","Tree_3","Tree_4"]
        
        for t in trees:
            tree_dir = os.path.join(base_dir, t)
            if os.path.exists(tree_dir): 
                run_nested_tree_analysis(tree_dir)