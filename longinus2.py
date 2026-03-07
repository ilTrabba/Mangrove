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
USE_VALUE        = False   # Contenuto dell'Attention (v_proj, value)
USE_ROUTING      = False  # Query e Key dell'Attention (q_proj, k_proj)
# ------------------------------------------------------------------------------

# ==============================================================================
# 1. CONFIGURAZIONE & FILTRI DINAMICI (MULTI-STAGE)
# ==============================================================================

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
    
    # Esclusioni fisse (Mai analizzare questa roba)
    if any(x in key for x in ["norm", "embed", "lora", "bias", "bn", "wte", "wpe", "classifier", "pooler", "patch_embeddings"]): 
        return False
    
    # FASE 1: I veri Integrators e Intermediate (Ripristinati esattamente dal test al 92%)
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
# 2. ENGINE V2.6.FINAL (Geometria Continua dell'Accumulo)
# ==============================================================================

class LineageEngineV2_6_1:
    def __init__(self, name_a, name_b):
        self.name_a = name_a
        self.name_b = name_b
        
        self.contribs = [] 
        self.skipped_identical = 0
        self.processed = 0
        self.k = 16
        self.eps = 1e-9

    @torch.no_grad()
    def _get_U_exact(self, W):
        mn = min(W.shape)
        if mn <= 1: return None
        k = min(self.k, mn - 1)
        
        try:
            U, _, _ = torch.linalg.svd(W, full_matrices=False)
            return U[:, :k]
        except Exception:
            return None 

    def analyze_layer(self, w_a, w_b):
        w_a, w_b = w_a.float(), w_b.float()
        delta = w_b - w_a
        norm_delta = float(torch.norm(delta, p="fro"))
        
        if norm_delta < 1e-6:
            self.skipped_identical += 1
            return

        U_A = self._get_U_exact(w_a)
        U_B = self._get_U_exact(w_b)
        if U_A is None or U_B is None: return

        # Calcolo dell'energia geometrica pura
        ck_A = float(torch.norm(U_A.T @ delta, p="fro")**2) / (norm_delta**2 + self.eps)
        ck_B = float(torch.norm(U_B.T @ delta, p="fro")**2) / (norm_delta**2 + self.eps)

        # La proiezione continua stabilisce chi possiede le feature del Delta
        score = ck_B - ck_A

        # Micro-Tie-Breaker Termodinamico per i checkpoint CPT ravvicinati
        if abs(score) < 1e-4:
            vol_A = float(torch.norm(w_a, p="fro"))
            vol_B = float(torch.norm(w_b, p="fro"))
            score += (vol_B - vol_A) / (max(vol_A, vol_B) + self.eps) * 1e-5

        self.contribs.append(score)
        self.processed += 1

    def get_verdict(self):
        if self.processed == 0:
            if self.skipped_identical > 0:
                return {"error": f"Trovate {self.skipped_identical} matrici identiche. Nessun gradiente."}
            return {"error": "Nessuna matrice analizzata."}
            
        xs = np.array(self.contribs, dtype=np.float64)
        n = len(xs)
        iters = 400
        m = max(1, int(round(0.75 * n)))
        
        rng = np.random.default_rng(42)
        signs = np.empty(iters, dtype=np.int8)
        
        for i in range(iters):
            idx = rng.integers(0, n, size=m)
            s = xs[idx].sum()
            signs[i] = 1 if s > 0 else 0
            
        p_A = float(signs.mean())
        conf = float(abs(p_A - 0.5) * 2.0 * 100.0)
        
        father = self.name_a if p_A >= 0.5 else self.name_b
        son = self.name_b if p_A >= 0.5 else self.name_a
        
        stats = f"Mats({self.processed}) Logic: Direzione di Accumulo"
        return {"father": father, "son": son, "conf": min(conf, 99.9), "stats": stats, "error": None}

# ==============================================================================
# 3. INTERFACCIA E CONFRONTO
# ==============================================================================

# ==============================================================================
# 3. INTERFACCIA E CONFRONTO (LOGICA MULTI-STAGE)
# ==============================================================================

def compare_models(path_a, path_b):
    sd_a = load_chunk(path_a)
    sd_b = load_chunk(path_b)
    if sd_a is None or sd_b is None: return {"error": "Load Fail: File non trovato o corrotto"}

    map_a = {normalize_key(k): v for k, v in sd_a.items()}
    map_b = {normalize_key(k): v for k, v in sd_b.items()}
    
    # FIX DETERMINISMO: Ordiniamo le chiavi per garantire che i layer vengano
    # processati sempre nello stesso identico ordine.
    keys = sorted(list(set(map_a.keys()) & set(map_b.keys())))
    if len(keys) == 0: return {"error": "Key Mismatch (Architetture diverse o nomi incompatibili)"}

    # ---------------------------------------------------------
    # FASE 1: Scansione PURA (Solo Residual Stream)
    # ---------------------------------------------------------
    # Blocchiamo il seed prima di iniziare l'estrazione PCA
    torch.manual_seed(42)
    np.random.seed(42)
    
    engine_pure = LineageEngineV2_6_1(os.path.basename(path_a), os.path.basename(path_b))
    
    for k in keys:
        wa, wb = map_a.get(k), map_b.get(k)
        if wa is not None and wb is not None and is_target_layer(k, wa.shape, mode="pure") and wa.shape == wb.shape:
            engine_pure.analyze_layer(wa, wb)
            
    # Se abbiamo trovato pesi attivi, è un Full Fine-Tuning. Ci fermiamo qui.
    if engine_pure.processed > 0:
        res = engine_pure.get_verdict()
        res['stats'] = "[FULL] " + res['stats']
        del sd_a, sd_b, map_a, map_b
        gc.collect()
        return res

    # ---------------------------------------------------------
    # FASE 2: Scansione FALLBACK (Attivazione Value/Routing)
    # ---------------------------------------------------------
    # Nessun gradiente trovato sugli Integrators. Probabile PEFT/LoRA!
    torch.manual_seed(42) 
    np.random.seed(42)
    
    engine_fallback = LineageEngineV2_6_1(os.path.basename(path_a), os.path.basename(path_b))
    
    for k in keys:
        wa, wb = map_a.get(k), map_b.get(k)
        if wa is not None and wb is not None and is_target_layer(k, wa.shape, mode="lora_fallback") and wa.shape == wb.shape:
            engine_fallback.analyze_layer(wa, wb)
            
    res = engine_fallback.get_verdict()
    
    # Formattazione intelligente del risultato
    if res.get("error") and "Trovate" in res.get("error"):
        res['error'] = res['error'].replace("Trovate", "Fase 2 (Fallback) fallita. Trovate").replace("Nessun gradiente.", "Nessun gradiente nei layer Lineari/Attention (PEFT estremo).")
    elif not res.get("error"):
        res['stats'] = "[PEFT] " + res['stats']
        
    del sd_a, sd_b, map_a, map_b
    gc.collect()
    
    return res

# ==============================================================================
# 4. NAVIGAZIONE UI (ALBERO E SINGOLO)
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
    return name.replace(".safetensors", "").replace(".bin", "")[:28]

def print_result_row(p_path, res):
    """Funzione helper per stampare i risultati formattati in modo pulito."""
    p_name_clean = clean_name(os.path.basename(p_path))
    
    if res.get("error"):
        print(f"{p_name_clean:<28} | ERROR: {res['error']}")
        return False
        
    f_name_clean = clean_name(res['father'])
    s_name_clean = clean_name(res['son'])
    
    is_correct = (res['father'] == os.path.basename(p_path))
    status = "✅" if is_correct else "❌"
    lineage = f"{f_name_clean}->{s_name_clean}"
    
    print(f"{p_name_clean:<28} | {lineage:<48} | {res['conf']:>4.0f}% | {status} {res['stats']}")
    return is_correct

def print_active_layers():
    """Stampa un riepilogo visuale dei layer che verranno utilizzati nell'analisi."""
    print(f"\n{'='*130}")
    print(f"⚙️  CONFIGURAZIONE LAYER ATTIVI:")
    print(f"   - INTEGRATORS  : {'✅' if USE_INTEGRATORS else '❌'} (es. o_proj, down_proj, out_proj)")
    print(f"   - INTERMEDIATE : {'✅' if USE_INTERMEDIATE else '❌'} (es. up_proj, gate_proj, fc1, wi)")
    print(f"   - VALUE        : {'✅' if USE_VALUE else '❌'} (es. v_proj, value)")
    print(f"   - ROUTING      : {'✅' if USE_ROUTING else '❌'} (es. q_proj, k_proj, query, key)")
    print(f"{'='*130}")

# --- OPZIONE 1: SINGOLA ---
def run_single_comparison(path_a, path_b):
    print(f"\n{'='*130}")
    print(f"🔍 CONFRONTO DIRETTO: A vs B")
    print(f"{'='*130}")
    
    file_a = find_checkpoint(path_a)
    file_b = find_checkpoint(path_b)
    
    if not file_a or not file_b:
        print("❌ Errore: Uno dei due percorsi non contiene file validi.")
        return

    name_a = clean_name(os.path.basename(file_a))
    name_b = clean_name(os.path.basename(file_b))

    print(f"{'MODELLO A':<28} | {'MODELLO B':<28} | {'PREDIZIONE P -> F':<42} | {'CONF':<5}")
    print("-" * 130)

    res = compare_models(file_a, file_b)
    
    if res.get("error"):
        print(f"ERROR: {res['error']}")
        return
        
    f_name_clean = clean_name(res['father'])
    s_name_clean = clean_name(res['son'])
    lineage = f"{f_name_clean} -> {s_name_clean}"
    
    print(f"{name_a:<28} | {name_b:<28} | {lineage:<42} | {res['conf']:>4.0f}%")
    print(f"\n   Dettagli: {res['stats']}")
    print("-" * 130)

# --- OPZIONE 2: ALBERO STANDARD (depth_1/Child_X e depth_2/Child_X) ---
def run_tree_analysis(root_dir):
    tree_name = os.path.basename(root_dir)
    print(f"\n{'='*130}")
    print(f"🌲 TREE ANALYSIS | STRUTTURA STANDARD: {tree_name}")
    print(f"{'='*130}")

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

    print(f"{'PADRE (GT)':<28} | {'PRED: P -> F':<48} | {'CONF':<5} | {'ENGINE STATS'}")
    print("-" * 130)

    passed = 0
    for p_path, c_path in gt_pairs:
        res = compare_models(p_path, c_path) 
        if print_result_row(p_path, res): passed += 1
        
    acc = (passed / len(gt_pairs)) * 100 if gt_pairs else 0
    print("-" * 130)
    print(f"📊 ACCURACY ({tree_name}): {acc:.2f}%\n")

# --- OPZIONE 3: ALBERO ANNIDATO PROFONDO (Child_X/depth_2/depth_3...) ---
def run_nested_tree_analysis(root_dir):
    tree_name = os.path.basename(root_dir)
    print(f"\n{'='*130}")
    print(f"🌳 NESTED TREE ANALYSIS | STRUTTURA PROFONDA: {tree_name}")
    print(f"{'='*130}")

    root_model_path = find_checkpoint(root_dir)
    if not root_model_path: 
        print(f"Radice non trovata in {root_dir}")
        return

    gt_pairs = []
    
    # Trova tutte le cartelle che iniziano con "Child" direttamente nella root
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
            print(f"⚠️ Nessun modello trovato in {child_dir}")
            continue

        # Coppia Radice -> Livello 1
        gt_pairs.append((root_model_path, model_l1))

        # Esplorazione ricorsiva/iterativa delle profondità (depth_2, depth_3, ...)
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
            
            # Avanza alla cartella successiva
            current_dir = next_dir
            current_depth += 1

    if not gt_pairs:
        print("Nessuna coppia trovata da confrontare nell'albero.")
        return

    print(f"{'PADRE (GT)':<28} | {'PRED: P -> F':<48} | {'CONF':<5} | {'ENGINE STATS'}")
    print("-" * 130)

    passed = 0
    for p_path, c_path in gt_pairs:
        res = compare_models(p_path, c_path)
        if print_result_row(p_path, res): passed += 1
        
    acc = (passed / len(gt_pairs)) * 100 if gt_pairs else 0
    print("-" * 130)
    print(f"📊 ACCURACY ({tree_name}): {acc:.2f}%\n")


if __name__ == "__main__":
    
    # Stampa iniziale della configurazione dei layer scelti
    print_active_layers()
    
    # =========================================================================
    # ⚙️ SCEGLI LA MODALITÀ DI ESECUZIONE 
    # =========================================================================
    # SINGLE      -> Confronto diretto tra 2 file
    # TREE        -> Struttura: root, depth_1/Child_X, depth_2/Child_X
    # NESTED_TREE -> Struttura: root, Child_X/depth_2/depth_3...
    # =========================================================================
    
    MODE = "SINGLE"  
    
    if MODE == "SINGLE":
        # 🟢 MODALITÀ 1: Confronto diretto
        modello_1 = "/home/trabbo/Downloads/Qwen2.5-0.5B-family/Qwen2.5-0.5B.safetensors"
        modello_2 = "/home/trabbo/Downloads/Qwen2.5-0.5B-family/depth_1/Child_1/Qwen2.5-0.5B-Instruct.safetensors"
        
        run_single_comparison(modello_1, modello_2)

    elif MODE == "TREE":
        # 🔵 MODALITÀ 2: Albero Standard (cartelle separate depth_1, depth_2)
        base_dir = "/home/trabbo/Documents/Universita/BigData/Models_for_Project/MoTHer"
        trees = ["Tree_0", "Tree_1", "Tree_2", "Tree_3", "Tree_4"]
        
        for t in trees:
            tree_dir = os.path.join(base_dir, t)
            if os.path.exists(tree_dir): 
                run_tree_analysis(tree_dir)

    elif MODE == "NESTED_TREE":
        # 🟣 MODALITÀ 3: Albero Annidato (Child_1/depth_2/depth_3)
        base_dir = "/home/trabbo/Documents/Universita/BigData/Models_for_Project/WHISPER-SMALL"
        trees = ["Tree_0","Tree_1","Tree_2","Tree_3","Tree_4"]
        trees=[""]
        
        for t in trees:
            tree_dir = os.path.join(base_dir, t)
            if os.path.exists(tree_dir): 
                run_nested_tree_analysis(tree_dir)