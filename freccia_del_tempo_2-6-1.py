import torch
import numpy as np
from safetensors.torch import load_file
import os
import re
import json
import time  

# ==============================================================================
# CONFIGURAZIONE E FILTRO
# ==============================================================================

def normalize_key(key):
    key = key.replace('model.', '').replace('vit.', '').replace('transformer.', '')
    match = re.search(r'(layers|blocks|h|encoder\.layer|decoder\.layer|layer)\.(\d+)\.(.*)', key)
    if match: return f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
    return key

def is_integrator_layer(key, shape, exclude_value=False):
    if len(shape) not in [2, 4]: 
        return False
        
    key_lower = key.lower()
    
    exclusions = [
        "norm", "embed", "lora", "bias", "time", "bn", "wte", "wpe", 
        "classifier", "pooler", "patch_embeddings",
        "visual", "vision_model", "vit", "merger", "projector" 
    ]
    
    attention_q = ["to_q", "q_proj", ".q.", "query"]
    attention_k = ["to_k", "k_proj", ".k.", "key"]
    exclusions.extend(attention_q)
    exclusions.extend(attention_k)
    
    if exclude_value:
        attention_v = ["to_v", "v_proj", ".v.", "value"]
        exclusions.extend(attention_v)
        
    if any(bad_word in key_lower for bad_word in exclusions):
        return False
        
    return True

def load_chunk(path):
    try:
        if path.endswith(".safetensors"): 
            return load_file(path, device="cpu")
            
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            return ckpt["state_dict"]
            
        return ckpt
    except Exception as e: 
        print(f"Errore nel caricamento di {path}: {e}")
        return None

# ==============================================================================
# ENGINE V2.6.FINAL (CON DIAGNOSTICA AVANZATA)
# ==============================================================================

class LineageEngineV2_6:
    def __init__(self, name_a, name_b):
        self.name_a = name_a
        self.name_b = name_b
        
        self.votes_a = 0 
        self.votes_b = 0 
        self.n_layers = 0
        self.little_layers = 0
        
        # Nuovi contatori diagnostici
        self.skipped_identical = 0
        self.failed_svd = 0
        
        self.layer_details = []
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

    def analyze_layer(self, layer_key, w_a, w_b):
        if w_a.ndim != 2:
            w_a = w_a.view(w_a.shape[0], -1)
            w_b = w_b.view(w_b.shape[0], -1)
            
        w_a, w_b = w_a.float(), w_b.float()
        delta = w_b - w_a
        norm_delta = float(torch.norm(delta, p="fro"))
        
        # Se il delta è microscopico, i modelli sono identici in questo layer
        if norm_delta < 1e-6: 
            self.skipped_identical += 1
            return

        U_A = self._get_U_exact(w_a)
        U_B = self._get_U_exact(w_b)
        
        # Se la SVD fallisce matematicamente
        if U_A is None or U_B is None: 
            self.failed_svd += 1
            return

        ck_A = float(torch.norm(U_A.T @ delta, p="fro")**2) / (norm_delta**2 + self.eps)
        ck_B = float(torch.norm(U_B.T @ delta, p="fro")**2) / (norm_delta**2 + self.eps)

        score = ck_B - ck_A
        winner = "TIE"

        if abs(score) < 1e-4:
            self.little_layers += 1

        if score > 0:
            self.votes_a += 1
            winner = self.name_a
        elif score < 0:
            self.votes_b += 1
            winner = self.name_b

        self.layer_details.append({
            "layer": layer_key,
            "ck_A": ck_A,
            "ck_B": ck_B,
            "score": score,
            "winner": winner
        })

        self.n_layers += 1

    def get_verdict(self):
        # Se non ha analizzato layer, restituisce l'autopsia esatta
        if self.n_layers == 0: 
            return {"error": f"0 Layers Analizzati. Diagnostica: {self.skipped_identical} scartati perché IDENTICI, {self.failed_svd} falliti per errore SVD."}
        
        is_a_father = self.votes_a >= self.votes_b
        
        total = self.votes_a + self.votes_b + self.eps
        ratio = max(self.votes_a, self.votes_b) / total
        conf = (ratio - 0.5) * 200.0
        
        father = self.name_a if is_a_father else self.name_b
        son = self.name_b if is_a_father else self.name_a
        
        stats = f"[Votes: A={self.votes_a} | B={self.votes_b} | Ties={self.little_layers}]"
        
        return {
            "father": father,
            "son": son,
            "conf": min(conf, 99.9),
            "stats": stats,
            "layer_details": self.layer_details,
            "error": None
        }

# ==============================================================================
# GESTIONE MODELLI E CARTELLE
# ==============================================================================

def get_model_name(model_path):
    return os.path.basename(model_path)

# ==============================================================================
# RUNNER
# ==============================================================================

def compare_models_v2_6(path_a, path_b, exclude_value=False):
    torch.manual_seed(42)
    np.random.seed(42)
    
    sd_a = load_chunk(path_a)
    sd_b = load_chunk(path_b)
    if sd_a is None or sd_b is None: return {"error": "Load Fail"}

    map_a = {normalize_key(k): v for k, v in sd_a.items()}
    map_b = {normalize_key(k): v for k, v in sd_b.items()}
    
    name_a = get_model_name(path_a)
    name_b = get_model_name(path_b)
    
    engine = LineageEngineV2_6(name_a, name_b)
    keys = sorted(list(set(map_a.keys()) & set(map_b.keys())))
    
    valid_count = 0
    start_time = time.time()
    
    for k in keys:
        if k in map_a and k in map_b:
            wa, wb = map_a[k], map_b[k]
            if is_integrator_layer(k, wa.shape, exclude_value) and wa.shape == wb.shape:
                valid_count += 1
                engine.analyze_layer(k, wa, wb)
            
    calc_time = time.time() - start_time
    
    # Se i layer sono zero, inseriamo anche i layer che avevano passato il filtro geometrico nel log
    if engine.n_layers == 0: 
        error_msg = f"0 Layers Analizzati. Passati al filtro geometrico: {valid_count} -> Di cui identici (Delta=0): {engine.skipped_identical} -> Di cui falliti SVD: {engine.failed_svd}"
        return {"error": error_msg, "time": calc_time}
    
    verdict = engine.get_verdict()
    
    if not verdict.get("error"):
        verdict["stats"] += f" LayersUsed({valid_count})"
    
    verdict["time"] = calc_time
        
    del sd_a, sd_b, map_a, map_b
    return verdict

# ==============================================================================
# MAIN LOGICS (SINGLE)
# ==============================================================================

def run_single_analysis(path_a, path_b, exclude_value=False):
    print(f"\n{'='*130}")
    print(f"🔍 ANALISI SINGOLA DETTAGLIATA (Exclude Value: {exclude_value})")
    print(f"{'='*130}")

    if not os.path.exists(path_a) or not os.path.exists(path_b):
        print("❌ Errore: Uno dei file/cartelle non esiste. Controlla i percorsi.")
        return

    res = compare_models_v2_6(path_a, path_b, exclude_value=exclude_value)

    if res.get("error"):
        print(f"ERROR: {res['error']}")
        print(f"Tempo esecuzione matematica: {res.get('time', 0):.2f}s")
        return

    name_a = get_model_name(path_a)
    name_b = get_model_name(path_b)

    print(f"\n📑 DETTAGLIO LAYER PER LAYER:")
    print(f"{'NOME LAYER':<50} | {'ck_A':<10} | {'ck_B':<10} | {'SCORE (Δ)':<12} | {'VINCITORE (PADRE)'}")
    print("-" * 130)

    for det in res["layer_details"]:
        layer_name = det["layer"]
        if len(layer_name) > 48:
            layer_name = layer_name[:45] + "..."
            
        print(f"{layer_name:<50} | {det['ck_A']:<10.6f} | {det['ck_B']:<10.6f} | {det['score']:<12.6f} | {det['winner']}")

    print("-" * 130)
    
    print(f"\n🏆 RISULTATO FINALE")
    print(f"Modello A: {name_a}")
    print(f"Modello B: {name_b}")
    print(f"Direzione predetta : {res['father']} -> {res['son']}")
    print(f"Confidenza         : {res['conf']:.2f}%")
    print(f"Statistiche        : {res['stats']}")
    print(f"Tempo calcolo SVD  : {res['time']:.2f}s")
    print("=" * 130 + "\n")

if __name__ == "__main__":
    FILE_A = "/home/cristian/projects/Model_Graph/model_heritage_backend/weights/models/7e9ecd6f-06b9-4f43-82a7-83b4efaede27_gemma-3-1b-it.safetensors"
    FILE_B = "/home/cristian/projects/Model_Graph/model_heritage_backend/weights/models/84816cf9-22f7-4a56-b936-a7d62f6c5f3e_WolfInk_max.safetensors"

    run_single_analysis(FILE_A, FILE_B, exclude_value=False)