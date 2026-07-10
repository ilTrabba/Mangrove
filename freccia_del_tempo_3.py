import torch
import numpy as np
from safetensors.torch import load_file
import os
import re
import json
import time

# ==============================================================================
# CONFIGURAZIONE E FILTRO (Dal Primo Script)
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

# ==============================================================================
# GESTIONE FILE E CARTELLE (Dal Secondo Script, adattato per il primo)
# ==============================================================================

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

def load_model_weights(model_path):
    valid_exts = ('.safetensors', '.bin', '.ckpt', '.pt', '.pth')
    if os.path.isfile(model_path):
        return load_chunk(model_path)
    elif os.path.isdir(model_path):
        files = [f for f in os.listdir(model_path) if f.endswith(valid_exts)]
        if not files: return None
        combined_state_dict = {}
        for file in files:
            file_path = os.path.join(model_path, file)
            chunk = load_chunk(file_path)
            if chunk is not None: combined_state_dict.update(chunk)
        return combined_state_dict if combined_state_dict else None
    return None

def find_model_path(tree_dir, model_name, depth=None):
    valid_exts = ('.safetensors', '.bin', '.ckpt', '.pt', '.pth')
    root_file = os.path.join(tree_dir, model_name)
    if os.path.isfile(root_file) and root_file.endswith(valid_exts): return root_file
    
    root_dir = os.path.join(tree_dir, model_name)
    if os.path.isdir(root_dir) and not model_name.startswith("depth_"):
        files = [f for f in os.listdir(root_dir) if f.endswith(valid_exts)]
        if files: return root_dir

    depth_dirs = [f'depth_{depth}'] if (depth is not None and depth > 0) else ['depth_1', 'depth_2', 'depth_3', 'depth_4']
    for depth_dir in depth_dirs:
        depth_path = os.path.join(tree_dir, depth_dir)
        if not os.path.exists(depth_path): continue
        
        model_dir = os.path.join(depth_path, model_name)
        if os.path.isdir(model_dir):
            if any(f.endswith(valid_exts) for f in os.listdir(model_dir)): return model_dir
        
        model_file = os.path.join(depth_path, model_name)
        if os.path.isfile(model_file) and model_file.endswith(valid_exts): return model_file
    return None

def get_model_name(model_path):
    return os.path.basename(model_path)

# ==============================================================================
# ENGINE MATEMATICO DETTAGLIATO (Esattamente come il Primo Script)
# ==============================================================================

class LineageEngineV2_6:
    def __init__(self, name_a, name_b):
        self.name_a = name_a
        self.name_b = name_b
        
        self.votes_a = 0 
        self.votes_b = 0 
        self.n_layers = 0
        self.little_layers = 0
        
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
        
        if norm_delta < 1e-6: 
            self.skipped_identical += 1
            return

        U_A = self._get_U_exact(w_a)
        U_B = self._get_U_exact(w_b)
        
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

def compare_models_v2_6(path_a, path_b, exclude_value=False):
    torch.manual_seed(42)
    np.random.seed(42)
    
    sd_a = load_model_weights(path_a)
    sd_b = load_model_weights(path_b)
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
    
    if engine.n_layers == 0: 
        error_msg = f"0 Layers Analizzati. Passati al filtro: {valid_count} -> Identici: {engine.skipped_identical} -> Falliti SVD: {engine.failed_svd}"
        return {"error": error_msg, "time": calc_time}
    
    verdict = engine.get_verdict()
    
    if not verdict.get("error"):
        verdict["stats"] += f" LayersUsed({valid_count})"
    
    verdict["time"] = calc_time
        
    del sd_a, sd_b, map_a, map_b
    return verdict

# ==============================================================================
# BATCH RUNNER (Estratto dal Secondo Script, stampa come il Primo)
# ==============================================================================

def extract_pairs_from_tree(tree_data):
    pairs = []
    root = tree_data.get("root")
    
    if "d1" in tree_data:
        d1_data = tree_data["d1"]
        if isinstance(d1_data, list):
            for child in d1_data: pairs.append((root, child, 0, 1))
        elif isinstance(d1_data, dict):
            for parent, children in d1_data.items():
                for child in children: pairs.append((parent, child, 0, 1))
    
    for depth in ["d2", "d3", "d4"]:
        if depth in tree_data:
            d_idx = int(depth[-1])
            for parent, children in tree_data[depth].items():
                for child in children: pairs.append((parent, child, d_idx-1, d_idx))
                
    return pairs

def run_family_analysis(tree_dir, ground_truth_json, tree_id=None, exclude_value=False):
    tree_name = os.path.basename(tree_dir)
    print(f"\n{'='*130}")
    print(f"🌲 ANALISI FAMIGLIA DETTAGLIATA: {tree_name} (Exclude Value: {exclude_value})")
    print(f"{'='*130}")
    
    with open(ground_truth_json, 'r') as f:
        gt_data = json.load(f)
    
    trees = gt_data.get("TREES", gt_data)
    normalized_trees = {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in trees.items()}
    
    if tree_id is not None:
        if tree_id not in normalized_trees:
            print(f"❌ Errore: Tree {tree_id} non trovato. Alberi disponibili: {list(normalized_trees.keys())}")
            return
        normalized_trees = {tree_id: normalized_trees[tree_id]}
    
    for tid, tree_data in normalized_trees.items():
        print(f"\n📁 INIZIO ANALISI ALBERO {tid}")
        gt_pairs = extract_pairs_from_tree(tree_data)
        
        for parent_name, child_name, parent_depth, child_depth in gt_pairs:
            print(f"\n\n{'='*130}")
            print(f"🔍 ANALISI COPPIA: {parent_name} (Padre reale) vs {child_name} (Figlio reale)")
            print(f"{'='*130}")
            
            parent_path = find_model_path(tree_dir, parent_name, parent_depth)
            child_path = find_model_path(tree_dir, child_name, child_depth)
            
            if not parent_path or not child_path:
                print(f"❌ File non trovati: Padre ({bool(parent_path)}), Figlio ({bool(child_path)})")
                continue
                
            # Avvia il motore matematico passandoli come Modello A e Modello B
            res = compare_models_v2_6(parent_path, child_path, exclude_value=exclude_value)
            
            if res.get("error"):
                print(f"❌ ERRORE MATEMATICO: {res['error']}")
                continue

            print("-" * 130)
            
            # Valutazione della predizione
            pred_father = res['father']
            is_correct = (pred_father == parent_name) or (pred_father == get_model_name(parent_path))
            status_icon = "✅ CORRETTO" if is_correct else "❌ ERRATO"
            
            # Stampa log finali (Dal 1° Script + validazione)
            print(f"\n🏆 RISULTATO DELLA PREDIZIONE")
            print(f"Modello A: {get_model_name(parent_path)}")
            print(f"Modello B: {get_model_name(child_path)}")
            print(f"Direzione Calcolata : {res['father']} -> {res['son']}")
            print(f"Ground Truth Reale  : {parent_name} -> {child_name}")
            print(f"Esito Test          : {status_icon}")
            print(f"Confidenza          : {res['conf']:.2f}%")
            print(f"Statistiche         : {res['stats']}")
            print(f"Tempo calcolo SVD   : {res['time']:.2f}s")
            print("=" * 130)

if __name__ == "__main__":
    # ==============================================================================
    # CONFIGURAZIONE RUNNER
    # ==============================================================================
    
    TREE_DIRECTORY = "/home/cristian/projects/dataset/QWEN" 
    GROUND_TRUTH = "~/projects/Model_Graph/ground_truth.json"  
    TREE_ID = 1  
    
    TREE_DIRECTORY = os.path.expanduser(TREE_DIRECTORY)
    GROUND_TRUTH = os.path.expanduser(GROUND_TRUTH)
    
    if os.path.exists(TREE_DIRECTORY) and os.path.exists(GROUND_TRUTH):
        # Valuta l'intero albero con i log estesi del primo script
        run_family_analysis(TREE_DIRECTORY, GROUND_TRUTH, tree_id=TREE_ID, exclude_value=False)
    else:
        print(f"Errore: Verifica che esistano:")
        print(f"  - Tree directory: {TREE_DIRECTORY}")
        print(f"  - Ground truth JSON: {GROUND_TRUTH}")