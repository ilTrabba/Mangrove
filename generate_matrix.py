import torch
import numpy as np
import pandas as pd
import os
import logging
from tqdm import tqdm
from typing import Dict, Any, List

# ==============================================================================
# ⚙️ CONFIGURAZIONE
# ==============================================================================
MODELS_DIR = "/home/cristian/projects/Model_Graph/model_heritage_backend/weights/models"
OUTPUT_CSV = "lineage_matrix_final_v8_4_resnet50.csv"
DEVICE = "cpu"
# ==============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

class MatrixGenerator:
    def __init__(self, device="cpu"):
        self.device = device
        self.BACKBONE_KEYWORDS = ["layers", "blocks", "h.", "transformer.h", "encoder", "decoder"]
        self.EXCLUDE_KEYWORDS = ["lm_head", "embed", "token", "norm", "ln_", "bn_", "bias", "classifier", "output", "final", "wte", "wpe", "rotary", "lora"]

    def load_weights(self, path: str) -> Dict[str, torch.Tensor]:
        try:
            from safetensors.torch import load_file
            return load_file(path, device=self.device)
        except: return None

    def _clean_layer_name(self, name: str) -> str:
        prefixes = ["module.", "base_model.model.", "base_model.", "model.", "_orig_mod."]
        for p in prefixes:
            if name.startswith(p): name = name[len(p):]
        return name

    def get_strategic_layers(self, w1, w2):
        def is_valid(n, t):
            nl = n.lower()
            return (any(k in nl for k in self.BACKBONE_KEYWORDS) and 
                    not any(k in nl for k in self.EXCLUDE_KEYWORDS) and 
                    len(t.shape) == 2 and t.shape[0] == t.shape[1])

        m1 = {self._clean_layer_name(k): v for k, v in w1.items() if is_valid(self._clean_layer_name(k), v)}
        m2 = {self._clean_layer_name(k): v for k, v in w2.items() if is_valid(self._clean_layer_name(k), v)}
        common = set(m1.keys()) & set(m2.keys())
        return [(k, m1[k], m2[k]) for k in sorted(list(common)) if m1[k].shape == m2[k].shape]

    def compute_metrics(self, w1, w2):
        pairs = self.get_strategic_layers(w1, w2)
        if not pairs: return None

        cos_list, sign_list, bcs_list = [], [], []
        drank_list, dist_list, ler_list = [], [], []

        for _, t1, t2 in pairs:
            t1, t2 = t1.float(), t2.float()
            v1, v2 = t1.view(-1), t2.view(-1)
            norm1, norm2 = torch.norm(v1), torch.norm(v2)
            if norm1 == 0 or norm2 == 0: cos = 0.0
            else: cos = torch.dot(v1, v2) / (norm1 * norm2)
            cos = max(-1.0, min(1.0, cos.item()))
            sign = (torch.sign(t1) == torch.sign(t2)).float().mean().item()
            bcs = max(0.0, 1.0 - cos) * (max(0.0, 1.0 - sign) ** 2) * 1000

            delta = t1 - t2
            frob_sq = torch.sum(delta ** 2).item()
            t1_en = torch.sum(t1 ** 2).item()
            rel_dist = np.sqrt(frob_sq / (t1_en + 1e-9))
            
            try: spec_norm = torch.linalg.matrix_norm(delta, ord=2).item()
            except: spec_norm = 0.0
            
            if spec_norm > 1e-9:
                stable_rank = frob_sq / (spec_norm ** 2)
                norm_rank = stable_rank / min(t1.shape)
            else: norm_rank = 0.0
            
            ler = rel_dist / (norm_rank + 1e-9)

            cos_list.append(cos)
            sign_list.append(sign)
            bcs_list.append(bcs)
            drank_list.append(norm_rank)
            dist_list.append(rel_dist)
            ler_list.append(ler)

        return {
            "Cosine_Sim": np.mean(cos_list),
            "Sign_Agreement": np.mean(sign_list),
            "BCS_Score": np.median(bcs_list),
            "Relative_Dist": np.mean(dist_list),
            "Delta_Rank": np.mean(drank_list),
            "LER_Score": np.mean(ler_list)
        }

    def classify_relation(self, m):
        """
        Logica v8.4: "Multi-Scale Clustering"
        Gestisce sia Micro-tuning (Twitter) che Macro-tuning (Roberta) che Standard (T0/T1).
        """
        sign = m["Sign_Agreement"]
        bcs = m["BCS_Score"]
        drank = m["Delta_Rank"]
        ler = m["LER_Score"]
        dist = m["Relative_Dist"]

        # 1. BASIN WALL (Separazione Bacini)
        # Twitter vs Roberta è > 15%. T0 vs T1 è < 8%.
        # Alziamo il muro a 12% per sicurezza su Roberta, ma T0/T1 richiede rank.
        if dist > 0.12 or sign < 0.60 or bcs > 5.0:
            return "NO (Estranei)", "N/A"

        is_same_basin = "SI (Bacino)"
        
        # 2. CORSIA MICRO (Twitter Family)
        # Se dist < 2%, sono parenti stretti (anche se Rank è rumoroso).
        if dist < 0.02:
            return is_same_basin, "PARENT-CHILD (Micro)"

        # 3. CORSIA MACRO (Roberta Family & T2 Efficient)
        # Se dist > 5.5%, sono parenti "Lontani ma Coerenti".
        # Roberta->STSB: Dist 0.06, Rank 0.027, LER ~2.2 -> PASSA.
        # T0->T1 Intrusi: Dist 0.037 (troppo bassa per qui).
        if dist > 0.055:
            # LER < 3.2 è la chiave. I cugini lontani hanno LER > 4 o 5.
            if ler < 3.2 and drank < 0.035:
                return is_same_basin, "PARENT-CHILD (Macro)"
            # T2 Efficient: LER alto
            if ler > 5.0 and drank < 0.015:
                return is_same_basin, "PARENT-CHILD (Efficient)"

        # 4. CORSIA STANDARD (T0/T1 - The Danger Zone)
        # Qui (Dist 2% - 5.5%) vivono sia i figli T0 che gli intrusi T1.
        # Dobbiamo essere severi.
        
        # Rank standard
        if drank < 0.010: # Molto stretto
            return is_same_basin, "PARENT-CHILD (Standard)"

        # Rescue Chirurgico (per figli T0 persi)
        # I figli persi hanno Dist ~0.040, LER ~1.9.
        # Gli intrusi hanno Dist ~0.037, LER ~3.0.
        if dist > 0.02 and dist < 0.05:
            if ler < 2.5: # Questo blocca l'intruso a 3.0
                return is_same_basin, "PARENT-CHILD (Rescued)"

        # Se fallisce tutto
        return is_same_basin, "SIBLINGS (Cross-Task)"

def clean_filename(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    if "_" in base: return base.split("_", 1)[1]
    return base

def main():
    if not os.path.exists(MODELS_DIR):
        print(f"❌ Errore path")
        return

    files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith('.safetensors')])
    n_files = len(files)
    inspector = MatrixGenerator(device=DEVICE)
    data_rows = []

    print(f"Analisi v8.4 (Roberta & T0 Compatible)...")
    with tqdm(total=n_files*(n_files-1)//2) as pbar:
        for i in range(n_files):
            w_a = inspector.load_weights(os.path.join(MODELS_DIR, files[i]))
            if w_a is None:
                pbar.update(n_files - 1 - i)
                continue
            name_a = clean_filename(files[i])

            for j in range(i + 1, n_files):
                w_b = inspector.load_weights(os.path.join(MODELS_DIR, files[j]))
                if w_b is None:
                    pbar.update(1)
                    continue
                name_b = clean_filename(files[j])

                metrics = inspector.compute_metrics(w_a, w_b)
                if metrics:
                    basin, rel = inspector.classify_relation(metrics)
                    data_rows.append({
                        "Model_A": name_a, "Model_B": name_b,
                        "Cosine_Sim": round(metrics["Cosine_Sim"], 6),
                        "Sign_Agreement": round(metrics["Sign_Agreement"], 6),
                        "BCS_Score": round(metrics["BCS_Score"], 6),
                        "Delta_Rank": round(metrics["Delta_Rank"], 6),
                        "Relative_Dist": round(metrics["Relative_Dist"], 6),
                        "LER_Score": round(metrics["LER_Score"], 6),
                        "IS_SAME_BASIN": basin, "RELATION_TYPE": rel
                    })
                del w_b
                pbar.update(1)
            del w_a

    df = pd.DataFrame(data_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Fatto: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()