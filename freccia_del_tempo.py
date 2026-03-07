import torch
import numpy as np
from scipy.stats import kurtosis
from safetensors.torch import load_file
import gc
import os

# --- CONFIGURAZIONE ---
# Inserisci i percorsi ESATTI ai file .safetensors

MODEL_ROOT_TREE3 = "/home/trabbo/Documents/Universita/BigData/Models_for_Project/MoTHer/Tree_3/VTHR-FT-ModelTree_3-Depth_0-VP4Kawke.safetensors"
MODEL_FIGLIO_SBAG_TREE3 = "/home/trabbo/Documents/Universita/BigData/Models_for_Project/MoTHer/Tree_3/Child_2/T3-D1-Node_ayxKspj2.safetensors"


MODEL_ROOT_PATH = "/home/trabbo/Documents/Universita/BigData/Models_for_Project/MoTHer/Tree_2/VTHR-FT-ModelTree_2-Depth_0-Ab5S2TPh.safetensors"
MODEL_PADRE_1 = "/home/trabbo/Documents/Universita/BigData/Models_for_Project/MoTHer/Tree_2/Child_1/T2-D1-Node_LTkBYqJ9.safetensors" 
# ----------------------

MODEL_PADRE_2 = "/home/trabbo/Documents/Universita/BigData/Models_for_Project/MoTHer/Tree_2/Child_2/T2-D1-Node_mCGpCzGm.safetensors" 

MODEL_PADRE_3 = "/home/trabbo/Documents/Universita/BigData/Models_for_Project/MoTHer/Tree_2/Child_3/T2-D1-Node_aBR8mQgk.safetensors"

MODEL_PADRE_4 = "/home/trabbo/Documents/Universita/BigData/Models_for_Project/MoTHer/Tree_2/Child_4/T2-D1-Node_26ZrHTei.safetensors"

class GranularAnalyzer:
    def __init__(self):
        pass

    def get_precision_metrics(self, tensor):
        """Calcola metriche con precisione float64."""
        if len(tensor.shape) != 2: return None, None

        w = tensor.float().numpy().astype(np.float64)
        
        # Kurtosi (Fisher)
        k_val = kurtosis(w.flatten(), fisher=True)
        
        # Stable Rank
        try:
            s = np.linalg.svd(w, compute_uv=False)
            frobenius_sq = np.sum(s**2)
            spectral_sq = s[0]**2
            rank_val = frobenius_sq / (spectral_sq + 1e-16)
        except:
            return None, None

        return k_val, rank_val

    def identify_layer_type(self, key):
        """Identifica FFN vs Attention Output (Compatibile Llama, GPT, BERT, ViT)"""
        key = key.lower()
        if "bias" in key or "norm" in key or "embeddings" in key: return None

        # 1. ATTENTION OUTPUT (Quadrate)
        if "attn" in key or "attention" in key:
            if "o_proj" in key or "c_proj" in key or "out_proj" in key or "to_out.0" in key or ("output" in key and "dense" in key):
                return "attn"

        # 2. FFN OUTPUT (Rettangolari - Conoscenza)
        is_ffn_block = ("mlp" in key) or ("feed_forward" in key) or ("ff" in key) or ("intermediate" not in key and "output" in key)
        if is_ffn_block:
            if "down_proj" in key: return "ffn"
            if "c_proj" in key and "attn" not in key: return "ffn"
            if "output" in key and "dense" in key and "attention" not in key: return "ffn"
            if ".wo." in key: return "ffn"
            if "fc2" in key: return "ffn"
            if "ff.net.2" in key: return "ffn"

        return None

    def scan_file(self, file_path):
        print(f"🔬 Scansione file: {os.path.basename(file_path)} ...")
        try: state_dict = load_file(file_path)
        except Exception as e:
            print(f"❌ Errore caricamento: {e}"); return None

        data = {'ffn': {'k': [], 'r': []}, 'attn': {'k': [], 'r': []}}
        
        for key, tensor in state_dict.items():
            l_type = self.identify_layer_type(key)
            if l_type:
                k, r = self.get_precision_metrics(tensor)
                if k is not None:
                    data[l_type]['k'].append(k)
                    data[l_type]['r'].append(r)
        
        del state_dict; gc.collect()
        return data

    def print_single_report(self, k_vals_A, r_vals_A, k_vals_B, r_vals_B, label_A, label_B, context_name, metric_mode):
        """
        Stampa un singolo report isolato per una specifica combinazione (es. FFN - Somma).
        """
        print(f"\n{'='*60}")
        print(f"REPORT: {context_name}  |  Metrica: {metric_mode.upper()}")
        print(f"{'='*60}")

        if not k_vals_A or not k_vals_B:
            print("⚠️ Dati mancanti per questa sezione.")
            return

        # Calcolo in base alla modalità (Media o Somma)
        func = np.mean if metric_mode == "MEDIA" else np.sum
        
        k_A, r_A = func(k_vals_A), func(r_vals_A)
        k_B, r_B = func(k_vals_B), func(r_vals_B)
        
        delta_k = k_B - k_A
        delta_r = r_B - r_A

        # Stampa Tabella
        print(f"{'Modello':<15} | {'Kurtosi':<12} | {'Stable Rank':<12}")
        print("-" * 45)
        print(f"{label_A:<15} | {k_A:12.4f} | {r_A:12.4f}")
        print(f"{label_B:<15} | {k_B:12.4f} | {r_B:12.4f}")
        print("-" * 45)
        print(f"{'DELTA (B-A)':<15} | {delta_k:+12.4f} | {delta_r:+12.4f}")

        # --- DIAGNOSI SPECIFICA PER QUESTO REPORT ---
        print("\n🔍 VERDETTO:")
        
        # 1. Check Inversione (Freccia del tempo basata sul Rank)
        # Se usiamo la somma, la soglia di tolleranza deve essere più alta
        tol = -0.01 if metric_mode == "MEDIA" else -0.1
        
        if delta_r > abs(tol):
            print(f"🚨 ALERT: Inversione o Aumento Complessità ({delta_r:+.4f}).")
            print(f"   In questo contesto ({context_name}), B ha più 'capacità geometrica' di A.")
            print("   -> Possibile inversione Padre/Figlio.")
        else:
            print(f"✅ RELAZIONE OK: Rank diminuito ({delta_r:.4f}). A è il Padre.")
            
            # 2. Check Scenario (Kurtosi)
            if delta_k > 0:
                print(f"📈 SCENARIO 2 (Aggressive): La Kurtosi è SALITA.")
                print("   Il modello ha concentrato l'informazione su meno neuroni (picchi più alti).")
            else:
                print(f"📉 SCENARIO 1 (Standard): La Kurtosi è SCESA.")
                print("   Il modello ha spalmato l'informazione (distribuzione più piatta).")

    def run_full_analysis(self, stats_A, stats_B, label_A="Modello A", label_B="Modello B"):
        # Preparazione dati combinati
        comb_k_A = stats_A['ffn']['k'] + stats_A['attn']['k']
        comb_r_A = stats_A['ffn']['r'] + stats_A['attn']['r']
        comb_k_B = stats_B['ffn']['k'] + stats_B['attn']['k']
        comb_r_B = stats_B['ffn']['r'] + stats_B['attn']['r']

        # --- LE 6 STAMPE RICHIESTE ---

        # 1. FFN - MEDIA
        #self.print_single_report(stats_A['ffn']['k'], stats_A['ffn']['r'], 
                                 #stats_B['ffn']['k'], stats_B['ffn']['r'], 
                                 #label_A, label_B, "FFN (Conoscenza)", "MEDIA")

        # 2. FFN - SOMMA
        self.print_single_report(stats_A['ffn']['k'], stats_A['ffn']['r'], 
                                 stats_B['ffn']['k'], stats_B['ffn']['r'], 
                                 label_A, label_B, "FFN (Conoscenza)", "SOMMA")

        # 3. ATTENTION - MEDIA
        #self.print_single_report(stats_A['attn']['k'], stats_A['attn']['r'], 
                                 #stats_B['attn']['k'], stats_B['attn']['r'], 
                                 #label_A, label_B, "ATTENTION (Output)", "MEDIA")

        # 4. ATTENTION - SOMMA
        self.print_single_report(stats_A['attn']['k'], stats_A['attn']['r'], 
                                 stats_B['attn']['k'], stats_B['attn']['r'], 
                                 label_A, label_B, "ATTENTION (Output)", "SOMMA")

        # 5. COMBINED - MEDIA
        #self.print_single_report(comb_k_A, comb_r_A, 
                                 #comb_k_B, comb_r_B, 
                                 #label_A, label_B, "GLOBALE (Combined)", "MEDIA")

        # 6. COMBINED - SOMMA
        self.print_single_report(comb_k_A, comb_r_A, 
                                 comb_k_B, comb_r_B, 
                                 label_A, label_B, "GLOBALE (Combined)", "SOMMA")

# --- MAIN ---
if __name__ == "__main__":
    analyzer = GranularAnalyzer()

    FILE_B = "/home/trabbo/Documents/GitHub/Model_Graph/model_heritage_backend/weights/models/T0-D1-6hLsBteR.safetensors"
    FILE_A = "/home/trabbo/Documents/GitHub/Model_Graph/model_heritage_backend/weights/models/T0-D0-2Fch5Myt.safetensors"
    print(f"File A: {os.path.basename(FILE_A)}")
    print(f"File B: {os.path.basename(FILE_B)}")

    # Caricamento
    stats_A = analyzer.scan_file(FILE_A)
    stats_B = analyzer.scan_file(FILE_B) 
    

    if stats_A and stats_B:
        analyzer.run_full_analysis(stats_A, stats_B, label_A="Padre (A)", label_B="Figlio (B)")