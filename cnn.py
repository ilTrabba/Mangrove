import torch
import numpy as np
import scipy.stats as stats
from safetensors.torch import load_file
import os

# ==============================================================================
# ⚙️ CONFIGURAZIONE: INSERISCI QUI I TUOI PATH
# ==============================================================================
PATH_MODEL_ROOT = "/home/cristian/projects/Model_Graph/model_heritage_backend/weights/models/38768cb0-44ee-4ade-8b63-712b3eeccea6_resnet-50.safetensors" 
PATH_MODEL_B = "/home/cristian/projects/Model_Graph/model_heritage_backend/weights/models/f4415195-b92a-4286-aaab-c17b840aa085_pneumonia.safetensors"
PATH_MODEL_C = "/home/cristian/projects/Model_Graph/model_heritage_backend/weights/models/4658f7d6-e59a-483d-bbac-062feed721ed_pneumonia1.safetensors"
PATH_MODEL_D = "/home/cristian/projects/Model_Graph/model_heritage_backend/weights/models/969fecc1-907b-419e-b9b7-afaee1cac3dc_resnet50_jellyfish_classifier.safetensors"
PATH_MODEL_E = "/home/cristian/projects/Model_Graph/model_heritage_backend/weights/models/5797dc9a-bb4d-4403-b7d2-a031d57e833d_cat_dog_classifier.safetensors"
PATH_MODEL_F = "/home/cristian/projects/Model_Graph/model_heritage_backend/weights/models/ace73406-bb50-4e43-b898-0cb45a45ece3_microsoft-resnet-50-batch32-lr0.safetensors"
PATH_MODEL_G = "/home/cristian/projects/Model_Graph/model_heritage_backend/weights/models/bea5960f-e55b-4289-a563-f34c1dc7b147_msi-resnet-pretrain.safetensors"
PATH_MODEL_H = "/home/cristian/projects/Model_Graph/model_heritage_backend/weights/models/972ef200-e3c6-4344-8f52-c64f35f34c44_msi-resnet-50.safetensors"
PATH_MODEL_I = "/home/cristian/projects/Model_Graph/model_heritage_backend/weights/models/39932d72-9f19-4336-ba31-831de42dc256_resnet50_rvl-cdip.safetensors"
PATH_MODEL_L = "/home/cristian/projects/Model_Graph/model_heritage_backend/weights/models/759250f8-c01c-4598-aa6d-b3e04183b4fc_paper_model_DP_1.safetensors"
PATH_MODEL_M = "/home/cristian/projects/Model_Graph/model_heritage_backend/weights/models/bba3ff1e-9dc9-4bf7-814b-c67013b82f97_fruits-and-vegetables-detector-36.safetensors"
PATH_MODEL_N = "/home/cristian/projects/Model_Graph/model_heritage_backend/weights/models/c4c6ce59-e197-4aba-872f-3f29a22d1604_results.safetensors"
PATH_MODEL_O = "/home/cristian/projects/Model_Graph/model_heritage_backend/weights/models/e3d61721-312d-40dd-8720-66d8b9cd5dc3_resnet-Alzheimer.safetensors"
PATH_MODEL_P = "/home/cristian/projects/Model_Graph/model_heritage_backend/weights/models/e097f7d4-8138-4489-a7bb-3fe0d7280a50_resnet-50_ferplus.safetensors"


# ==============================================================================

def calculate_safetensor_distance(weights_a: dict, weights_b: dict) -> float:
    """
    Calcola la distanza l_FT (MoTHer) considerano SOLAMENTE i kernel convoluzionali veri.
    Ignora tassativamente: Normalizzazioni, Bias, Embedding e Layer Lineari/Classificatori.
    """
    distances = []
    common_keys = set(weights_a.keys()).intersection(set(weights_b.keys()))

    for key in common_keys:
        tensor_a = weights_a[key]
        tensor_b = weights_b[key]

        # 1. Filtro esplicito sul NOME (deve essere una convoluzione, non una normalizzazione)
        if "conv" not in key.lower():
            continue
            
        # 2. Escludiamo i bias (nel caso un layer conv avesse il bias attivo)
        if "bias" in key.lower():
            continue

        # 3. FILTRO DIMENSIONALE (Il test definitivo)
        # Un kernel convoluzionale 2D in PyTorch/Safetensors DEVE avere 4 dimensioni.
        # Es: [64, 64, 3, 3]. Se non è 4D, lo scartiamo.
        if tensor_a.ndim != 4:
            continue

        # 4. Verifica che l'architettura non sia mutata in quel layer
        if tensor_a.shape != tensor_b.shape:
            continue

        # Flattening [Out, In, K, K] -> [N]
        pa = tensor_a.detach().cpu().float().flatten()
        pb = tensor_b.detach().cpu().float().flatten()
        
        # Distanza Euclidea (L2 Norm)
        layer_dist = torch.norm(pa - pb, p=2).item()
        distances.append(layer_dist)

    if not distances:
        return 0.0
    
    # Media delle distanze
    return sum(distances) / len(distances)

def calculate_kurtosis_from_dict(weights: dict, mode: str = 'dense') -> float:
    """
    Calcola la Directional Score (Kurtosis) direttamente dal dizionario dei pesi.
    
    Modes:
      - 'dense': Cerca solo layer fully connected (fc, classifier).
      - 'conv':  Cerca tutti i layer convoluzionali (conv).
    """
    # Pattern per identificare i layer nelle chiavi del dizionario
    DENSE_KEYWORDS = [
        'fc.weight', 
        'classifier.weight', 
        'classifier.1.weight', # <--- AGGIUNTO PER IL TUO MODELLO
        'linear.weight', 
        'head.weight',
        'head.fc.weight'
    ]
    
    CONV_KEYWORDS = [
        'conv.weight', 
        'downsample.0.weight', 
        'shortcut.weight',
        'convolution.weight',   # <--- AGGIUNTO: Il colpevole principale!
        'embedder.convolution.weight'
    ]    
    total_kurtosis = 0.0
    valid_layers = 0
    
    for key, tensor in weights.items():
        key_lower = key.lower()

        # LOGICA DI SELEZIONE LAYER
        is_target = False
        
        if mode == 'dense':
            # Deve contenere una keyword densa E essere 2D
            if any(k in key_lower for k in DENSE_KEYWORDS) and tensor.ndim == 2:
                is_target = True
                
        elif mode == 'conv':
            # Deve contenere una keyword conv E essere 4D (tipico CNN: Out, In, H, W)
            if any(k in key_lower for k in CONV_KEYWORDS) and tensor.ndim == 4:
                is_target = True

        if is_target:
            # Flattening
            w_cpu = tensor.detach().cpu().float().numpy().ravel()
            
            # Calcolo Kurtosis (Fisher)
            k = stats.kurtosis(w_cpu)
            
            if not np.isnan(k) and not np.isinf(k):
                total_kurtosis += k
                valid_layers += 1
                # Debug opzionale
                # print(f"  -> {key}: k={k:.4f}")

    if valid_layers == 0:
        print(f"⚠️  Warning: Nessun layer valido trovato per la modalità '{mode}'.")
        return 0.0

    # Il paper usa la SOMMA, non la media [cite: 150]
    return total_kurtosis

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    if not os.path.exists(PATH_MODEL_ROOT) or not os.path.exists(PATH_MODEL_B):
        print("❌ ERRORE: Uno dei file non esiste. Controlla i PATH nel codice.")
        exit()

    print(f"📂 Caricamento A: {os.path.basename(PATH_MODEL_ROOT)}...")
    weights_a = load_file(PATH_MODEL_ROOT)
    
    print(f"📂 Caricamento B: {os.path.basename(PATH_MODEL_ROOT)}...")
    weights_b = load_file(PATH_MODEL_ROOT)

    print("\n" + "="*40)
    print("   RISULTATI ANALISI MOTHER (CNN)")
    print("="*40)

    # 1. DISTANZA
    dist = calculate_safetensor_distance(weights_a, weights_b)
    print(f"\n📏 Weight Distance (l_FT): {dist:.6f}")
    if dist == 0:
        print("   (I modelli sono identici)")
    elif dist < 10.0:
        print("   (Probabile parentela stretta / Fine-tuning leggero)")
    else:
        print("   (Distanza elevata, potrebbero non essere parenti stretti)")

    # 2. KURTOSIS (DIRECTIONAL SCORE)
    print(f"\n📊 Kurtosis Analysis (Directional Score)")
    
    # Modo A: Dense (Rigoroso secondo paper)
    ka_dense = calculate_kurtosis_from_dict(weights_a, mode='dense')
    kb_dense = calculate_kurtosis_from_dict(weights_b, mode='dense')
    
    print(f"\n  [MODE: DENSE/FC LAYERS] (Rigoroso)")
    print(f"  Modello A: {ka_dense:.4f}")
    print(f"  Modello B: {kb_dense:.4f}")
    
    if ka_dense > kb_dense:
        print(f"  👉 DIREZIONE: A -> B (A è il Padre)")
    elif kb_dense > ka_dense:
        print(f"  👉 DIREZIONE: B -> A (B è il Padre)")
    else:
        print(f"  👉 DIREZIONE: Incerta (Valori uguali)")

    # Modo B: Conv (Adattato CNN)
    ka_conv = calculate_kurtosis_from_dict(weights_a, mode='conv')
    kb_conv = calculate_kurtosis_from_dict(weights_b, mode='conv')
    
    print(f"\n  [MODE: CONV LAYERS] (Aggregato)")
    print(f"  Modello A: {ka_conv:.4f}")
    print(f"  Modello B: {kb_conv:.4f}")

    if ka_conv > kb_conv:
        print(f"  👉 DIREZIONE: A -> B (A è il Padre)")
    elif kb_conv > ka_conv:
        print(f"  👉 DIREZIONE: B -> A (B è il Padre)")
    else:
        print(f"  👉 DIREZIONE: Incerta")