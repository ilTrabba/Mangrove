import os
import time
import torch
import itertools
import numpy as np
import pandas as pd

from tqdm import tqdm
from safetensors.torch import load_file
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

LORA = True
K = 5  # Numero di cluster desiderato
BASE_MODELS_DIR = "/home/cristian/projects/Model_Graph/model_heritage_backend/weights/models"

def extract_model_tensors(file_path: str):
    """
    Legge direttamente il file .safetensors.
    Converte i pesi in float32 per evitare crash con i modelli in bfloat16.
    """
    state_dict = load_file(file_path)
    valid_tensors = {}
    
    for name, layer in state_dict.items():
        # Filtro per layer bidimensionali e quadrati
        if len(layer.shape) == 2 and layer.shape[0] == layer.shape[1]:
            # Cast a float32 -> CPU -> NumPy
            valid_tensors[name] = layer.to(torch.float32).cpu().numpy()
            
    return valid_tensors

def perform_clustering_and_timing(base_dir: str, lora: bool = True, k_clusters: int = 5):
    start_time = time.time()
    
    # 1. Identificazione esclusiva dei file .safetensors
    model_files = [f for f in os.listdir(base_dir) if f.endswith('.safetensors')]
    num_models = len(model_files)
    
    if num_models < k_clusters:
        raise ValueError(f"Trovati solo {num_models} file .safetensors, ma K è impostato a {k_clusters}.")

    print(f"Trovati {num_models} modelli (.safetensors). Inizio l'estrazione dei pesi...")
    
    # 2. Estrazione diretta dei tensori
    state_dicts = {}
    for filename in tqdm(model_files, desc="Estrazione Tensori"):
        file_path = os.path.join(base_dir, filename)
        model_id = filename.replace('.safetensors', '')
        state_dicts[model_id] = extract_model_tensors(file_path)
        
    # 3. Calcolo della matrice delle distanze
    idx_ = sorted(list(state_dicts.keys()))
    dist_ = pd.DataFrame(0.0, index=idx_, columns=idx_)
    
    print("\nCalcolo delle distanze a coppie...")
    for i_, j_ in tqdm(list(itertools.combinations(idx_, 2)), desc="Distanze"):
        dict_i = state_dicts[i_]
        dict_j = state_dicts[j_]
        
        model_dist = 0.0
        shared_layers = 0  # <--- NUOVO: Contatore per i layer in comune
        
        # Calcolo distanza confrontando i layer con lo stesso nome
        for layer_name in dict_i.keys():
            if layer_name not in dict_j:
                continue
                
            layer_i = dict_i[layer_name]
            layer_j = dict_j[layer_name]
            
            if layer_i.shape != layer_j.shape:
                continue
            
            shared_layers += 1  # Incrementiamo se c'è un layer valido in comune
                
            if lora:
                layer_dist = np.linalg.matrix_rank(layer_i - layer_j)
            else:
                layer_dist = np.abs(layer_i.flatten() - layer_j.flatten()).mean()
                
            model_dist += layer_dist
            
        # <--- NUOVO: Applicazione della distanza infinita
        if shared_layers == 0:
            model_dist = 1e9  # Usiamo 1 miliardo per evitare l'errore ValueError di SciPy
            
        dist_.loc[i_, j_] = model_dist
        dist_.loc[j_, i_] = model_dist

    # Salvataggio matrice su file CSV
    dist_filename = f'{"lora_" if lora else "full_"}distance_matrix.csv'
    dist_.to_csv(dist_filename)
    
    # 4. Clustering Gerarchico
    print(f"\nEsecuzione Clustering (K={k_clusters})...")
    tmp_dist = squareform(dist_.values)
    Z = linkage(tmp_dist, method='ward')
    clusters = fcluster(Z, k_clusters, criterion='maxclust')
    
    # 5. Mapping dei risultati
    results = pd.DataFrame({
        'Modello': idx_,
        'Cluster_ID': clusters
    }).sort_values(by='Cluster_ID').reset_index(drop=True)
    
    # 6. Calcolo dei tempi
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_model = total_time / num_models
    
    # Output dei risultati
    print("\n" + "="*40)
    print(" RISULTATI CLUSTERING")
    print("="*40)
    print(results.to_string(index=False))
    
    print("\n" + "="*40)
    print(" STATISTICHE DI ESECUZIONE")
    print("="*40)
    print(f"Tempo Totale Complessivo:  {total_time:.2f} secondi")
    print(f"Tempo Medio per Modello:   {avg_time_per_model:.2f} secondi/modello")
    print("="*40)

if __name__ == '__main__':
    perform_clustering_and_timing(
        base_dir=BASE_MODELS_DIR, 
        lora=LORA, 
        k_clusters=K
    )
