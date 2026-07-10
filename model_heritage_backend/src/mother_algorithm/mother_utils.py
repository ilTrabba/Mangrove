"""
MoTHer Algorithm Utilities
Based on the paper "Unsupervised Model Tree Heritage Recovery" (ICLR 2025)
"""

from datetime import datetime, timezone
import logging
import numpy as np
import torch
import safetensors
import networkx as nx

import numpy as np
from scipy.stats import kurtosis
from numpy.typing import NDArray
from scipy import stats
from typing import Dict, List, Optional, Any, Tuple
from src.log_handler import logHandler
from src.services.neo4j_service import neo4j_service
from src.utils.architecture_filtering import FilteringPatterns

logger = logging.getLogger(__name__)

# da spulciare bene (forse da eliminare)
def normalize_parent_child_orientation(tree: nx.DiGraph) -> nx.DiGraph:
    """
    Ensure edges are oriented parent -> child.
    If the tree has no nodes with in_degree == 0 but has sinks (out_degree == 0),
    it likely means edges are child -> parent; in that case, reverse it.
    """
    if tree is None or tree.number_of_nodes() == 0:
        return tree
    roots = [n for n in tree.nodes if tree.in_degree(n) == 0]
    sinks = [n for n in tree.nodes if tree.out_degree(n) == 0]
    if len(roots) == 0 and len(sinks) >= 1:
        return nx.reverse(tree, copy=True)
    return tree

def load_model_weights(file_path: str) -> Optional[Dict[str, Any]]:
    """Load model weights from file"""
    try:
        if file_path.endswith('.safetensors'):
            with safetensors.safe_open(file_path, framework="pt", device="cpu") as f:
                weights = {key: f.get_tensor(key) for key in f.keys()}
        elif file_path.endswith(('.pt', '.pth', '.bin')):
            weights = torch.load(file_path, map_location='cpu')

            # Handle state_dict format
            if isinstance(weights, dict) and 'state_dict' in weights:
                weights = weights['state_dict']
        else:
            logger.warning(f"Unsupported file format: {file_path}")
            return None
            
        return weights
    except Exception as e:
        logHandler.error_handler(e, "load_model_weights", {"file_path": file_path})

def calc_ku(weights: Dict[str, Any]) -> float:
    """
    Calculate kurtosis of model weights using a fallback strategy:
    1. First, try to find 2D SQUARE matrices (legacy behavior, best for Llama/BERT).
    2. If NO square matrices are found, fallback to RECTANGULAR matrices (essential for Gemma).
    """
    
    # Lista estesa dei layer supportati
    LAYER_KINDS = [
        'output.dense',  # BERT style
        'o_proj',        # Llama Attention Output
        'out_proj',      # GPT-Neo/J style
        'c_proj',        # GPT-2 style
        'wo',            # T5 style
        'dense.weight',  # pythia style 
        'self_attn.out',
        'down_proj',     # Gemma/Llama MLP Output (Spesso rettangolare)
        'gate_proj',     # Gemma/Llama MLP Gate (Spesso rettangolare)
        'up_proj'        # Gemma/Llama MLP Up (Spesso rettangolare)
    ]

    def _compute_kurtosis_internal(allow_rectangular: bool) -> tuple[float, int, str]:
        """Funzione interna per evitare duplicazione di codice."""
        model_ku = 0.0
        valid_count = 0
        stats_log = {"excluded": 0, "kind_filtered": 0, "shape_filtered": 0, "nan": 0}

        for param_name, param_tensor in weights.items():
            param_lower = param_name.lower()

            # 1. Filtro Esclusioni (Backbone only, etc.)
            # Assumo che FilteringPatterns sia disponibile nel contesto globale
            if any(pattern in param_lower for pattern in FilteringPatterns.BACKBONE_ONLY):
                stats_log["excluded"] += 1
                continue

            # 2. Filtro Nome Layer
            if not any(kind in param_lower for kind in LAYER_KINDS):
                stats_log["kind_filtered"] += 1
                continue

            # 3. Verifica Tipo
            if not isinstance(param_tensor, torch.Tensor):
                continue

            # 4. Filtro Forma (Shape)
            is_2d = param_tensor.ndim == 2
            if not is_2d:
                stats_log["shape_filtered"] += 1
                continue
            
            # LOGICA CORE: Quadrato vs Rettangolare
            is_square = (param_tensor.shape[0] == param_tensor.shape[1])
            
            if not allow_rectangular and not is_square:
                # Se siamo in modalità Strict, scartiamo i rettangolari
                stats_log["shape_filtered"] += 1
                continue
            
            # Se siamo qui, il layer è valido per la modalità corrente

            # Conversione bfloat16 -> float32
            tensor_cpu = param_tensor.detach().cpu()
            if tensor_cpu.dtype == torch.bfloat16:
                tensor_cpu = tensor_cpu.float()

            param_weights = tensor_cpu.numpy().ravel()
            
            # Calcolo Kurtosis
            ku = stats.kurtosis(param_weights) # Fisher=True di default in scipy

            if np.isnan(ku) or np.isinf(ku):
                stats_log["nan"] += 1
                continue

            model_ku += float(ku)
            valid_count += 1
        
        return model_ku, valid_count, str(stats_log)

    try:
        # --- PASSAGGIO 1: Cerca solo matrici QUADRATE (Comportamento Classico) ---
        ku_val, count, log_details = _compute_kurtosis_internal(allow_rectangular=False)

        if count > 0:
            logger.debug(f"Kurtosis (Square Mode): {ku_val:.4f} from {count} layers. Stats: {log_details}")
            return ku_val
        
        # --- PASSAGGIO 2: FALLBACK (Se 0 quadrati, cerca Rettangolari) ---
        logger.info("⚠️ Nessun layer quadrato trovato. Attivazione fallback su layer rettangolari (Gemma Mode)...")
        
        ku_val, count, log_details = _compute_kurtosis_internal(allow_rectangular=True)
        
        if count > 0:
            logger.info(f"✅ Kurtosis (Rectangular Fallback): {ku_val:.4f} from {count} layers. Stats: {log_details}")
            return ku_val

        # Se ancora 0, non c'è niente da fare
        logger.warning(f"Kurtosis calculation failed even after fallback. Stats: {log_details}")
        return 0.0

    except Exception as e:
        logger.error(f"Error calculating kurtosis: {e}", exc_info=True)
        return 0.0

def compute_lambda(distance_matrix: np.ndarray, c: float = 0.3) -> float:
    """
    Compute lambda as defined in the MoTHer paper:
    
        λ = c * (1/n^2) * Σ_{i,j} D_ij
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Matrix of pairwise distances between models (n x n).
    c : float, optional
        Scaling constant, default = 0.3 as in the paper.
    
    Returns
    -------
    float
        Value of λ.
    """
    n = distance_matrix.shape[0]
    mean_distance = np.sum(distance_matrix) / (n * n)
    lam = c * mean_distance
    return lam

def update_family_statistics(family_id: str, distance_matrix: NDArray[np.float64], edge_list: List[Tuple[int, int]]) -> None:
    """
    Update family statistics based on the current distance matrix and selected edges.
    """
    try:
        
        total_distance = 0.0
        num_nodes = distance_matrix.shape[0]

        for i, j in edge_list:
            total_distance += distance_matrix[i, j]

        avg_distance = total_distance / (num_nodes - 1) if num_nodes > 1 else 0.0

        # Update family in Neo4j
        updates = {
            'member_count': num_nodes,
            'avg_intra_distance': avg_distance,
            'updated_at': datetime.now(timezone.utc)
        }
        neo4j_service.update_family(family_id, updates)

        logger.info(f"Updated statistics for family {family_id}: {num_nodes} members, avg_distance: {avg_distance:.4f}")

    except Exception as e:
        logHandler.error_handler(f"Error updating family statistics: {e}", "update_family_statistics")

############################# FUNZIONI DI SUPPORTO CHE ANDRANNO FIXATE (NON IMPORTANTI) ##################################

def fallback_directed_mst(G: nx.DiGraph) -> nx.DiGraph:
    """
    Fallback algorithm using greedy approach for directed MST
    RIPRISTINO IMPLEMENTAZIONE ORIGINALE
    """
    logger.debug("Using fallback directed MST algorithm")
    
    # Sort edges by weight
    edges = [(u, v, data['weight']) for u, v, data in G.edges(data=True)]
    edges.sort(key=lambda x: x[2])
    
    result = nx.DiGraph()
    result.add_nodes_from(G.nodes())
    
    # Greedily add edges, avoiding cycles
    for u, v, weight in edges:
        # Temporarily add the edge
        result.add_edge(u, v, weight=weight)
        
        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(result))
            if cycles:
                # Remove the edge if it creates a cycle
                result.remove_edge(u, v)
        except:
            # Keep the edge if cycle detection fails
            pass
        
        # Stop when we have enough edges for a spanning tree
        if result.number_of_edges() >= len(G.nodes()) - 1:
            break
    
    return result

def calculate_confidence_scores(tree: nx.DiGraph, original_graph: nx.DiGraph, 
                              ku_values: List[float]) -> Dict[int, float]:
    """
    Calculate confidence scores for each node based on tree structure and kurtosis
    RIPRISTINO IMPLEMENTAZIONE ORIGINALE
    """
    confidence_scores = {}
    
    if tree.number_of_edges() == 0:
        # No edges, return default confidence for all nodes
        for node in tree.nodes():
            confidence_scores[node] = 0.5
        return confidence_scores
    
    # Get weight statistics for normalization
    all_weights = [data['weight'] for _, _, data in original_graph.edges(data=True)]
    if not all_weights:
        for node in tree.nodes():
            confidence_scores[node] = 0.5
        return confidence_scores
    
    min_weight = min(all_weights)
    max_weight = max(all_weights)
    weight_range = max_weight - min_weight
    
    for node in tree.nodes():
        predecessors = list(tree.predecessors(node))
        
        if not predecessors:
            # Root node - high confidence
            confidence_scores[node] = 0.85
        else:
            # Child node - confidence based on parent relationship quality
            parent = predecessors[0]  # Should only have one parent in a tree
            
            if tree.has_edge(parent, node):
                edge_weight = tree[parent][node]['weight']
                
                # Normalize weight to confidence (lower weight = higher confidence)
                if weight_range > 0:
                    normalized_weight = (edge_weight - min_weight) / weight_range
                    weight_confidence = 1.0 - normalized_weight
                else:
                    weight_confidence = 0.5
                
                # CORREZIONE: Parent should have HIGHER kurtosis than child
                kurtosis_diff = ku_values[parent] - ku_values[node]
                if kurtosis_diff > 0:
                    kurtosis_confidence = min(1.0, kurtosis_diff * 2)  # Good parent-child relationship
                else:
                    kurtosis_confidence = 0.3  # Questionable relationship
                
                # Combined confidence
                confidence = (weight_confidence + kurtosis_confidence) / 2
                confidence_scores[node] = max(0.1, min(0.95, confidence))
            else:
                confidence_scores[node] = 0.4  # Default for orphaned nodes
    
    return confidence_scores


def find_max_root_distance(spanning_tree: nx.DiGraph, distance_matrix: np.ndarray) -> float:
    """
    Trova la radice dello spanning tree ed estrae la distanza massima
    tra la radice e tutti gli altri nodi usando la matrice delle distanze.
    """
    if spanning_tree.number_of_nodes() == 0:
        return 0.0

    # 1. Trova la radice in modo ottimizzato (si ferma al primo match)
    # Se il generatore si svuota senza trovare nulla, restituisce None
    root = next((n for n, d in spanning_tree.in_degree() if d == 0), None)

    # Caso limite: grafo senza radice valida (es. un ciclo puro, impossibile in un'arborescenza ma sicuro controllare)
    if root is None:
        return 0.0

    # 2. Estrae la distanza massima
    # distance_matrix[root, :] prende l'intera riga della radice (distanze verso TUTTI i nodi)
    max_distance = np.max(distance_matrix[root, :])
    
    return float(max_distance)