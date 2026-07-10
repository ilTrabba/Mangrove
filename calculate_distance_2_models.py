import sys
import argparse
import numpy as np
import torch
from safetensors.torch import load_file
from typing import Dict, Any, Optional, Set, Tuple

# --- ENUMS E COSTANTI ---
class DistanceMetric:
    L2_DISTANCE = "l2"
    COSINE_SIMILARITY = "cosine"
    RMS_L2_DISTANCE = "rms_l2"
    HYBRID_DISTANCE = "hybrid"


class FilteringPatterns:
    # =============================================================================
    # 1. BASE EXCLUSIONS (Normalization, Embeddings, Heads, Pools)
    # =============================================================================
    BASE_EXCLUSIONS = frozenset([
        # --- Normalization ---
        'layernorm', 'layer_norm', 'ln_', '.ln.', 'ln_1', 'ln_2', 'ln_f',
        'batchnorm', 'batch_norm', '.bn.', 'bn1', 'bn2', 'bn3',
        'groupnorm', 'group_norm', 'gn',
        'instancenorm', 'instance_norm',
        'rmsnorm', 'rms_norm',
        'norm.', '_norm', '.norm',
        'norm1', 'norm2', 'norm3',
        'final_norm', 'model.norm',

        # --- Embeddings / Input Stems ---
        'embed', 'embedding', 'embeddings',
        'token_embed', 'word_embed', 'wte', 'wpe',
        'position_embed', 'positional_embed', 'pos_embed', 'abs_pos_embed',
        'patch_embed', 'patch_embedding',
        'input_embedding', 'input_embed',
        'shared',
        'proj_stem',

        # --- Positional / Rotary / Special Tokens ---
        'rope', 'rotary', 'alibi',
        'pos_bias', 'relative_position', 'rel_pos',
        'relative_attention_bias',
        'cls_token', 'dist_token', 'mask_token',

        # --- Output Heads / Classifiers / Poolers ---
        'lm_head', 'language_model_head',
        'classifier', 'classification_head', 'cls_head',
        'segmentation_head', 'mask_head', 'det_head',
        'prediction_head', 'pred_head',
        'qa_outputs', 'seq_relationship', 'next_sentence',
        'pooler', 'global_pool', 'avgpool', 'maxpool',
        'logits', 'final_logits_bias',
        'visual_projection', 'text_projection',

        # --- Final Layers Generic Names ---
        'classifier.', 'head.', 'final_fc', 'final_linear',
        'output_head', 'output_layer', 'final_layer'
    ])

    # =============================================================================
    # 2. LISTA BACKBONE ONLY
    # =============================================================================
    BACKBONE_ONLY = BASE_EXCLUSIONS

    # =============================================================================
    # 3. LISTA ATTENTION ONLY
    # =============================================================================
    MLP_AND_CONV_EXCLUSIONS = frozenset([
        # --- Classic MLP / Feed Forward ---
        'mlp', 'ffn', 'feedforward', 'feed_forward',
        'intermediate',      
        'output.dense',      # BERT: 'output.dense' (ma salviamo attention.output.dense via codice)
        'dense_h_to_4h', 'dense_4h_to_h',
        'fc1', 'fc2',
        'linear1', 'linear2',

        # --- Modern LLM MLP ---
        'gate_proj', 'up_proj', 'down_proj',

        # --- T5 / PaLM MLP ---
        'wi', 'wi_0', 'wi_1', 'wo',

        # --- Mixture of Experts (MoE) ---
        'experts', 'block_sparse_moe',

        # --- Convolutional Layers ---
        'conv', 'conv1', 'conv2', 'conv3',
        'downsample'
    ])

    ATTENTION_ONLY = BASE_EXCLUSIONS | MLP_AND_CONV_EXCLUSIONS


# --- DISTANCE CALCULATOR ---
class ModelDistanceCalculator:

    def calculate_l2_layer_distance(self, t1: torch.Tensor, t2: torch.Tensor) -> float:
        t1 = t1.detach().cpu().float()
        t2 = t2.detach().cpu().float()
        return float(np.linalg.norm((t1.numpy() - t2.numpy()).ravel()))

    def calculate_cosine_layer_distance(self, t1: torch.Tensor, t2: torch.Tensor) -> Optional[float]:
        t1 = t1.detach().cpu().float().numpy().ravel()
        t2 = t2.detach().cpu().float().numpy().ravel()

        n1 = np.linalg.norm(t1)
        n2 = np.linalg.norm(t2)

        if n1 == 0 or n2 == 0:
            return None

        cosine_sim = np.dot(t1, t2) / (n1 * n2)
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        return float(1.0 - cosine_sim)

    def calculate_rms_l2_layer_distance(self, t1: torch.Tensor, t2: torch.Tensor) -> float:
        t1 = t1.detach().cpu().float()
        t2 = t2.detach().cpu().float()
        diff = (t1.numpy() - t2.numpy()).ravel()
        return float(np.sqrt(np.mean(diff ** 2)))

    def calculate_hybrid_layer_distance(self, t1: torch.Tensor, t2: torch.Tensor) -> Optional[float]:
        alpha = 0.3
        l2 = self.calculate_l2_layer_distance(t1, t2)
        cosine = self.calculate_cosine_layer_distance(t1, t2)
        if cosine is None:
            return None
        return float(alpha * l2 + (1 - alpha) * cosine)

    def calculate_distance(
        self,
        weights1: Dict[str, Any],
        weights2: Dict[str, Any],
        metric_type: str,
        excluded_patterns: Optional[frozenset] = None,
        return_layers: bool = False
    ) -> Tuple[float, Optional[Set[str]]]:

        if excluded_patterns is None:
            excluded_patterns = FilteringPatterns.BACKBONE_ONLY

        keys1 = set(weights1.keys())
        keys2 = set(weights2.keys())
        common_params = keys1 & keys2

        print(f"🔎 Found {len(common_params)} common parameters.")

        total_distance = 0.0
        param_count = 0
        excluded_count = 0
        used_layers: Set[str] = set()

        for name in common_params:
            name_lower = name.lower()

            # --- FILTRO 1: BASE EXCLUSIONS (Priorità Assoluta) ---
            # Scartiamo SEMPRE normalization, embeddings, bias posizionali, etc.
            # Questo evita che il check successivo su "attention" salvi cose come "attention.LayerNorm"
            if any(p in name_lower for p in FilteringPatterns.BASE_EXCLUSIONS):
                excluded_count += 1
                continue

            # --- FILTRO 2: ATTENTION WHITELIST (Priorità Alta) ---
            # Se contiene "attention" o "attn", lo teniamo a prescindere dal resto della blacklist.
            # Questo salva "attention.output.dense" che altrimenti verrebbe ucciso da "output.dense"
            if "attention" in name_lower or "attn" in name_lower:
                pass # È valido, salta il check successivo

            # --- FILTRO 3: STANDARD BLACKLIST (Priorità Bassa) ---
            # Se non è attention, controlliamo se è nella lista di esclusione passata (es. MLP, Conv)
            elif any(p in name_lower for p in excluded_patterns):
                excluded_count += 1
                continue

            # --- CALCOLO DISTANZA ---
            t1 = weights1[name]
            t2 = weights2[name]

            if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor):
                continue
            
            # Controllo dimensioni
            if t1.shape != t2.shape:
                excluded_count += 1
                continue
            
            # Controllo tensori 2D (Spesso vogliamo solo matrici di peso, non bias vettoriali, ma dipende dal caso)
            # Qui mantengo il tuo check originale: deve essere 2D e quadrata?
            # ATTENZIONE: La condizione `t1.shape[0] != t1.shape[1]` forza matrici QUADRATE.
            # Molti layer (FFN, Attention Projections non quadrate) verrebbero scartati.
            # Rimuovo il check sulla matrice quadrata se vuoi calcolare tutto, 
            # ma mantengo il check 2D se vuoi escludere bias (che sono 1D).
            if len(t1.shape) != 2: 
                # Se vuoi includere anche i bias (1D), rimuovi questo if.
                excluded_count += 1
                continue
                
            # Nota: Ho rimosso `or t1.shape[0] != t1.shape[1]` perché escluderebbe layer rettangolari (es. FFN up/down proj)

            if metric_type == DistanceMetric.L2_DISTANCE:
                dist = self.calculate_l2_layer_distance(t1, t2)
            elif metric_type == DistanceMetric.COSINE_SIMILARITY:
                dist = self.calculate_cosine_layer_distance(t1, t2)
            elif metric_type == DistanceMetric.RMS_L2_DISTANCE:
                dist = self.calculate_rms_l2_layer_distance(t1, t2)
            elif metric_type == DistanceMetric.HYBRID_DISTANCE:
                dist = self.calculate_hybrid_layer_distance(t1, t2)
            else:
                raise ValueError("Invalid metric")

            if dist is None:
                continue

            total_distance += dist
            param_count += 1
            used_layers.add(name)

        print(f"📊 Stats: {param_count} layers calculated, {excluded_count} excluded.")

        if param_count == 0:
            result = float("inf")
        else:
            result = total_distance / param_count

        if return_layers:
            return result, used_layers
        return result, None


# --- MAIN ---
def main():
    # Modifica i path con i tuoi file
    model_a = "/home/cristian/projects/Model_Graph/model_heritage_backend/weights/models/f350a2a7-90f9-47b0-9914-d002f6dbb0ff_stablelm-2-1_6b.safetensors"
    model_d = "/home/cristian/projects/Model_Graph/model_heritage_backend/weights/models/0dbee8de-4fd1-40ec-bc0f-e4de5d3d29b6_stablelm-2-1_6b-orpo-full-v3.safetensors"
    model_b = "/home/cristian/projects/Model_Graph/model_heritage_backend/weights/models/fe196f01-9a6b-4269-b23c-ee1563cdd7ee_stablelm-2-1_6b-orpo-full-v1.safetensors"
    model_c = "/home/cristian/projects/Model_Graph/model_heritage_backend/weights/models/ccef1a61-e29d-4bd9-b9d4-17c744fa6999_stablelm-2-1_6b-orpo-full-v2.safetensors"
    
    # Esempio dummy se non hai file reali al momento
    # model_a = "dummy_a.safetensors"
    # model_b = "dummy_b.safetensors"
    
    # Se vuoi testare senza file, commenta la parte di load e usa dizionari finti
    try:
        print("📂 Loading models...")
        weights_a = load_file(model_a, device="cpu")
        weights_b = load_file(model_d, device="cpu")
    except Exception as e:
        print(f"⚠️ Errore caricamento file (normale se path non esistono): {e}")
        return

    calc = ModelDistanceCalculator()

    print("\n🚀 ATTENTION_ONLY")
    dist_bb, layers_bb = calc.calculate_distance(
        weights_a,
        weights_b,
        metric_type=DistanceMetric.L2_DISTANCE,
        excluded_patterns=FilteringPatterns.ATTENTION_ONLY,
        return_layers=True
    )

    print("\n🚀 BACKBONE_ONLY")
    dist_full, layers_full = calc.calculate_distance(
        weights_a,
        weights_b,
        metric_type=DistanceMetric.L2_DISTANCE,
        excluded_patterns=FilteringPatterns.BACKBONE_ONLY,
        return_layers=True
    )

    print("\n" + "-" * 40)
    print(f"✅ Final Distance (ATTENTION_ONLY): {dist_bb:.6f}")
    print(f"✅ Final Distance (BACKBONE_ONLY):   {dist_full:.6f}")
    print("-" * 40)
"""
    if layers_bb is not None and layers_full is not None:
        # Mostra cosa c'è in più nella backbone (che dovrebbe essere l'MLP)
        if len(layers_full) != len(layers_bb):
            extra = sorted(layers_full - layers_bb)
            print("\n🧠 Layers included ONLY in BACKBONE (should be MLP layers):")
            for name in extra:
                print(f"  + {name}")
        else:
            print("\n🧠 No additional layers found (Check filtering logic).")

"""
if __name__ == "__main__":
    main()
