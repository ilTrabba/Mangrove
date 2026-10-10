"""
Delta-only lineage inference (variant: ``mother-with-delta-v01``).

Ported from ``freccia_del_tempo.py`` (Ratio-Aware Geometric Lineage V2.6).
Direction (parent -> child) is inferred ONLY from the weight difference
(delta) and from an SVD of the two weight matrices:

* delta energy projected on the top-k left singular vectors of each model
* spectral concentration (sigma_max / sum(sigma))

No kurtosis (per-model or per-family) is computed or used here.

Public entry point::

    process_lineage_graph(input_graph, penalty, exclude_value)
        -> Tuple[nx.DiGraph, Dict[str, Any]]
"""

import itertools
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import torch

logger = logging.getLogger(__name__)

VALID_EXTS = ('.safetensors', '.bin', '.ckpt', '.pt', '.pth')
SPECTRAL_RANK = 10            # number of left singular vectors used for the delta energy
ENERGY_RATIO_THRESHOLD = 1.01  # above: ADDITIVE mode (trust delta), else SUBTRACTIVE (trust spectrum)
MAX_CACHED_MODELS = 4


# ---------------------------------------------------------------------------
# Key normalisation / layer filtering
# ---------------------------------------------------------------------------

def normalize_key(key: str) -> str:
    """Strip common prefixes so layer names match across checkpoints."""
    key = key.replace('model.', '').replace('vit.', '').replace('transformer.', '')
    match = re.search(r'(layers|blocks|h|encoder\.layer|decoder\.layer|layer)\.(\d+)\.(.*)', key)
    if match:
        return f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
    return key


def is_integrator_layer(key: str, shape, exclude_value: bool = False) -> bool:
    """Only 2D / 4D tensors; skip norm/embed/lora/bias/time, Q and K (and V if requested)."""
    if len(shape) not in (2, 4):
        return False
    key_lower = key.lower()
    exclusions = ["norm", "embed", "lora", "bias", "time",
                  "to_q", "q_proj", ".q.", "query",
                  "to_k", "k_proj", ".k.", "key"]
    if exclude_value:
        exclusions.extend(["to_v", "v_proj", ".v.", "value"])
    return not any(bad in key_lower for bad in exclusions)


# ---------------------------------------------------------------------------
# Weight loading (single file or sharded directory)
# ---------------------------------------------------------------------------

def _load_chunk(path: str) -> Optional[Dict[str, Any]]:
    try:
        if path.endswith('.safetensors'):
            from safetensors.torch import load_file
            return load_file(path, device='cpu')
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=True)
        except Exception:
            logger.warning("weights_only load failed for %s, retrying with weights_only=False "
                           "(only do this for trusted checkpoints)", path)
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            return ckpt['state_dict']
        return ckpt
    except Exception as e:
        logger.error("Failed to load %s: %s", path, e)
        return None


def load_model_weights_any(model_path: str) -> Optional[Dict[str, Any]]:
    """Load a single weights file or merge all shards of a directory."""
    if not model_path:
        return None
    if os.path.isfile(model_path):
        return _load_chunk(model_path)
    if os.path.isdir(model_path):
        files = sorted(f for f in os.listdir(model_path) if f.endswith(VALID_EXTS))
        if not files:
            return None
        merged: Dict[str, Any] = {}
        for f in files:
            chunk = _load_chunk(os.path.join(model_path, f))
            if chunk is not None:
                merged.update(chunk)
        return merged or None
    return None


# ---------------------------------------------------------------------------
# Pairwise engine
# ---------------------------------------------------------------------------

class DeltaLineageEngine:
    """Accumulates per-layer votes on who is the father between model A and B."""

    def __init__(self):
        self.votes_a = 0.0
        self.votes_b = 0.0
        self.mode_additive = 0
        self.mode_subtractive = 0
        self.n_layers = 0

    @torch.no_grad()
    def analyze_layer(self, w_a: torch.Tensor, w_b: torch.Tensor) -> None:
        w_a, w_b = w_a.float(), w_b.float()
        if w_a.ndim == 4:  # conv kernels -> 2D
            w_a = w_a.reshape(w_a.shape[0], -1)
            w_b = w_b.reshape(w_b.shape[0], -1)

        delta = w_b - w_a
        if torch.norm(delta) < 1e-6:
            return
        try:
            U_a, S_a, _ = torch.linalg.svd(w_a, full_matrices=False)
            U_b, S_b, _ = torch.linalg.svd(w_b, full_matrices=False)
            U_a, U_b = U_a[:, :SPECTRAL_RANK], U_b[:, :SPECTRAL_RANK]
        except Exception:
            return

        en_a = torch.norm(U_a.T @ delta) ** 2
        en_b = torch.norm(U_b.T @ delta) ** 2
        conc_a = S_a[0] / (torch.sum(S_a) + 1e-9)
        conc_b = S_b[0] / (torch.sum(S_b) + 1e-9)

        ratio = max(en_a, en_b) / (min(en_a, en_b) + 1e-9)
        if ratio > ENERGY_RATIO_THRESHOLD:       # ADDITIVE: trust delta
            if en_b > en_a:
                self.votes_a += 1
            else:
                self.votes_b += 1
            self.mode_additive += 1
        else:                                     # SUBTRACTIVE: trust spectrum
            if conc_b > conc_a:
                self.votes_a += 1
            else:
                self.votes_b += 1
            self.mode_subtractive += 1
        self.n_layers += 1

    def verdict(self) -> Dict[str, Any]:
        total = self.votes_a + self.votes_b + 1e-9
        conf = abs(self.votes_a / total - 0.5) * 2 * 100
        return {
            'a_is_father': self.votes_a > self.votes_b,
            'confidence': min(conf, 99.9),   # percent
            'mode_additive': self.mode_additive,
            'mode_subtractive': self.mode_subtractive,
        }


def compare_pair(map_a: Dict[str, Any], map_b: Dict[str, Any], exclude_value: bool) -> Dict[str, Any]:
    """Run the delta-only test on two normalised weight maps."""
    engine = DeltaLineageEngine()
    valid_layers = 0
    t0 = time.perf_counter()
    for k in set(map_a) & set(map_b):
        wa, wb = map_a[k], map_b[k]
        if not (isinstance(wa, torch.Tensor) and isinstance(wb, torch.Tensor)):
            continue
        if wa.shape == wb.shape and is_integrator_layer(k, wa.shape, exclude_value):
            engine.analyze_layer(wa, wb)
            valid_layers += 1
    process_time = time.perf_counter() - t0
    if engine.n_layers == 0:
        return {'error': '0 layers analysed', 'process_time': process_time, 'valid_layers': 0}
    res = engine.verdict()
    res.update(process_time=process_time, valid_layers=engine.n_layers, error=None)
    return res


# ---------------------------------------------------------------------------
# Graph-level API
# ---------------------------------------------------------------------------

def _node_path(g: nx.DiGraph, node: Any) -> str:
    data = g.nodes[node]
    return data.get('path') or data.get('file_path') or str(node)


def _candidate_pairs(g: nx.DiGraph) -> List[Tuple[Any, Any]]:
    """Unordered pairs to test: input edges if any, otherwise all node pairs."""
    if g.number_of_edges() > 0:
        seen, pairs = set(), []
        for u, v in g.edges():
            key = frozenset((u, v))
            if u != v and key not in seen:
                seen.add(key)
                pairs.append((u, v))
        return pairs
    return list(itertools.combinations(sorted(g.nodes(), key=str), 2))


def process_lineage_graph(input_graph: nx.DiGraph,
                          penalty: float,
                          exclude_value: bool) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """
    Delta-only lineage inference.

    Input nodes must carry a ``path`` (or ``file_path``) attribute pointing to a
    weights file or a sharded directory (otherwise the node id is used as path).
    Input edges, if present, select the pairs to test; otherwise all pairs are tested.

    For every tested pair BOTH edges are inserted:
      * predicted parent -> child : weight = base_confidence,           is_correct_direction = True
      * child -> parent           : weight = base_confidence + penalty, is_correct_direction = False

    ``base_confidence`` is a cost in [0.001, 1]: ``1 - confidence/100`` (lower = more
    confident), so that minimum spanning arborescence solvers prefer confident edges.
    The raw confidence (percent) is stored in the edge attribute ``confidence``.
    """
    t_total = time.perf_counter()
    out = nx.DiGraph()
    for n, data in input_graph.nodes(data=True):
        out.add_node(n, **data)

    cache: Dict[Any, Optional[Dict[str, Any]]] = {}

    def get_map(node):
        if node not in cache:
            if len(cache) >= MAX_CACHED_MODELS:
                cache.pop(next(iter(cache)))
            sd = load_model_weights_any(_node_path(input_graph, node))
            cache[node] = None if sd is None else {normalize_key(k): v for k, v in sd.items()}
        return cache[node]

    pairs = _candidate_pairs(input_graph)
    pair_stats: List[Dict[str, Any]] = []
    skipped: List[Tuple[Any, Any, str]] = []
    layers_per_node: Dict[Any, List[int]] = {n: [] for n in input_graph.nodes()}
    add_layers = sub_layers = 0
    fwd_w, rev_w = [], []

    for a, b in pairs:
        ma, mb = get_map(a), get_map(b)
        if ma is None or mb is None:
            logger.warning("Skipping pair (%s, %s): weights could not be loaded", a, b)
            skipped.append((a, b, 'load_failed'))
            continue
        res = compare_pair(ma, mb, exclude_value)
        logger.info("[delta] pair (%s, %s) process_time=%.4fs valid_layers=%d",
                    a, b, res['process_time'], res['valid_layers'])
        if res.get('error'):
            logger.warning("Skipping pair (%s, %s): %s", a, b, res['error'])
            skipped.append((a, b, res['error']))
            continue

        parent, child = (a, b) if res['a_is_father'] else (b, a)
        base_confidence = max(0.001, 1.0 - res['confidence'] / 100.0)
        common = dict(confidence=res['confidence'], process_time=res['process_time'],
                      valid_layers=res['valid_layers'], mode_additive=res['mode_additive'],
                      mode_subtractive=res['mode_subtractive'])
        out.add_edge(parent, child, weight=base_confidence, is_correct_direction=True,
                     penalty=0.0, **common)
        out.add_edge(child, parent, weight=base_confidence + penalty, is_correct_direction=False,
                     penalty=penalty, **common)

        fwd_w.append(base_confidence)
        rev_w.append(base_confidence + penalty)
        add_layers += res['mode_additive']
        sub_layers += res['mode_subtractive']
        layers_per_node[a].append(res['valid_layers'])
        layers_per_node[b].append(res['valid_layers'])
        pair_stats.append(dict(parent=parent, child=child, **common, base_confidence=base_confidence))

    n = len(pair_stats)
    node_means = [sum(v) / len(v) for v in layers_per_node.values() if v]
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    stats: Dict[str, Any] = {
        'algorithm': 'mother-with-delta-v01',
        'penalty': penalty,
        'exclude_value': exclude_value,
        'n_pairs_tested': n,
        'n_pairs_skipped': len(skipped),
        'skipped_pairs': skipped,
        'pairs': pair_stats,
        'mean_confidence': mean([p['confidence'] for p in pair_stats]),          # percent
        'mean_process_time': mean([p['process_time'] for p in pair_stats]),      # seconds
        'total_process_time': sum(p['process_time'] for p in pair_stats),
        'total_time': time.perf_counter() - t_total,                              # incl. I/O
        'mean_valid_layers_per_pair': mean([p['valid_layers'] for p in pair_stats]),
        'mean_layers_per_model': mean(node_means),
        'additive_layers': add_layers,
        'subtractive_layers': sub_layers,
        'additive_subtractive_ratio': (add_layers / sub_layers) if sub_layers else float('inf') if add_layers else 0.0,
        'directed_edges': n,
        'penalized_reverse_edges': n,
        'mean_forward_weight': mean(fwd_w),
        'mean_reverse_weight': mean(rev_w),
    }
    return out, stats
