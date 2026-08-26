"""
delta_graph_engine.py — mother-with-delta-v01

Delta-only genealogy inference variant for intra-family model graph processing.
Implements process_lineage_graph() using pure delta-based geometric inference
(no kurtosis per-family computation).

Interface (preserved):
    process_lineage_graph(
        input_graph: nx.DiGraph,
        penalty: float,
        exclude_value: bool
    ) -> Tuple[nx.DiGraph, Dict[str, Any]]
"""

import logging
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import networkx as nx
import torch

try:
    from safetensors.torch import load_file as safetensors_load_file
except ImportError:  # pragma: no cover
    safetensors_load_file = None  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Key normalisation helpers
# ---------------------------------------------------------------------------

def normalize_key(key: str) -> str:
    """Strip common prefixes and normalise layer numbering for cross-model matching."""
    key = key.replace("model.", "").replace("vit.", "").replace("transformer.", "")
    match = re.search(
        r"(layers|blocks|h|encoder\.layer|decoder\.layer|layer)\.(\d+)\.(.*)", key
    )
    if match:
        return f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
    return key


def is_integrator_layer(key: str, shape: Tuple[int, ...], exclude_value: bool = False) -> bool:
    """
    Geometric filter: accept only 2-D matrices and 4-D conv tensors that carry
    meaningful directional signal.  Q and K projections are always excluded.
    """
    if len(shape) not in (2, 4):
        return False

    key_lower = key.lower()
    exclusions = ["norm", "embed", "lora", "bias", "time"]
    exclusions += ["to_q", "q_proj", ".q.", "query"]
    exclusions += ["to_k", "k_proj", ".k.", "key"]
    if exclude_value:
        exclusions += ["to_v", "v_proj", ".v.", "value"]

    return not any(bad in key_lower for bad in exclusions)


# ---------------------------------------------------------------------------
# Weight loaders (robust, multi-format)
# ---------------------------------------------------------------------------

def _load_chunk(path: str) -> Optional[Dict[str, torch.Tensor]]:
    """Load a single weight file (.safetensors, .ckpt, .bin, .pt, .pth)."""
    try:
        if path.endswith(".safetensors"):
            if safetensors_load_file is None:
                raise ImportError("safetensors not installed")
            return safetensors_load_file(path, device="cpu")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            return ckpt["state_dict"]
        return ckpt
    except Exception as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return None


def load_model_weights(model_path: str) -> Optional[Dict[str, torch.Tensor]]:
    """Load weights from a single file or merge all shards in a directory."""
    valid_exts = (".safetensors", ".bin", ".ckpt", ".pt", ".pth")

    if os.path.isfile(model_path):
        return _load_chunk(model_path)

    if os.path.isdir(model_path):
        files = [f for f in os.listdir(model_path) if f.endswith(valid_exts)]
        if not files:
            return None
        merged: Dict[str, torch.Tensor] = {}
        for fname in files:
            chunk = _load_chunk(os.path.join(model_path, fname))
            if chunk is not None:
                merged.update(chunk)
        return merged or None

    return None


# ---------------------------------------------------------------------------
# LineageEngineV2_6 — delta-only direction inference
# ---------------------------------------------------------------------------

class LineageEngineV2_6:
    """
    Per-layer voting engine that uses delta-energy and spectral concentration
    to infer genealogical direction between two weight tensors.

    No kurtosis is computed; inference is purely delta/SVD-based.
    """

    def __init__(self, name_a: str, name_b: str) -> None:
        self.name_a = name_a
        self.name_b = name_b
        self.votes_a = 0.0
        self.votes_b = 0.0
        self.mode_additive = 0
        self.mode_subtractive = 0
        self.n_layers = 0

    def analyze_layer(self, w_a: torch.Tensor, w_b: torch.Tensor) -> None:
        """Cast one directional vote for a single weight matrix."""
        w_a = w_a.float()
        w_b = w_b.float()

        # Flatten conv kernels [Out, In, H, W] -> [Out, In*H*W]
        if w_a.ndim == 4:
            w_a = w_a.view(w_a.shape[0], -1)
            w_b = w_b.view(w_b.shape[0], -1)

        delta = w_b - w_a
        if torch.norm(delta) < 1e-6:
            return

        try:
            U_a, S_a, _ = torch.linalg.svd(w_a, full_matrices=False)
            U_b, S_b, _ = torch.linalg.svd(w_b, full_matrices=False)
            U_a = U_a[:, :10]
            U_b = U_b[:, :10]
        except Exception as exc:
            logger.debug("SVD failed for layer: %s", exc)
            return

        # Delta energy projected onto each model's subspace
        en_a = float(torch.norm(torch.matmul(U_a.T, delta)) ** 2)
        en_b = float(torch.norm(torch.matmul(U_b.T, delta)) ** 2)

        # Spectral concentration (first singular value / sum)
        conc_a = float(S_a[0] / (torch.sum(S_a) + 1e-9))
        conc_b = float(S_b[0] / (torch.sum(S_b) + 1e-9))

        max_e = max(en_a, en_b)
        min_e = min(en_a, en_b) + 1e-9
        ratio = max_e / min_e

        if ratio > 1.01:
            # Additive mode: trust delta direction
            if en_b > en_a:
                self.votes_a += 1
            else:
                self.votes_b += 1
            self.mode_additive += 1
        else:
            # Subtractive / refining mode: trust spectral concentration
            if conc_b > conc_a:
                self.votes_a += 1
            else:
                self.votes_b += 1
            self.mode_subtractive += 1

        self.n_layers += 1

    def get_verdict(self) -> Dict[str, Any]:
        """Aggregate layer votes into a final directional verdict."""
        if self.n_layers == 0:
            return {"error": "0 layers analysed", "n_layers": 0}

        is_a_father = self.votes_a > self.votes_b
        total = self.votes_a + self.votes_b + 1e-9
        ratio = self.votes_a / total
        conf = abs(ratio - 0.5) * 2 * 100.0

        return {
            "father": self.name_a if is_a_father else self.name_b,
            "son": self.name_b if is_a_father else self.name_a,
            "conf": min(conf, 99.9),
            "mode_additive": self.mode_additive,
            "mode_subtractive": self.mode_subtractive,
            "n_layers": self.n_layers,
            "error": None,
        }


# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------

def compare_model_pair(
    path_a: str,
    path_b: str,
    exclude_value: bool = False,
) -> Dict[str, Any]:
    """
    Compare two model weight files/directories and return a directional verdict.

    Returns a dict with keys:
        father, son, conf, mode_additive, mode_subtractive, n_layers, time, error
    """
    sd_a = load_model_weights(path_a)
    sd_b = load_model_weights(path_b)

    if sd_a is None or sd_b is None:
        missing = path_a if sd_a is None else path_b
        return {"error": f"Failed to load weights: {missing}", "time": 0.0, "n_layers": 0}

    map_a = {normalize_key(k): v for k, v in sd_a.items()}
    map_b = {normalize_key(k): v for k, v in sd_b.items()}

    name_a = os.path.basename(path_a.rstrip("/"))
    name_b = os.path.basename(path_b.rstrip("/"))

    engine = LineageEngineV2_6(name_a, name_b)
    shared_keys = set(map_a.keys()) & set(map_b.keys())

    start = time.perf_counter()
    for k in shared_keys:
        wa, wb = map_a[k], map_b[k]
        if is_integrator_layer(k, wa.shape, exclude_value) and wa.shape == wb.shape:
            engine.analyze_layer(wa, wb)
    elapsed = time.perf_counter() - start

    verdict = engine.get_verdict()
    verdict["time"] = elapsed

    logger.debug(
        "Pair (%s, %s): father=%s conf=%.1f%% layers=%d time=%.3fs",
        name_a,
        name_b,
        verdict.get("father"),
        verdict.get("conf", 0.0),
        verdict.get("n_layers", 0),
        elapsed,
    )
    return verdict


# ---------------------------------------------------------------------------
# process_lineage_graph — main public interface
# ---------------------------------------------------------------------------

def process_lineage_graph(
    input_graph: nx.DiGraph,
    penalty: float,
    exclude_value: bool,
) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """
    Delta-only genealogy inference over a model family graph.

    For every undirected node pair in *input_graph* the function:
      1. Runs delta/SVD-based pairwise inference (LineageEngineV2_6).
      2. Adds the inferred **correct** directed edge (father→son) with
         ``weight = confidence`` and ``is_correct_direction = True``.
      3. Adds the **reverse** edge (son→father) with
         ``weight = confidence + penalty`` and ``is_correct_direction = False``.

    Args:
        input_graph: DiGraph whose nodes have a ``path`` attribute pointing to
                     the model weights (file or sharded directory).
        penalty:     Non-negative float added to the reverse-edge weight so that
                     downstream algorithms can distinguish inferred direction.
        exclude_value: When True, Value projection layers are excluded from the
                     geometric analysis.

    Returns:
        output_graph: Annotated DiGraph with inferred edges.
        stats:        Aggregate statistics dictionary compatible with the
                      benchmark script (see scripts/benchmark_lineage.py).
    """
    output_graph = nx.DiGraph()
    output_graph.add_nodes_from(input_graph.nodes(data=True))

    nodes = list(input_graph.nodes())
    total_start = time.perf_counter()

    # Per-pair accumulators
    pair_times: list = []
    valid_layers_list: list = []
    confidences: list = []
    correct_edges = 0
    reversed_edges = 0
    total_additive = 0
    total_subtractive = 0

    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if j <= i:
                continue  # process each undirected pair once

            path_u: str = input_graph.nodes[u].get("path", str(u))
            path_v: str = input_graph.nodes[v].get("path", str(v))

            verdict = compare_model_pair(path_u, path_v, exclude_value=exclude_value)

            pair_time = verdict.get("time", 0.0)
            n_layers = verdict.get("n_layers", 0)

            # Diagnostic logging (requirement: per-pair algebraic time + valid layers)
            logger.info(
                "[delta-v01] pair (%s, %s) → algebraic_time=%.4fs valid_layers=%d",
                u,
                v,
                pair_time,
                n_layers,
            )

            pair_times.append(pair_time)
            valid_layers_list.append(n_layers)

            if verdict.get("error"):
                logger.warning("Skipping pair (%s, %s): %s", u, v, verdict["error"])
                continue

            father_name = verdict["father"]
            conf = float(verdict["conf"])
            confidences.append(conf)
            total_additive += verdict.get("mode_additive", 0)
            total_subtractive += verdict.get("mode_subtractive", 0)

            # Resolve which node id is the father
            name_u = os.path.basename(path_u.rstrip("/"))
            name_v = os.path.basename(path_v.rstrip("/"))

            if father_name == name_u:
                father_node, son_node = u, v
            else:
                father_node, son_node = v, u

            base_distance = max(100.0 - conf, 0.1)

            # Correct directed edge (father → son)
            output_graph.add_edge(
                father_node,
                son_node,
                weight=conf,
                distance=base_distance,
                confidence=conf,
                is_correct_direction=True,
                valid_layers=n_layers,
                pair_time=pair_time,
            )
            correct_edges += 1

            # Penalised reverse edge (son → father)
            output_graph.add_edge(
                son_node,
                father_node,
                weight=conf + penalty,
                distance=base_distance,
                confidence=conf,
                is_correct_direction=False,
                valid_layers=n_layers,
                pair_time=pair_time,
            )
            reversed_edges += 1

    total_elapsed = time.perf_counter() - total_start

    stats: Dict[str, Any] = {
        # Timing
        "total_execution_time_s": round(total_elapsed, 4),
        "mean_pair_time_s": round(float(np.mean(pair_times)) if pair_times else 0.0, 4),
        # Layers
        "mean_valid_layers": round(float(np.mean(valid_layers_list)) if valid_layers_list else 0.0, 2),
        # Confidence
        "mean_confidence_pct": round(float(np.mean(confidences)) if confidences else 0.0, 2),
        # Edge counts
        "correct_directed_edges": correct_edges,
        "penalised_reverse_edges": reversed_edges,
        # Mode counts
        "total_additive_layers": total_additive,
        "total_subtractive_layers": total_subtractive,
        "additive_subtractive_ratio": (
            round(total_additive / (total_subtractive + 1e-9), 3)
        ),
        # Config
        "penalty": penalty,
        "exclude_value": exclude_value,
        "algorithm": "mother-with-delta-v01",
    }

    logger.info(
        "[delta-v01] finished: pairs=%d correct_edges=%d total_time=%.2fs",
        len(pair_times),
        correct_edges,
        total_elapsed,
    )

    return output_graph, stats
