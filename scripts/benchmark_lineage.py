#!/usr/bin/env python3
"""
Benchmark for the delta-only lineage variant ``mother-with-delta-v01``.

Runs ``process_lineage_graph`` on an input graph and compares the result with a
ground truth, then prints (and optionally writes) two paper-ready tables in
Markdown and LaTeX:

  Table 1 - Comparative Performance Metrics
  Table 2 - Structural & Directional Analysis

Input graph (--graph), JSON, node-link format::

    {"nodes": [{"id": "modelA", "path": "/data/modelA.safetensors"},
               {"id": "modelB", "path": "/data/shards/modelB"}],
     "edges": [{"source": "modelA", "target": "modelB"}]}

``path`` may point to a single weights file or to a sharded directory. Relative
paths are resolved against --models-dir. If ``edges`` is empty/missing, all node
pairs are tested.

Ground truth (--ground-truth), JSON, one of:
  * {"edges": [["parent", "child"], ...]}
  * the freccia_del_tempo format: {"root": ..., "d1": ..., "d2": {parent: [children]}, ...}
    (optionally wrapped in {"TREES": {"<id>": {...}}}, select with --tree-id)

Example::

    python scripts/benchmark_lineage.py --graph graph.json --ground-truth gt.json \
        --penalty 0.5 --output-dir results/
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
from typing import Any, Dict, List, Set, Tuple

import networkx as nx

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DELTA_MODULE = os.path.join(REPO_ROOT, "model_heritage_backend", "src",
                            "mother_algorithm", "delta_lineage.py")
ALGO_NAME = "mother-with-delta-v01"


def load_delta_module():
    """Load delta_lineage.py directly (avoids importing the Neo4j-dependent backend package)."""
    spec = importlib.util.spec_from_file_location("delta_lineage", DELTA_MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def load_graph(path: str, models_dir: str) -> nx.DiGraph:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    g = nx.DiGraph()
    for n in data.get("nodes", []):
        nid = n["id"]
        p = n.get("path") or n.get("file_path") or nid
        if models_dir and not os.path.isabs(p):
            p = os.path.join(models_dir, p)
        attrs = {k: v for k, v in n.items() if k not in ("id", "path", "file_path")}
        g.add_node(nid, path=p, **attrs)
    for e in data.get("edges", data.get("links", [])):
        g.add_edge(e["source"], e["target"])
    if g.number_of_nodes() < 2:
        sys.exit("Graph must contain at least 2 nodes")
    return g


def load_ground_truth(path: str, tree_id=None) -> Set[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "TREES" in data:
        trees = data["TREES"]
        if tree_id is None:
            if len(trees) != 1:
                sys.exit(f"--tree-id required, available: {list(trees)}")
            data = next(iter(trees.values()))
        else:
            data = trees[str(tree_id)]
    if "edges" in data:
        return {(str(p), str(c)) for p, c in data["edges"]}
    pairs: Set[Tuple[str, str]] = set()
    root = data.get("root")
    for level in ("d1", "d2", "d3", "d4"):
        block = data.get(level)
        if block is None:
            continue
        if isinstance(block, list):
            pairs.update((str(root), str(c)) for c in block)
        else:
            for parent, children in block.items():
                pairs.update((str(parent), str(c)) for c in children)
    return pairs


def _resolve(name: str, node_ids: Dict[str, Any]):
    if name in node_ids:
        return node_ids[name]
    base = os.path.basename(name)
    return node_ids.get(base)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate(out: nx.DiGraph, gt: Set[Tuple[str, str]], g: nx.DiGraph) -> Dict[str, Any]:
    ids = {str(n): n for n in g.nodes()}
    ids.update({os.path.basename(str(n)): n for n in g.nodes()})
    evaluated = correct = 0
    unmatched: List[Tuple[str, str]] = []
    for p, c in gt:
        pn, cn = _resolve(p, ids), _resolve(c, ids)
        if pn is None or cn is None or not (out.has_edge(pn, cn) or out.has_edge(cn, pn)):
            unmatched.append((p, c))
            continue
        evaluated += 1
        if out.has_edge(pn, cn) and out[pn][cn].get("is_correct_direction"):
            correct += 1
    return {"gt_pairs": len(gt), "evaluated": evaluated, "correct": correct,
            "unmatched": unmatched,
            "accuracy": (100.0 * correct / evaluated) if evaluated else 0.0}


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def _fmt_ratio(r: float) -> str:
    return "inf" if r == float("inf") else f"{r:.2f}"


def build_tables(stats: Dict[str, Any], ev: Dict[str, Any]) -> Tuple[Tuple[List[str], List[List[str]]],
                                                                    Tuple[List[str], List[List[str]]]]:
    h1 = ["Algorithm/Version", "Accuracy (%) vs GT", "Mean Confidence (%)",
          "Mean Pair Processing Time (s)", "Total Execution Time (s)", "Mean Layers Used per Model"]
    r1 = [[ALGO_NAME, f"{ev['accuracy']:.2f}", f"{stats['mean_confidence']:.2f}",
           f"{stats['mean_process_time']:.4f}", f"{stats['total_time']:.2f}",
           f"{stats['mean_layers_per_model']:.1f}"]]
    h2 = ["Algorithm/Version", "Correct Directed Edges Identified", "Penalized Reverse Edges",
          "Additive/Subtractive Mode Ratio", "Penalty Factor Impact (applied penalty on weight)"]
    impact = stats["mean_reverse_weight"] - stats["mean_forward_weight"]
    r2 = [[ALGO_NAME, f"{ev['correct']}/{ev['evaluated']}", str(stats["penalized_reverse_edges"]),
           f"{_fmt_ratio(stats['additive_subtractive_ratio'])} "
           f"({stats['additive_layers']}/{stats['subtractive_layers']})",
           f"+{impact:.3f} (penalty={stats['penalty']})"]]
    return (h1, r1), (h2, r2)


def to_markdown(title: str, headers: List[str], rows: List[List[str]]) -> str:
    lines = [f"**{title}**", "", "| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _tex(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}")):
        if a == "\\":
            s = s.replace(a, b)
        else:
            s = s.replace(a, b)
    return s


def to_latex(caption: str, label: str, headers: List[str], rows: List[List[str]]) -> str:
    cols = "l" + "r" * (len(headers) - 1)
    out = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{_tex(caption)}}}",
           rf"\label{{{label}}}", r"\resizebox{\linewidth}{!}{%", rf"\begin{{tabular}}{{{cols}}}",
           r"\toprule", " & ".join(_tex(h) for h in headers) + r" \\", r"\midrule"]
    out += [" & ".join(_tex(c) for c in r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True, help="input graph JSON (node-link, nodes carry 'path')")
    ap.add_argument("--ground-truth", required=True, help="ground truth JSON")
    ap.add_argument("--tree-id", default=None, help="tree id inside a multi-tree ground truth")
    ap.add_argument("--models-dir", default="", help="base dir for relative node paths")
    ap.add_argument("--penalty", type=float, default=0.5, help="penalty added to child->parent edges")
    ap.add_argument("--exclude-value", action="store_true", help="exclude V-projection layers")
    ap.add_argument("--output-dir", default=None, help="write tables.md / tables.tex / stats.json here")
    ap.add_argument("-v", "--verbose", action="store_true", help="log per-pair process_time / valid_layers")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")

    delta = load_delta_module()
    g = load_graph(args.graph, args.models_dir)
    gt = load_ground_truth(args.ground_truth, args.tree_id)

    out, stats = delta.process_lineage_graph(g, args.penalty, args.exclude_value)
    ev = evaluate(out, gt, g)
    (h1, r1), (h2, r2) = build_tables(stats, ev)

    t1 = "Table 1: Comparative Performance Metrics"
    t2 = "Table 2: Structural & Directional Analysis"
    md = to_markdown(t1, h1, r1) + "\n" + to_markdown(t2, h2, r2)
    tex = (to_latex(t1, "tab:performance", h1, r1) + "\n" + to_latex(t2, "tab:structure", h2, r2))

    print(md)
    print(tex)
    if ev["unmatched"]:
        print(f"[warn] {len(ev['unmatched'])} ground-truth pairs not evaluated "
              f"(unknown node or pair skipped)", file=sys.stderr)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "tables.md"), "w", encoding="utf-8") as f:
            f.write(md)
        with open(os.path.join(args.output_dir, "tables.tex"), "w", encoding="utf-8") as f:
            f.write(tex)
        with open(os.path.join(args.output_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump({"stats": stats, "evaluation": ev}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
