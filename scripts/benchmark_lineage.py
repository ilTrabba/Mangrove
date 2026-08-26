#!/usr/bin/env python3
"""
benchmark_lineage.py — Comparative benchmark for lineage inference algorithms.

Evaluates one or more algorithm versions and generates publication-ready output
in both Markdown and LaTeX tabular format.

Usage
-----
    python scripts/benchmark_lineage.py \
        --graph   path/to/family_graph.json \
        --gt      path/to/ground_truth.json \
        --penalty 10.0 \
        --out-md  results/tables.md \
        --out-tex results/tables.tex

Graph JSON format
-----------------
A JSON object where each key is a node name and the value is a dict with at
least a "path" key pointing to model weights::

    {
        "modelA": {"path": "/data/modelA"},
        "modelB": {"path": "/data/modelB"},
        "modelC": {"path": "/data/modelC"}
    }

Ground-truth JSON format
------------------------
A list of [parent, child] pairs that represent the known correct directed
edges::

    [["modelA", "modelB"], ["modelA", "modelC"]]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("benchmark_lineage")


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

def _run_delta_v01(
    graph: nx.DiGraph,
    penalty: float,
    exclude_value: bool,
) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """Run the mother-with-delta-v01 variant."""
    # Import here so the benchmark can be run from any working directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_heritage_backend"))
    from src.clustering.delta_graph_engine import process_lineage_graph  # noqa: PLC0415
    return process_lineage_graph(graph, penalty=penalty, exclude_value=exclude_value)


ALGORITHMS = {
    "mother-with-delta-v01": _run_delta_v01,
}


# ---------------------------------------------------------------------------
# Graph / ground-truth helpers
# ---------------------------------------------------------------------------

def load_graph_from_json(path: str) -> nx.DiGraph:
    """Build a DiGraph from the benchmark graph JSON file."""
    with open(path, encoding="utf-8") as fh:
        data: Dict[str, Dict[str, Any]] = json.load(fh)

    g = nx.DiGraph()
    for node_name, attrs in data.items():
        g.add_node(node_name, **attrs)

    # Add a placeholder edge for every pair so process_lineage_graph can
    # iterate them; the engine replaces edges with inferred ones anyway.
    nodes = list(g.nodes())
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if j > i:
                g.add_edge(u, v)

    return g


def load_ground_truth(path: str) -> List[Tuple[str, str]]:
    """Load ground-truth parent→child pairs from a JSON file."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    pairs: List[Tuple[str, str]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append((str(item[0]), str(item[1])))
        elif isinstance(item, dict):
            pairs.append((str(item["parent"]), str(item["child"])))
    return pairs


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    output_graph: nx.DiGraph,
    stats: Dict[str, Any],
    gt_pairs: List[Tuple[str, str]],
    algorithm: str,
) -> Dict[str, Any]:
    """
    Compute accuracy / precision against ground truth and combine with the
    aggregate stats returned by process_lineage_graph.
    """
    correct_gt = 0
    for parent, child in gt_pairs:
        edge_data = output_graph.get_edge_data(parent, child)
        if edge_data and edge_data.get("is_correct_direction", False):
            correct_gt += 1

    total_gt = len(gt_pairs)
    precision_pct = (correct_gt / total_gt * 100.0) if total_gt else 0.0

    return {
        "algorithm": algorithm,
        "precision_pct": round(precision_pct, 2),
        "mean_confidence_pct": stats.get("mean_confidence_pct", 0.0),
        "mean_pair_time_s": stats.get("mean_pair_time_s", 0.0),
        "total_execution_time_s": stats.get("total_execution_time_s", 0.0),
        "mean_valid_layers": stats.get("mean_valid_layers", 0.0),
        # structural
        "correct_directed_edges": stats.get("correct_directed_edges", 0),
        "penalised_reverse_edges": stats.get("penalised_reverse_edges", 0),
        "additive_subtractive_ratio": stats.get("additive_subtractive_ratio", 0.0),
        "penalty": stats.get("penalty", 0.0),
    }


# ---------------------------------------------------------------------------
# Table formatters
# ---------------------------------------------------------------------------

_TABLE1_HEADERS = [
    "Algorithm/Version",
    "Precision (%)",
    "Mean Confidence (%)",
    "Mean Pair Time (s)",
    "Total Exec Time (s)",
    "Mean Layers Used",
]

_TABLE2_HEADERS = [
    "Algorithm/Version",
    "Correct Directed Edges",
    "Penalised Reverse Edges",
    "Additive/Subtractive Ratio",
    "Penalty Factor",
]


def _row1(m: Dict[str, Any]) -> List[str]:
    return [
        m["algorithm"],
        f"{m['precision_pct']:.2f}",
        f"{m['mean_confidence_pct']:.2f}",
        f"{m['mean_pair_time_s']:.4f}",
        f"{m['total_execution_time_s']:.4f}",
        f"{m['mean_valid_layers']:.2f}",
    ]


def _row2(m: Dict[str, Any]) -> List[str]:
    return [
        m["algorithm"],
        str(m["correct_directed_edges"]),
        str(m["penalised_reverse_edges"]),
        f"{m['additive_subtractive_ratio']:.3f}",
        str(m["penalty"]),
    ]


# --- Markdown ---

def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    data_lines = [
        "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep] + data_lines)


def build_markdown(metrics_list: List[Dict[str, Any]]) -> str:
    rows1 = [_row1(m) for m in metrics_list]
    rows2 = [_row2(m) for m in metrics_list]

    lines = [
        "## Table 1: Comparative Performance Metrics",
        "",
        _md_table(_TABLE1_HEADERS, rows1),
        "",
        "## Table 2: Structural & Directional Analysis",
        "",
        _md_table(_TABLE2_HEADERS, rows2),
        "",
    ]
    return "\n".join(lines)


# --- LaTeX ---

def _latex_table(caption: str, label: str, headers: List[str], rows: List[List[str]]) -> str:
    col_spec = "l" + "r" * (len(headers) - 1)
    header_row = " & ".join(f"\\textbf{{{h}}}" for h in headers) + r" \\"
    data_rows = [" & ".join(r) + r" \\" for r in rows]

    return "\n".join([
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        fr"\caption{{{caption}}}",
        fr"\label{{{label}}}",
        fr"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header_row,
        r"\midrule",
        *data_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])


def build_latex(metrics_list: List[Dict[str, Any]]) -> str:
    rows1 = [_row1(m) for m in metrics_list]
    rows2 = [_row2(m) for m in metrics_list]

    t1 = _latex_table(
        "Comparative Performance Metrics",
        "tab:performance",
        _TABLE1_HEADERS,
        rows1,
    )
    t2 = _latex_table(
        "Structural \\& Directional Analysis",
        "tab:structural",
        _TABLE2_HEADERS,
        rows2,
    )
    return t1 + "\n\n" + t2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark lineage inference algorithms and produce paper tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--graph", required=True, help="Path to graph JSON file.")
    parser.add_argument("--gt", required=False, default=None, help="Path to ground-truth JSON file.")
    parser.add_argument("--penalty", type=float, default=10.0, help="Penalty added to reverse-edge weight.")
    parser.add_argument("--exclude-value", action="store_true", help="Exclude Value projection layers.")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(ALGORITHMS.keys()),
        choices=list(ALGORITHMS.keys()),
        help="Algorithm versions to benchmark.",
    )
    parser.add_argument("--out-md", default=None, help="Write Markdown tables to this file.")
    parser.add_argument("--out-tex", default=None, help="Write LaTeX tables to this file.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    graph = load_graph_from_json(args.graph)
    gt_pairs = load_ground_truth(args.gt) if args.gt else []

    metrics_list: List[Dict[str, Any]] = []

    for algo_name in args.algorithms:
        run_fn = ALGORITHMS[algo_name]
        print(f"Running {algo_name} …", flush=True)
        t0 = time.perf_counter()
        out_graph, stats = run_fn(graph.copy(), args.penalty, args.exclude_value)
        wall = time.perf_counter() - t0
        print(f"  done in {wall:.2f}s", flush=True)

        m = compute_metrics(out_graph, stats, gt_pairs, algo_name)
        metrics_list.append(m)

    # Sort deterministically by algorithm name
    metrics_list.sort(key=lambda x: x["algorithm"])

    md_text = build_markdown(metrics_list)
    tex_text = build_latex(metrics_list)

    # Always print to stdout
    print("\n" + md_text)

    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(md_text, encoding="utf-8")
        print(f"Markdown saved to {args.out_md}")

    if args.out_tex:
        Path(args.out_tex).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_tex).write_text(tex_text, encoding="utf-8")
        print(f"LaTeX saved to {args.out_tex}")
    else:
        print("\n--- LaTeX ---\n")
        print(tex_text)


if __name__ == "__main__":
    main()
