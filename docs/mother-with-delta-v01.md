# mother-with-delta-v01: Delta-Only Lineage Inference

This document describes the **mother-with-delta-v01** variant of the Mangrove genealogy recovery algorithm and how to run it alongside the comparative benchmark tool.

---

## Overview

`mother-with-delta-v01` infers parent–child relationships between fine-tuned deep-learning models using **only delta-based geometric reasoning** — no per-family kurtosis computation is performed.

For every undirected model pair *(u, v)* in a family graph the engine:

1. Loads weight tensors from both models (supports `.safetensors`, `.ckpt`, `.bin`, `.pt`, `.pth` and sharded directories).
2. Aligns layers by normalised key names.
3. Applies `LineageEngineV2_6`: for each eligible 2-D / 4-D layer, computes the SVD of both weight matrices and uses delta-energy projection and spectral concentration to cast a directional vote.
4. Aggregates layer votes into a confidence score and decides the direction (father → son).
5. Writes the inferred correct edge and a penalised reverse edge into the output `nx.DiGraph`.

---

## Module location

```
model_heritage_backend/src/clustering/delta_graph_engine.py
```

### Public interface

```python
from src.clustering.delta_graph_engine import process_lineage_graph

output_graph, stats = process_lineage_graph(
    input_graph,   # nx.DiGraph — nodes carry a "path" attribute
    penalty,       # float — added to reverse-edge weight
    exclude_value, # bool  — exclude Value projection layers
)
```

The function signature is **identical** to the existing pipeline contract:

```python
process_lineage_graph(
    input_graph: nx.DiGraph,
    penalty: float,
    exclude_value: bool,
) -> Tuple[nx.DiGraph, Dict[str, Any]]
```

### Node attribute requirements

Each node in `input_graph` must carry a `"path"` attribute pointing to the model weights file or sharded directory:

```python
g = nx.DiGraph()
g.add_node("modelA", path="/data/models/modelA")
g.add_node("modelB", path="/data/models/modelB")
```

---

## Running the delta-only inference

```python
import networkx as nx
from src.clustering.delta_graph_engine import process_lineage_graph

g = nx.DiGraph()
g.add_node("base",   path="/models/base_model.safetensors")
g.add_node("finetune", path="/models/finetuned_model.safetensors")

output_graph, stats = process_lineage_graph(g, penalty=10.0, exclude_value=False)

for u, v, data in output_graph.edges(data=True):
    print(u, "->", v, data)

print(stats)
```

---

## Running the benchmark

The benchmark script evaluates one or more algorithm versions and outputs **Table 1** (Performance Metrics) and **Table 2** (Structural & Directional Analysis) in both Markdown and LaTeX.

### Graph JSON format

```json
{
    "modelA": {"path": "/data/modelA"},
    "modelB": {"path": "/data/modelB"},
    "modelC": {"path": "/data/modelC"}
}
```

### Ground-truth JSON format

```json
[["modelA", "modelB"], ["modelA", "modelC"]]
```

Each entry is a `[parent, child]` pair.

### Command

```bash
python scripts/benchmark_lineage.py \
    --graph   /path/to/family_graph.json \
    --gt      /path/to/ground_truth.json \
    --penalty 10.0 \
    --out-md  results/tables.md \
    --out-tex results/tables.tex
```

#### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--graph` | *(required)* | Path to graph JSON |
| `--gt` | `None` | Path to ground-truth JSON (precision set to 0 if omitted) |
| `--penalty` | `10.0` | Penalty added to reverse-edge `weight` |
| `--exclude-value` | `False` | Exclude Value projection layers |
| `--algorithms` | all | Algorithm versions to run |
| `--out-md` | `None` | Save Markdown tables to file |
| `--out-tex` | `None` | Save LaTeX tables to file |

---

## Expected output artifacts

| Artifact | Description |
|----------|-------------|
| `results/tables.md` | Markdown tables ready for README/GitHub |
| `results/tables.tex` | LaTeX `tabular` blocks ready for paper |
| stdout | Live Markdown table + LaTeX (always printed) |

### Example Markdown output

```
## Table 1: Comparative Performance Metrics

| Algorithm/Version     | Precision (%) | Mean Confidence (%) | Mean Pair Time (s) | Total Exec Time (s) | Mean Layers Used |
| --------------------- | ------------- | ------------------- | ------------------ | ------------------- | ---------------- |
| mother-with-delta-v01 | 83.33         | 76.42               | 1.2301             | 3.6903              | 48.00            |

## Table 2: Structural & Directional Analysis

| Algorithm/Version     | Correct Directed Edges | Penalised Reverse Edges | Additive/Subtractive Ratio | Penalty Factor |
| --------------------- | ---------------------- | ----------------------- | -------------------------- | -------------- |
| mother-with-delta-v01 | 3                      | 3                       | 2.140                      | 10.0           |
```

---

## Design notes

- **No kurtosis**: direction inference relies entirely on the delta/SVD engine — the existing kurtosis-based `MoTHerTreeBuilder` is untouched.
- **Penalty convention**: the reverse edge carries `weight = confidence + penalty`; algorithms that **minimise** weight will correctly prefer the inferred direction.
- **Diagnostic logging**: per-pair algebraic processing time and valid layer count are logged at `INFO` level and included in the `stats` dict.
- **Sharded models**: `load_model_weights` transparently merges sharded directories into a single state dict.
