# MANGROVE🌿 — Model Lake Versioning Models

<div align="center">

<a href="#link-to-the-demo-paper">⚙️Demo paper</a>   |  
<a href="#link-to-the-technical-report">📄Technical report</a>   |  
<a href="#link-to-the-video">🎬Video demonstration</a>

</div>

MANGROVE is a full‑stack system for curating, versioning, and auditing a “model lake” of machine-learning checkpoints, **specifically designed for environments dominated by iterative fine-tuning operations**. 

Unlike traditional model registries that rely primarily on human-provided metadata, MANGROVE emphasizes weight-based fingerprinting and lineage reconstruction. The system helps organize collections of checkpoints into families and recovers plausible parent–child relations. The resulting evolutionary structure is stored as a graph in Neo4j.

By reworking and optimizing these methods within an incremental setting, this work challenges the model‑lake paradigm and addresses the additional complexities introduced by big‑data environments where thousands of fine-tuned derivatives stem from a few foundation models. The proposed and implemented approach is based on the incremental ingestion of models which, under a weight‑only assumption, are processed and assigned to the appropriate lineage. This is followed by a more complex genealogy recovery stage aimed at reconstructing parent–child relationships and broader evolutionary dependencies. The system, built on a Neo4j‑based infrastructure, is supported by two main methodological components: (i) a centroid‑based family assignment strategy for adaptive clustering in a model lake, and (ii) an unsupervised, optimized algorithm for model genealogy recovery.

<img width="1895" height="908" alt="image" src="https://github.com/user-attachments/assets/a846021c-6994-4991-9c26-cd001bc50524" />

All the technical details about the methodology and the implementations are discussed in the technical report and the demo paper. 

---

## Table of contents

- [System overview](#system-overview)
- [Architecture](#architecture)
- [Graph data model (Neo4j)](#graph-data-model-neo4j)
- [Repository structure](#repository-structure)
- [Quickstart (recommended)](#quickstart-recommended)
- [Reproducibility notes](#reproducibility-notes)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## System overview

<img width="1280" height="322" alt="image" src="https://github.com/user-attachments/assets/8d5134e8-39bf-4f99-89fe-6536a611f9a2" />

---

## Architecture

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/8e7a6b79-edd2-406f-9000-3bef24c57cd6" />


### Backend (`model_heritage_backend/`)
- **Framework**: Flask
- **Graph DB**: Neo4j (Bolt)
- **Responsibilities**:
  - file ingestion (binary / sharded / zip)
  - conversion to `.safetensors`
  - normalization of layer names and mapping persistence
  - checksum + structural signature + kurtosis computation
  - centroid-based clustering
  - genealogy recovery
  - REST API for model management
  - NL → Cypher endpoint (optional; requires API key)


### Frontend (`model_heritage_frontend/`)
- **Framework**: React + Vite
- **Visualization**: `vis-network`
- **Markdown rendering**: `react-markdown` (+ `remark-gfm`)
- **Routing**: `react-router-dom`

---

## Graph data model (Neo4j)

The backend uses Neo4j as the authoritative persistence layer.

### Node labels
- `Model`
- `Family`
- `Centroid`

### Relationship types
- `IS_CHILD_OF` (child model → parent model)
- `BELONGS_TO` (model → family)
- `HAS_CENTROID` (family → centroid)

---

## Repository structure

```text
.
├── model_heritage_backend/       # Flask backend (API + ingestion + Neo4j integration)
├── model_heritage_frontend/      # React + Vite frontend
├── docs/                         # Documentation folder
├── requirements.txt              # Python dependencies (global)
├── setup_and_run.sh              # One-shot setup + run script
├── run.sh                        # Run script (expects venv already created)
└── README.md 
```
## Quickstart (recommended)
work in progress

## Reproducibility notes

Dependencies: All packages are strictly pinned in requirements.txt and model_heritage_frontend/package.json to ensure reproducible builds.

Hardware Requirements: Checkpoint ingestion and conversion (especially for PyTorch .bin to .safetensors) is memory-intensive. We recommend at least 16GB of system RAM for processing models up to 1B parameters.
For intensive usage we recommend greater computational power and increased storage capacity.

Testing: To evaluate the system locally without downloading massive checkpoints, we recommend using small toy models or lightweight derivatives (e.g., TinyLlama or BERT fine-tunes).

## Citation
If you use MANGROVE in academic work, please cite this repository.

```bibtex
@software{mangrove_2026,
  title  = {MANGROVE: Model Lake Versioning Models},
  author = {ilTrabba, gabrulele},
  year   = {2026},
  url    = {[https://github.com/ilTrabba/Mangrove](https://github.com/ilTrabba/Mangrove)}
}
```
## Acknowledgments
This project relies heavily on the open-source ecosystem, including:

Neo4j for graph persistence

Flask & React + Vite for the application layer

PyTorch & safetensors tooling for checkpoint handling

LangChain & Groq for LLM-powered NL → Cypher capabilities

## License
See the repository license (if provided). If no license file is present, all rights are reserved by default.
