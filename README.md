# MANGROVE — Model Lake Versioning System

MANGROVE is a full-stack system for **versioning and managing machine learning models**.  
Unlike traditional model registries that rely mainly on metadata, MANGROVE focuses on **inferring and tracking model lineage (“heritage”) from model weights**, inspired by the approach introduced in *“Unsupervised Model Tree Heritage Recovery”* (Horwitz, Shul, Hoshen).

## Overview

MANGROVE supports the management of the ML model lifecycle by providing:

- **Model version tracking**
- **Model lineage / heritage management** (relationships between models)
- **Reproducibility and traceability** of model evolution over time
- A web-based interface and REST APIs for interacting with the system

## Architecture

The project is organized into three main components.

### Backend (`model_heritage_backend/`)
- **Language**: Python
- **API**: REST services (Flask/FastAPI-based)
- **Database**: SQLite (`app.db`)
- **Responsibilities**: model management, versioning logic, statistical computations

### Frontend (`model_heritage_frontend/`)
- **Language**: JavaScript
- **Package manager**: `pnpm`
- **Responsibilities**: web UI for system interaction

### Utilities & Scripts
- Shell scripts for setup and run automation
- Additional documentation in `docs/`

## Key Features

### Model Versioning
- Track multiple versions of ML models
- Maintain model heritage and dependencies between versions/models
- Store relevant information for each version

### Statistical Analysis
- Compute advanced metrics (e.g., kurtosis) via `calculate_kurtosi_2_model`
- Enable comparative analyses across model versions

### RESTful APIs
- CRUD endpoints for model management
- Authentication/authorization support (where configured)
- JSON serialization for interoperability

## Installation & Setup

### Automatic Setup
```bash
# Full environment setup
./setup_and_run.sh
```

### Manual Setup

#### Backend
```bash
cd model_heritage_backend
pip install -r requirements.txt
python run_server.py
```

#### Frontend
```bash
cd model_heritage_frontend
pnpm install
pnpm start
```

## Running

```bash
# Standard startup
./run.sh

# Startup with debugging
python run_with_debug.py
```

## Project Structure

```text
Model_Graph/
├── model_heritage_backend/      # Python API server
│   ├── src/                     # Backend source code
│   ├── requirements.txt         # Python dependencies
│   └── run_server.py            # Server entry point
├── model_heritage_frontend/     # Web interface
├── docs/                        # Technical documentation
├── app.db                       # SQLite database
├── run.sh                       # Startup script
├── setup_and_run.sh             # Automated setup
└── run_with_debug.py            # Debug mode
```

## Technologies Used

- **Backend**: Python, Flask/FastAPI, SQLAlchemy
- **Frontend**: JavaScript, Node.js, pnpm
- **Database**: SQLite
- **DevOps/Automation**: Shell scripts, Git
- **Packaging**: `requirements.txt`, `package.json`

## Usage (Typical Workflow)

1. Start the system: `./setup_and_run.sh`
2. Open the web UI in your browser
3. Upload/register models via API or UI
4. Manage versions and track model heritage
5. Compare versions through computed metrics

## Contributing

Contributions are welcome on both backend and frontend. The separation of concerns between components supports parallel development and easier maintenance.

---

*MANGROVE aims to provide a practical, end-to-end solution for model lifecycle management with an emphasis on weight-based lineage tracking.*
