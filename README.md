# Model Graph - Model Lake Versioning System

Un sistema completo per il versioning e la gestione di modelli di machine learning, sviluppato con un'architettura full-stack che combina Python per il backend e JavaScript per il frontend.

## 🎯 Panoramica

Model Graph è una piattaforma per la gestione del ciclo di vita dei modelli di machine learning, che fornisce funzionalità di versioning, tracciabilità e heritage dei modelli non basandosi sui metadatati ma sui pesi di un modello, seguendo come base l'approccio delineato nel paper "Unsupervised Model Tree Heritage Recovery" di Eliahu Horwitz, Asaf Shul, Yedid Hoshen. Il sistema permette di mantenere una cronologia completa delle modifiche ai modelli e delle loro relazioni.

## 🏗️ Architettura

Il progetto è strutturato in tre componenti principali:

### Backend (`model_heritage_backend/`)
- **Linguaggio**: Python (78.2% del codebase)
- **Framework**: Flask/FastAPI per le API REST
- **Database**: SQLite (`app.db`) per la persistenza dei dati
- **Funzionalità**: Gestione modelli, versioning, calcoli statistici

### Frontend (`model_heritage_frontend/`)
- **Linguaggio**: JavaScript (18.9% del codebase)
- **Gestione dipendenze**: pnpm
- **Interfaccia**: Web UI per l'interazione con il sistema

### Utility e Scripts
- **Shell scripts** (2.4%): Automazione setup e deployment
- **Documentazione**: Directory `docs/` per la documentazione tecnica

## 🚀 Funzionalità Principali

### Versioning dei Modelli
- Tracciamento delle versioni dei modelli ML
- Gestione dell'heritage e delle dipendenze tra modelli
- Metadati completi per ogni versione

### Analisi Statistiche
- Calcolo di metriche avanzate (es. curtosi) tramite `calculate_kurtosi_2_model`
- Supporto per analisi comparative tra versioni

### API RESTful
- Endpoint per CRUD operations sui modelli
- Sistema di autenticazione e autorizzazione
- Serializzazione JSON per l'interoperabilità

## 🛠️ Setup e Installazione

### Installazione Automatica
```bash
# Setup completo dell'ambiente
./setup_and_run.sh
```

### Installazione Manuale

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

### Esecuzione
```bash
# Avvio standard
./run.sh

# Avvio con debug
python run_with_debug.py
```

## 📁 Struttura del Progetto

```
Model_Graph/
├── model_heritage_backend/     # Server API Python
│   ├── src/                   # Codice sorgente backend
│   ├── requirements.txt       # Dipendenze Python
│   └── run_server.py         # Entry point server
├── model_heritage_frontend/    # Interfaccia web
├── docs/                      # Documentazione
├── app.db                     # Database SQLite
├── run.sh                     # Script di avvio
├── setup_and_run.sh          # Setup automatico
└── run_with_debug.py         # Modalità debug
```

## 🔧 Tecnologie Utilizzate

- **Backend**: Python, Flask/FastAPI, SQLAlchemy
- **Frontend**: JavaScript, Node.js, pnpm
- **Database**: SQLite
- **DevOps**: Shell scripting, Git versioning
- **Packaging**: Requirements.txt, package.json

## 🎮 Utilizzo

1. **Avvia il sistema**: `./setup_and_run.sh`
2. **Accedi all'interfaccia web** tramite il browser
3. **Carica modelli** attraverso l'API o l'interfaccia
4. **Gestisci versioni** e traccia l'heritage dei modelli
5. **Analizza metriche** e confronta performance

## 🤝 Contribuire

Il progetto è strutturato per supportare contributi sia sul frontend che sul backend. La separazione netta delle responsabilità facilita lo sviluppo parallelo e la manutenzione.

## 📊 Metriche del Progetto

- **Python**: 78.2% (Backend, API, ML utilities)
- **JavaScript**: 18.9% (Frontend, UI components)
- **Shell**: 2.4% (Automation, deployment)
- **Altri**: 0.5% (Configurazione, documentazione)

---

*Model Graph rappresenta una soluzione completa per il model lifecycle management, fornendo gli strumenti necessari per una gestione professionale dei modelli di machine learning in ambiente di produzione.*
