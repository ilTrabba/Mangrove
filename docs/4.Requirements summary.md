### Riepilogo dei Requisiti del Progetto: ModelGenealogy

**Obiettivo Principale:** Creare un sistema "Model Lake" per il versionamento automatico di modelli di Machine Learning, identificando le relazioni genitore-figlio basandosi esclusivamente sull'analisi dei pesi dei modelli, senza fare affidamento su metadati.

**Tecnologie Chiave:**
*   **Backend:** FastAPI (Python)
*   **Database:** Neo4j (per le relazioni tra modelli)
*   **Coda Asincrona:** Redis + RQ (per i processi di analisi)
*   **Frontend:** React (con Material-UI)
*   **Core Algorithm:** Basato sul paper MoTHer e l'implementazione di riferimento.
*   **Deployment:** Docker Compose

**Funzionalità Utente:**
1.  **Navigazione:** Gli utenti possono esplorare i modelli presenti nel sistema.
2.  **Ricerca:** Una barra di ricerca permette di filtrare i modelli per nome.
3.  **Visualizzazione:** Ogni modello ha una "model card" che mostra:
    *   Nome del modello.
    *   Modello "genitore" (da cui deriva).
    *   Modelli "figli" (derivati da esso).
4.  **Inserimento:** L'utente può caricare un nuovo modello. Il sistema avvierà in automatico la pipeline di versionamento per identificarne il genitore.

**Pipeline di Versionamento (in 4 fasi):**
1.  **Ingestion & Signature Extraction:** Quando un nuovo modello viene caricato, il sistema ne estrae una "firma" basata sulla struttura e sui pesi, senza usare metadati.
2.  **4-Step Clustering:**
    *   **Screening Strutturale:** Il nuovo modello viene confrontato rapidamente con le "famiglie" di modelli esistenti per trovare quelle strutturalmente compatibili.
    *   **Matching con Centroide:** Viene calcolata la distanza tra i pesi del nuovo modello e il centroide (modello medio) delle famiglie compatibili.
    *   **Validazione:** Se c'è incertezza, il modello viene confrontato con alcuni membri reali della famiglia candidata per conferma.
    *   **Update:** Il centroide della famiglia viene aggiornato in modo incrementale con il nuovo modello.
3.  **MoTHer Lineage (Intra-Family):** Una volta che un modello è assegnato a una famiglia, l'algoritmo MoTHer ricostruisce l'albero genealogico completo all'interno di quella famiglia, identificando le relazioni genitore-figlio più probabili.
4.  **Persistence su Neo4j:** Le relazioni identificate (genitore, figlio, famiglia) vengono salvate nel database a grafo Neo4j.

**Interfaccia Utente (UI):**
*   **Navbar:** Semplice, con il titolo "Model Heritage" e la sezione "Models".
*   **Sezione Models:**
    *   Mostra un catalogo di "model card" compatte (solo nome).
    *   Cliccando su una card, si apre una vista dettagliata con le informazioni di versionamento.
    *   Barra di ricerca per filtrare i modelli per nome.

Mi confermate che questa sintesi è corretta e completa?

