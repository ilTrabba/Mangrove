#!/bin/bash

# ================= CONFIGURAZIONE =================
REPO_DIR="$HOME/projects/Model_Graph"
API_URL="http://localhost:5002/api/models"
TIMEOUT=30  # Secondi massimi di attesa avvio server
# ==================================================

# 1. Controlla che sia stato passato un file
FILE_PATH="$1"

if [ -z "$FILE_PATH" ]; then
    echo "❌ Errore: Devi specificare il percorso del file."
    echo "Uso: ./ingest.sh /path/to/model.safetensors"
    exit 1
fi

# Converti in path assoluto per evitare problemi
FILE_PATH=$(realpath "$FILE_PATH")

if [ ! -f "$FILE_PATH" ]; then
    echo "❌ Errore: Il file '$FILE_PATH' non esiste."
    exit 1
fi

echo "=========================================="
echo "🚀 AVVIO PROCEDURA DI INGESTIONE"
echo "📂 File: $(basename "$FILE_PATH")"
echo "=========================================="

# 2. Spostati nella cartella del repository e avvia run.sh in background
cd "$REPO_DIR" || { echo "❌ Cartella repo non trovata"; exit 1; }

echo "⏳ Avvio del backend (run.sh)..."
# Avvia run.sh in background, nascondendo i log standard per pulizia (o togli > /dev/null per debug)
./run.sh > /dev/null 2>&1 &
SERVER_PID=$!

# 3. Attendi che il server sia pronto (polling sulla porta 5001)
echo "⏳ Attesa disponibilità server..."
START_TIME=$(date +%s)
SERVER_READY=false

while true; do
    # Prova a chiamare la root o l'API. Se risponde 200/404/405 è vivo.
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL")
    
    if [[ "$HTTP_CODE" != "000" ]]; then
        SERVER_READY=true
        break
    fi

    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        break
    fi
    
    sleep 2
done

if [ "$SERVER_READY" = false ]; then
    echo "❌ Timeout: Il server non si è avviato entro $TIMEOUT secondi."
    kill $SERVER_PID
    exit 1
fi

echo "✅ Backend attivo! Inizio upload..."

# 4. Esegui l'upload con cURL
# Estrae nome file e cartella genitore per simulare i metadati
FILENAME=$(basename "$FILE_PATH")
PARENT_DIR=$(basename "$(dirname "$FILE_PATH")")

# Nota: -F "file=@..." è la sintassi curl per inviare file
RESPONSE=$(curl -s -X POST "$API_URL" \
     -F "file=@$FILE_PATH" \
     -F "name=$FILENAME" \
     -F "description=Uploaded via CLI script from $PARENT_DIR")

echo ""
echo "📤 Risposta dal server:"
echo "$RESPONSE"

# 5. Pulizia: Uccidi il server
echo "=========================================="
echo "🛑 Arresto del backend (PID $SERVER_PID)..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null

echo "✅ Finito."