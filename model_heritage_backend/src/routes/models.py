import os
import hashlib
import uuid
import logging
import tempfile
import json
import io
import torch
import gc
import shutil
import zipfile
import torch

from pathlib import Path  
from safetensors import safe_open
from datetime import datetime, timezone
from src.log_handler import logHandler
from collections import Counter
from src.services.neo4j_service import neo4j_service
from src.config import Config
from src.clustering.model_management import ModelManagementSystem
from src.utils.sharded_file_error import ShardedFileError
from safetensors.torch import load_file as load_safetensors, save_file
from flask import send_file
from urllib.parse import urlparse
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from src.mother_algorithm.mother_utils import calc_ku
from src.utils.normalization_system import normalize_safetensors_layers, save_layer_mapping_json

### Setup instances initialization ###

logger = logging.getLogger(__name__)
models_bp = Blueprint('models', __name__)
mgmt_system = ModelManagementSystem()
sharded_file_error = ShardedFileError()

MODEL_FOLDER = Config.MODEL_FOLDER
README_FOLDER = 'readmes'
ALLOWED_EXTENSIONS = Config.ALLOWED_EXTENSIONS
ALLOWED_README_EXTENSIONS = {'md', 'txt'}
MAX_README_SIZE = 5 * 1024 * 1024  # 5MB

# In src/routes/models.py

@models_bp.route('/vm-files', methods=['GET'])
def list_vm_files():
    """Elenca i file .safetensors/.bin presenti in una cartella specifica della VM"""
    try:
        # Configura qui il percorso dove tieni i dataset nella VM
        VM_DATASET_ROOT = os.path.expanduser("~/projects/dataset") 
        
        if not os.path.exists(VM_DATASET_ROOT):
             return jsonify({'error': f'Path non trovato: {VM_DATASET_ROOT}'}), 404

        files_found = []
        for root, _, filenames in os.walk(VM_DATASET_ROOT):
            for filename in filenames:
                if filename.endswith(('.safetensors', '.bin', '.pt')):
                    full_path = os.path.join(root, filename)
                    # Restituisci path relativo o assoluto
                    files_found.append({
                        "name": filename,
                        "path": full_path,
                        "size": os.path.getsize(full_path),
                        "folder": os.path.basename(root)
                    })
        
        return jsonify({'files': files_found})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_readme_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_README_EXTENSIONS

def validate_url(url):
    """Validate URL format using urllib.parse"""
    if not url:
        return True  
    try:
        result = urlparse(url)
        return all([result.scheme in ('http', 'https'), result.netloc])
    except Exception:
        return False

def calculate_file_checksum(file_path):
    """Calculate SHA-256 checksum of file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def extract_weight_signature(file_path: str, num_layers: int) -> dict:
    """
    Estrae la signature strutturale. Versione ROBUSTA.
    """
    total_parameters = 0
    square_dims = Counter()
    norm_dims = Counter()  # <--- NUOVO: Conta le dimensioni delle norm
    
    with safe_open(file_path, framework="pt") as f:
        for key in f.keys():
            shape = f.get_slice(key).get_shape()
            
            # Calcolo parametri
            num_params = 1
            for dim in shape: num_params *= dim
            total_parameters += num_params
            
            # 1. Cerca Matrici Quadrate (Strategia Classica)
            if len(shape) == 2 and shape[0] == shape[1] and shape[0] >= 128:
                square_dims[shape[0]] += 1
            
            # 2. Cerca Normalizzazioni (Strategia Gemma/Fallback)
            # Deve essere 1D, >= 128 e avere "norm" nel nome
            if len(shape) == 1 and shape[0] >= 128:
                k_lower = key.lower()
                if "weight" in k_lower and ("norm" in k_lower or "ln" in k_lower):
                    norm_dims[shape[0]] += 1
    
    # LOGICA DI DEDUZIONE
    hidden_size = 0
    
    if square_dims:
        # Se ci sono matrici quadrate, usiamo quelle (più sicuro per modelli standard)
        hidden_size = square_dims.most_common(1)[0][0]
    elif norm_dims:
        # Se non ci sono matrici quadrate (Gemma), usiamo le norm
        hidden_size = norm_dims.most_common(1)[0][0]
    else:
        raise RuntimeError(
            f"Impossibile dedurre hidden_size dal file {os.path.basename(file_path)} "
            "(nessuna matrice quadrata o layer norm riconoscibile)."
        )
    
    # Hash strutturale
    base_string = f"{num_layers}_{hidden_size}"
    structural_hash = hashlib.md5(base_string.encode()).hexdigest()[:16]
    
    return {
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "structural_hash": structural_hash,
        "total_parameters": total_parameters
    }

def is_torch_file(file_path:  str) -> bool:
    """
    Verifica se un file è un checkpoint PyTorch leggendo solo l'header (magic bytes).
    NON carica il file in memoria.
    """
    TORCH_MAGIC = b'\x80\x02'  # Pickle protocol 2 (usato da torch.save)
    ZIP_MAGIC = b'PK'          # ZIP archive (PyTorch >= 1.6 usa ZIP)
    
    try:
        with open(file_path, 'rb') as f:
            header = f.read(2)
            return header in (TORCH_MAGIC, ZIP_MAGIC)
    except (IOError, OSError):
        return False

def is_safetensors_file(file_path: str) -> bool:
    """Verifica se un file è un safetensors leggendo l'header."""
    try:
        with open(file_path, 'rb') as f:
            # Safetensors inizia con 8 byte che indicano la lunghezza dell'header JSON
            header_size = int.from_bytes(f.read(8), 'little')
            return 0 < header_size < 100_000_000  # Header ragionevole
    except (IOError, OSError, ValueError):
        return False

def load_model_file(file_path: str) -> dict:
    """
    Carica un file modello (.safetensors, .bin, .pt, .pth, .ckpt).
    Restituisce un dizionario {nome_tensor: tensor}. 
    """
    ext = Path(file_path).suffix.lower()
    
    if ext == '.safetensors' or (ext == '' and is_safetensors_file(file_path)):
        return load_safetensors(file_path)
    
    # File PyTorch (.bin, .pt, .pth, .ckpt o senza estensione)
    data = torch.load(file_path, map_location='cpu', weights_only=True)
    
    # Gestisce checkpoint con struttura nidificata
    if isinstance(data, dict):
        if 'state_dict' in data:
            return data['state_dict']
        elif 'model' in data: 
            return data['model']
        return data
    
    raise ValueError(f"Formato non supportato per {file_path}")

def get_dict_size_bytes(state_dict: dict) -> int:
    """Calcola la dimensione in bytes di un state_dict."""
    return sum(t.element_size() * t.nelement() for t in state_dict.values())

def merge_and_convert_shards(source_dir:  str, output_file_path:  str, threshold_gb: float = 3.0):
    """
    Scansiona una cartella, trova tutti i file modello validi (.bin, .pt, .safetensors),
    li unisce e salva un UNICO file . safetensors.
    
    OTTIMIZZATO per uso RAM: 
    - Validazione file senza caricarli in memoria
    - Streaming su file temporanei se si supera la soglia
    - Nessuna duplicazione di tensori
    
    Args:
        source_dir:  Cartella contenente i file modello
        output_file_path: Percorso del file . safetensors di output
        threshold_gb:  Soglia in GB oltre la quale salvare su file temporaneo (default: 3.0)
    """
    # Include file senza estensione ('')
    VALID_EXTENSIONS = {'.bin', '.pt', '.pth', '.ckpt', '.safetensors', ''}
    shard_files = []
    
    # === FASE 1: Scansione file (senza caricarli in RAM) ===
    for root, _, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            ext = Path(file).suffix.lower()
            
            # File con estensione valida
            if ext in VALID_EXTENSIONS: 
                shard_files.append(file_path)
                continue
            
            # File SENZA estensione - valida con magic bytes (NO torch.load!)
            if ext == '': 
                if is_torch_file(file_path):
                    shard_files.append(file_path)
                    logger.info(f"[MERGE] Included extensionless torch file: {file}")
                elif is_safetensors_file(file_path):
                    shard_files.append(file_path)
                    logger.info(f"[MERGE] Included extensionless safetensors file: {file}")
                else:
                    logger. debug(f"[MERGE] Skipped non-model file: {file}")
    
    if not shard_files:
        raise FileNotFoundError("Nessun file modello valido (.bin, .safetensors, ecc.) trovato.")
    
    # Sort intelligente (assumo che sharded_file_error. sort_sharded_files esista)
    try:
        shard_files = sharded_file_error.sort_sharded_files(shard_files)
    except NameError:
        # Fallback:  ordine naturale che gestisce numeri (model-00001, model-00002, ecc.)
        import re
        def natural_sort_key(s):
            return [int(t) if t.isdigit() else t. lower() for t in re.split(r'(\d+)', s)]
        shard_files. sort(key=natural_sort_key)
    
    logger.info(f"[MERGE] Trovati {len(shard_files)} file da unire.  Inizio processo...")
    
    # === FASE 2: Merge con gestione memoria ===
    combined_state_dict = {}
    temp_files = []  # Per cleanup finale
    threshold_bytes = threshold_gb * (1024 ** 3)
    
    try:
        for i, shard_path in enumerate(shard_files, 1):
            logger.info(f"[MERGE] Caricamento {i}/{len(shard_files)}: {Path(shard_path).name}")
            
            shard_data = load_model_file(shard_path)
            
            # Unisci i tensori (senza . clone() - evita duplicazione)
            for key, tensor in shard_data. items():
                if key in combined_state_dict: 
                    logger.warning(f"[MERGE] Chiave duplicata '{key}' - verrà sovrascritta")
                combined_state_dict[key] = tensor
            
            # Libera riferimenti al dict originale
            del shard_data
            gc.collect()
            
            # Controlla dimensione accumulata
            current_size = get_dict_size_bytes(combined_state_dict)
            
            if current_size > threshold_bytes and i < len(shard_files):
                logger.info(f"[MERGE] Soglia superata ({current_size / (1024**3):.2f} GB), salvataggio intermedio...")
                
                # Crea file temporaneo
                temp_fd, temp_path = tempfile. mkstemp(suffix='.safetensors')
                os.close(temp_fd)  # Chiudi il file descriptor
                temp_files.append(temp_path)
                
                # Salva su file temporaneo
                save_file(combined_state_dict, temp_path)
                
                # Libera completamente la RAM
                del combined_state_dict
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Ricarica con memory-mapping (safetensors è efficiente qui)
                combined_state_dict = dict(load_safetensors(temp_path))
                gc.collect()
        
        # === FASE 3: Salvataggio finale ===
        logger.info(f"[MERGE] Salvataggio finale in:  {output_file_path}")
        save_file(combined_state_dict, output_file_path)
        logger.info("[MERGE] Completato con successo!")
        
    finally:
        # Cleanup file temporanei
        for temp_path in temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except OSError as e:
                logger.warning(f"[MERGE] Impossibile eliminare file temporaneo {temp_path}: {e}")
        
        # Libera memoria finale
        del combined_state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@models_bp.route('/models', methods=['POST'])
def upload_model():
    """
    Upload and process model files (single or multiple sharded files).
    
    Supports:
    - MODE A (Server-Side): Select local file on VM via JSON {'local_vm_path': '/path/to/file'}
    - MODE B (Upload): Multipart upload of files via browser/client
    
    Returns:
        JSON response with model data and processing status
    """
    t_start_clustering_preprocessing = datetime.now()

    # Track resources for cleanup
    tmp_path = None
    extract_tmp_dir = None
    file_path = None
    readme_path = None
    model_id = None
    
    try:
        # Variabili di stato per il metodo di input
        is_server_side = False
        server_side_path = None
        data_source = None
        files_list = []

        # ========================================================================
        # 1. INPUT CHECK (MODIFICATO PER SUPPORTO VM LOCALE)
        # ========================================================================
        
        # CASO A: Richiesta Server-Side (JSON con path locale)
        if request.is_json and 'local_vm_path' in request.json:
            is_server_side = True
            data_source = request.json
            server_side_path = data_source['local_vm_path']
            
            if not os.path.exists(server_side_path):
                return jsonify({'error': f'File non trovato sulla VM: {server_side_path}'}), 404
                
        # CASO B: Upload Classico (Multipart Form)
        elif 'file' in request.files:
            data_source = request.form
            files_list = request.files.getlist('file')
            
            if not files_list or files_list[0].filename == '':
                return jsonify({'error': 'No file selected'}), 400
        
        else:
            return jsonify({'error': 'No file provided (upload) or invalid JSON path (server-side)'}), 400

        # ========================================================================
        # 2. METADATA EXTRACTION 
        # ========================================================================
        model_id = str(uuid.uuid4())
        
        # Determina il nome default
        if is_server_side:
            default_name = os.path.basename(server_side_path)
        else:
            default_name = files_list[0].filename

        name = data_source.get('name', default_name)
        description = data_source.get('description', '')
        license_value = data_source.get('license', '')
        task_value = data_source.get('task', '')
        dataset_url = data_source.get('dataset_url', '')
        
        # Gestione robusta del booleano (che arriva come stringa nel form o bool nel json)
        is_foundation_val = data_source.get('is_foundation_model', False)
        if isinstance(is_foundation_val, str):
            is_foundation_model = is_foundation_val.lower() == 'true'
        else:
            is_foundation_model = bool(is_foundation_val)
        
        if dataset_url and not validate_url(dataset_url):
            return jsonify({'error': 'Invalid dataset URL format'}), 400

        # ========================================================================
        # 3. README FILE HANDLING (Solo per Upload Classico per ora)
        # ========================================================================
        readme_uri = None
        # Nella modalità server-side saltiamo il readme upload per semplicità
        if not is_server_side:
            readme_file = request.files.get('readme_file')
            
            if readme_file and readme_file.filename:
                if not allowed_readme_file(readme_file.filename):
                    return jsonify({'error': 'README must be . md or .txt file'}), 400
                
                readme_file.seek(0, os.SEEK_END)
                readme_size = readme_file.tell()
                readme_file.seek(0)
                
                if readme_size > MAX_README_SIZE:
                    return jsonify({'error': f'README file too large (max {MAX_README_SIZE // (1024*1024)}MB)'}), 400
                
                os.makedirs(README_FOLDER, exist_ok=True)
                readme_filename = secure_filename(f"{model_id}_readme.md")
                readme_path = os.path.join(README_FOLDER, readme_filename)
                readme_file.save(readme_path)
                readme_uri = f"readmes/{readme_filename}"

        # ========================================================================
        # 4. TEMPORARY PATHS SETUP
        # ========================================================================
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.safetensors')
        os.close(tmp_fd)
        extract_tmp_dir = tempfile.mkdtemp()

        loaded_object = None
        metadata = {}
        
        # ========================================================================
        # 5. FILE ACQUISITION LOGIC
        # ========================================================================

        # --- CASO A: SERVER SIDE (VM LOCALE) ---
        if is_server_side:
            logger.info(f"[UPLOAD] Server-Side Mode: Processing {server_side_path}")
            filename = os.path.basename(server_side_path)
            ext = os.path.splitext(filename)[1].lower()

            # Copia sicura per non modificare il file originale durante l'elaborazione
            # Se è .safetensors, lo mettiamo direttamente in tmp_path per la normalizzazione
            if ext == '.safetensors':
                shutil.copy2(server_side_path, tmp_path)
                # Estraiamo i metadati
                with safe_open(tmp_path, framework="pt", device="cpu") as f:
                    metadata = f.metadata() or {}
            
            # Se è un binario PyTorch, lo copiamo nella dir di estrazione e lo carichiamo
            elif ext in {'.bin', '.pt', '.pth', '.ckpt'}:
                dest_bin = os.path.join(extract_tmp_dir, filename)
                shutil.copy2(server_side_path, dest_bin)
                # Chiamiamo l'helper passando None come file object, dato che il file è già su disco
                loaded_object = _process_binary_file(
                    None, filename, extract_tmp_dir, tmp_path
                )
            else:
                return jsonify({'error': f'Formato file VM non supportato: {ext}'}), 400

        # --- CASO B: MULTIPART UPLOAD MULTIPLO (SHARDED) ---
        elif len(files_list) > 1:
            logger.info(f"[UPLOAD] Detected multiple file upload: {len(files_list)} files")
            
            if sharded_file_error.is_likely_sharded_upload(files_list):
                try:
                    validation_result = sharded_file_error.validate_sharded_safetensors(files_list)
                    logger.info(
                        f"✅ [UPLOAD] Sharded safetensors validated: {validation_result['base_name']} "
                        f"({validation_result['total_shards']} shards)"
                    )
                    files_list = validation_result['sorted_files']
                except ShardedFileError as e: 
                    logger.error(f"❌ [UPLOAD] Sharded validation failed: {e}")
                    return jsonify({'error': f'Invalid sharded files: {str(e)}'}), 400
            
            for f in files_list:
                fname = secure_filename(f.filename)
                if '/' in fname or '\\' in fname:
                    fname = os.path.basename(fname)
                if fname: 
                    f.save(os.path.join(extract_tmp_dir, fname))
            
            try:
                merge_and_convert_shards(extract_tmp_dir, tmp_path)
                logger.info(f"✅ [UPLOAD] Merge completed successfully")
            except ValueError as e:
                logger.error(f"❌ [UPLOAD] Merge failed (duplicate keys): {e}")
                return jsonify({'error': str(e)}), 400
            except Exception as e:
                logger. error(f"❌ [UPLOAD] Merge failed: {e}")
                return jsonify({'error': f'Failed to merge files: {str(e)}'}), 500

        # --- CASO C: MULTIPART UPLOAD SINGOLO ---
        else:
            file = files_list[0]
            filename = secure_filename(file.filename)
            ext = os.path.splitext(filename)[1].lower()
            
            logger.info(f"[UPLOAD] Single file upload: {filename}")

            # ZIP FILE
            if ext == '.zip': 
                loaded_object, metadata = _process_zip_file(
                    file, extract_tmp_dir, tmp_path
                )

            # BINARY FILE
            elif ext in {'.bin', '.pt', '.pth', '.ckpt'}:
                loaded_object = _process_binary_file(
                    file, filename, extract_tmp_dir, tmp_path
                )

            # SAFETENSORS FILE
            elif ext == '.safetensors': 
                file.save(tmp_path)
                with safe_open(tmp_path, framework="pt", device="cpu") as f:
                    metadata = f.metadata() or {}

            # UNSUPPORTED EXTENSION
            else:
                return jsonify({
                    'error': f'Invalid file type: {ext}.  Supported:  .safetensors, .pt, .bin, .pth, .zip'
                }), 400

        # ========================================================================
        # 6. BINARY -> SAFETENSORS CONVERSION 
        # ========================================================================
        if loaded_object is not None:
            _convert_to_safetensors(loaded_object, tmp_path)
            del loaded_object
            gc.collect()

        # ========================================================================
        # 7. UPLOAD AND NORMALIZATION
        # ========================================================================
        logger.info("[UPLOAD] Loading safetensors for normalization...")
        tensors_dict = load_safetensors(tmp_path)

        os.unlink(tmp_path)
        tmp_path = None  
        logger.info("[UPLOAD] Starting layer name normalization...")

        tensors_dict, mapping_dict = normalize_safetensors_layers(tensors_dict)

        logger.info(f"[UPLOAD] Normalization completed: {len(tensors_dict)} layers")

        # ========================================================================
        # 8. SAVING FINGERPRINT
        # ========================================================================
        # Usa il nome originale salvato prima (dal path VM o dal file upload)
        original_filename = os.path.basename(server_side_path) if is_server_side else files_list[0].filename
        
        num_layers = save_layer_mapping_json(
            mapping_dict, 
            model_id, 
            original_filename
        )
        logger.info(f"[UPLOAD] Fingerprint saved: {num_layers} structural layers")

        t_end_clustering_preprocessing = datetime.now()
        t_start_mother_preprocessing = datetime.now()

        # ========================================================================
        # 9. CALCULATING KURTOSIS 
        # ========================================================================
        logger.info("[UPLOAD] Calculating kurtosis...")
        kurtosis = calc_ku(tensors_dict)
        logger.info(f"[UPLOAD] Kurtosis: {kurtosis}")

        t_end_mother_preprocessing = datetime.now()
        t_start_clustering = datetime.now()

        # ========================================================================
        # 10. SAVING FINAL NORMALIZED SAFETENSORS FILE
        # ========================================================================
        os.makedirs(MODEL_FOLDER, exist_ok=True)
        
        final_clean_name = os.path.splitext(secure_filename(name))[0] + ".safetensors"
        final_filename = secure_filename(f"{model_id}_{final_clean_name}")
        file_path = os.path.join(MODEL_FOLDER, final_filename)

        save_file(tensors_dict, file_path, metadata=metadata)
        logger.info(f"[UPLOAD] Final file saved: {file_path}")
        
        del tensors_dict
        gc.collect()

        # ========================================================================
        # 11. POST-PROCESSING:  Checksum, Signature, Neo4j
        # ========================================================================
        checksum = calculate_file_checksum(file_path)
        
        existing = neo4j_service.get_model_by_checksum(checksum)
        if existing:
            logger.warning(f"⚠️ [UPLOAD] Duplicate model:  {existing.get('id')}")
           
            return jsonify({
                'error': 'Model already exists',
                'existing_id': existing.get('id')
            }), 409
        
        signature = extract_weight_signature(file_path, num_layers)
        logger.info(f"[UPLOAD] Signature:  {signature['total_parameters']} params")
        
        task_list = [t.strip() for t in task_value.split(',') if t.strip()] if task_value else []

        model_data = {
            'id': model_id,
            'name': name,
            'description': description,
            'file_path': file_path,
            'checksum': checksum,
            'total_parameters': signature['total_parameters'],
            'layer_count': num_layers,
            'structural_hash': signature['structural_hash'],
            'status': 'processing',
            'weights_uri': 'weights/' + final_filename,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'distance_from_parent': 0.0,
            'kurtosis': kurtosis,
            'license': license_value or None,
            'task': task_list,
            'dataset_url': dataset_url or None,
            'dataset_url_verified': None,
            'readme_uri':  readme_uri,
            'is_foundation_model': is_foundation_model
        }

        if not neo4j_service.create_model(model_data):
            raise Exception("Failed to save model to Neo4j")
        
        logger.info(f"[UPLOAD] Model saved to Neo4j: {model_id}")

        delta_clustering_preprocessing = t_end_clustering_preprocessing - t_start_clustering_preprocessing
        delta_mother_preprocessing = t_end_mother_preprocessing - t_start_mother_preprocessing
        
        result = mgmt_system.process_new_model(model_data,t_start_clustering ,delta_clustering_preprocessing, delta_mother_preprocessing) 
        if result.get('status') != 'success':
            raise Exception(f"ModelManagementSystem failed: {result.get('error', 'Unknown error')}")
        
        family_id = result.get('family_id')
        if family_id:
            neo4j_service.create_belongs_to_relationship(model_id, family_id)
        
        final_model_data = neo4j_service.get_model_by_id(model_id).to_dict()
        
        file_path = None
        readme_path = None
        logger.info(f"✅[UPLOAD] Model upload completed:  {model_id}")

        return jsonify({
            'model_id': model_id,
            'status':  'ok',
            'message': 'Processed successfully',
            'model':  final_model_data
        }), 201

    except ShardedFileError as e: 
        logger.error(f"❌ [UPLOAD] Sharded file error: {e}")
        return jsonify({'error': f'Sharded file validation failed: {str(e)}'}), 400
        
    except Exception as e: 
        logHandler.error_handler(e, "upload_model", f"model_id={model_id}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
        
    finally:
        # ========================================================================
        # FINAL CLEANUP
        # ========================================================================
        _cleanup_resources(tmp_path, extract_tmp_dir, file_path, readme_path)
# =============================================================================
# HELPER FUNCTIONS FOR UPLOAD MODEL
# =============================================================================

def _process_zip_file(file, extract_tmp_dir:  str, tmp_path: str) -> tuple: 
    """Processa un file ZIP e restituisce (loaded_object, metadata)."""
    logger.info(f"[UPLOAD] Processing ZIP file")
    zip_path = os.path.join(extract_tmp_dir, "upload.zip")
    file.save(zip_path)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_tmp_dir)
    except zipfile.BadZipFile:
        raise ValueError('Invalid ZIP file format')
    
    found_models = sharded_file_error.scan_for_model_files(extract_tmp_dir, include_no_extension=True)
    logger.info(f"[UPLOAD] Found {len(found_models)} model files in ZIP")
    
    if not found_models:
        raise FileNotFoundError("ZIP archive contains no valid model files")
    
    metadata = {}
    loaded_object = None
    
    if len(found_models) > 1:
        merge_and_convert_shards(extract_tmp_dir, tmp_path)
    elif found_models[0].lower().endswith('.safetensors'):
        shutil.move(found_models[0], tmp_path)
        with safe_open(tmp_path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
    else:
        loaded_object = torch.load(found_models[0], map_location="cpu", weights_only=True)
    
    return loaded_object, metadata

def _process_binary_file(file, filename: str, extract_tmp_dir: str, tmp_path: str):
    """Processa un file binario (. bin, .pt, etc.)."""
    logger.info(f"[UPLOAD] Processing binary file: {filename}")
    bin_path = os.path.join(extract_tmp_dir, filename)
    
    # MODIFICA QUI: Salva solo se il file object esiste (Upload classico)
    # Se file è None, assumiamo che bin_path esista già (modalità VM locale)
    if file is not None:
        file.save(bin_path)
    
    success, loaded_object = sharded_file_error.smart_load_bin(bin_path, extract_tmp_dir)
    
    if not success:
        raise ValueError(f'Unable to load {filename}:  not a valid PyTorch binary')
    
    if loaded_object is None:
        # File was extracted as archive
        found_models = sharded_file_error.scan_for_model_files(extract_tmp_dir, include_no_extension=True)
        
        if not found_models: 
            raise FileNotFoundError('No valid model files found in binary archive')
        
        if len(found_models) > 1:
            merge_and_convert_shards(extract_tmp_dir, tmp_path)
            return None
        else:
            loaded_object = torch.load(found_models[0], map_location="cpu", weights_only=True)
    
    return loaded_object

def _convert_to_safetensors(loaded_object, tmp_path: str):
    """Converte un oggetto PyTorch in safetensors."""
    logger.info("[UPLOAD] Converting PyTorch object to safetensors...")
    
    if isinstance(loaded_object, torch.nn.Module):
        state_dict = loaded_object.state_dict()
    elif isinstance(loaded_object, dict):
        if "state_dict" in loaded_object:
            state_dict = loaded_object["state_dict"]
        elif "model" in loaded_object:
            state_dict = loaded_object["model"]
        else:
            state_dict = loaded_object
    else:
        raise ValueError(f'Unexpected model format: {type(loaded_object)}')
    
    # ✅ FIX: Usiamo .clone() per rompere la condivisione della memoria (Shared Tensors)
    # Safetensors non accetta che due chiavi puntino allo stesso indirizzo di memoria.
    # .contiguous() assicura che il layout di memoria sia compatibile.
    clean_state_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            # detach() -> toglie dal grafo
            # clone() -> crea NUOVA memoria (risolve l'errore "shared memory")
            # contiguous() -> riordina i byte in memoria per safetensors
            clean_state_dict[k] = v.detach().clone().contiguous()
    
    save_file(clean_state_dict, tmp_path)
    logger.info(f"✅ [UPLOAD] Conversion completed: {len(clean_state_dict)} tensors")

def _cleanup_resources(tmp_path, extract_tmp_dir, file_path, readme_path):
    """Pulisce tutte le risorse temporanee."""
    # File temporaneo safetensors
    if tmp_path and os.path.exists(tmp_path):
        try:
            os.unlink(tmp_path)
        except OSError as e:
            logger.warning(f"[CLEANUP] Failed to remove tmp_path: {e}")
    
    # Directory temporanea
    if extract_tmp_dir and os.path.exists(extract_tmp_dir):
        try:
            shutil.rmtree(extract_tmp_dir)
        except OSError as e:
            logger.warning(f"[CLEANUP] Failed to remove extract_tmp_dir: {e}")
    
    # File modello (solo se upload fallito)
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"[CLEANUP] Removed failed upload file: {file_path}")
        except OSError as e:
            logger.warning(f"[CLEANUP] Failed to remove file_path: {e}")
    
    # README (solo se upload fallito)
    if readme_path and os.path.exists(readme_path):
        try:
            os.remove(readme_path)
        except OSError as e:
            logger.warning(f"[CLEANUP] Failed to remove readme:  {e}")

@models_bp.route('/models', methods=['GET'])
def list_models():
    try:
        """List all models with optional search"""
        search = request.args.get('search', '').strip()
        
        models_data = neo4j_service.get_all_models(search=search or None)
        models_data = sorted(models_data, key=lambda m: m.name.lower())

        models = []
        for m in models_data:
            models.append(m.to_dict())
        
        return jsonify({
            'models': models,
            'total': len(models)
        })
    except Exception as e:
        logHandler.error_handler(e, "list_models")
        return jsonify({'error': 'Failed to retrieve models', 'details': str(e)}), 500

@models_bp.route('/models/<model_id>', methods=['GET'])
def get_model(model_id):
    """Get model details by ID"""
    try:
        model = neo4j_service.get_model_by_id(model_id)
        
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        # Convert Model object to dictionary
        model_data = model.to_dict()
        
        # Get lineage information
        lineage = neo4j_service.get_model_lineage(model_id)
        model_data['lineage'] = lineage
        
        return jsonify(model_data)
        
    except Exception as e:
        logHandler.error_handler(e, "get_model_by_id")
        return jsonify({'error': 'Failed to retrieve model', 'details': str(e)}), 500

@models_bp.route('/families', methods=['GET'])
def list_families():
    """List all families"""
    families_data = neo4j_service.get_all_families()
    
    return jsonify({
        'families': families_data,
        'total': len(families_data)
    })

@models_bp.route('/models/<model_id>/readme', methods=['GET'])
def get_model_readme(model_id):
    """Get README content for a model"""
    try:
        model = neo4j_service.get_model_by_id(model_id)
        
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        readme_uri = model.readme_uri
        if not readme_uri:
            return jsonify({'error': 'No README available for this model'}), 404
        
        # Read README file
        readme_path = readme_uri  # readme_uri is already relative path like "readmes/xxx_readme.md"
        if not os.path.exists(readme_path):
            return jsonify({'error': 'README file not found'}), 404
        
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return jsonify({
            'model_id': model_id,
            'content': content,
            'readme_uri': readme_uri
        })
        
    except Exception as e:
        logHandler.error_handler(e, "get_model_readme")
        return jsonify({'error': 'Failed to retrieve README', 'details': str(e)}), 500

@models_bp.route('/families/<family_id>/models', methods=['GET'])
def get_family_models(family_id):
    """Get all models in a family"""

    # Verify family exists
    family_data = neo4j_service.get_family_by_id(family_id)
    if not family_data:
        return jsonify({'error': 'Family not found'}), 404
    
    # Get models in family
    models_data = neo4j_service.get_family_models(family_id, status='ok')
    
    return jsonify({
        'family': family_data,
        'models': models_data
    })

@models_bp.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    stats = neo4j_service.get_stats()
    return jsonify(stats)

@models_bp.route('/families/<family_id>/genealogy', methods=['GET'])
def get_family_genealogy(family_id):
    """Get complete genealogy information for a family"""
    try:
        from src.clustering.model_management import ModelManagementSystem
        
        mgmt_system = ModelManagementSystem()
        result = mgmt_system.get_family_genealogy(family_id)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 404
        
        return jsonify(result)
        
    except Exception as e:
        current_app.logger.error(f"Failed to get genealogy for family {family_id}: {e}")
        return jsonify({'error': f'Failed to get genealogy: {str(e)}'}), 500

@models_bp.route('/models/<model_id>/lineage', methods=['GET'])
def get_model_lineage(model_id):
    """Get complete lineage information for a specific model"""
    try:
        from src.clustering.model_management import ModelManagementSystem
        
        mgmt_system = ModelManagementSystem()
        result = mgmt_system.get_model_lineage(model_id)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 404
        
        return jsonify(result)
        
    except Exception as e:
        current_app.logger.error(f"Failed to get lineage for model {model_id}: {e}")
        return jsonify({'error': f'Failed to get lineage: {str(e)}'}), 500

@models_bp.route('/clustering/statistics', methods=['GET'])
def get_clustering_statistics():
    """Get comprehensive clustering system statistics"""
    try:
        from src.clustering.model_management import ModelManagementSystem
        
        mgmt_system = ModelManagementSystem()
        result = mgmt_system.get_system_statistics()
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
        
        return jsonify(result)
        
    except Exception as e:
        current_app.logger.error(f"Failed to get clustering statistics: {e}")
        return jsonify({'error': f'Failed to get statistics: {str(e)}'}), 500
    
@models_bp.route('/models/<model_id>/download', methods=['GET'])
def download_model_weights(model_id):
    """Download model weights with original layer names restored"""
    try:
        # 1. Get model from Neo4j
        model = neo4j_service.get_model_by_id(model_id)
        if not model: 
            return jsonify({'error': 'Model not found'}), 404
        
        file_path = model.file_path
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'Model file not found'}), 404
        
        # 2. Load fingerprint JSON to get mapping and original filename
        fingerprint_path = os.path.join('weights', 'fingerprints', f'{model_id}_mapping.json')
        if not os.path.exists(fingerprint_path):
            return jsonify({'error': 'Fingerprint file not found'}), 404
        
        with open(fingerprint_path, 'r') as f:
            fingerprint = json.load(f)
        
        mapping = fingerprint. get('mapping', {})
        original_filename = fingerprint.get('original_filename', f'{model_id}.safetensors')
        
        # 3. Load the stored safetensors file (normalized layer names)
        tensors_dict = load_safetensors(file_path)
        
        # 4. Load original metadata
        with safe_open(file_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
        
        # 5. Create new tensors dict with original layer names (key -> value mapping)
        restored_tensors = {}
        for normalized_name, tensor in tensors_dict.items():
            # mapping:  normalized_name (key) -> original_name (value)
            original_name = mapping.get(normalized_name, normalized_name)
            restored_tensors[original_name] = tensor
        
        # 6. Save to temporary file in memory
        with tempfile.NamedTemporaryFile(delete=False, suffix='.safetensors') as tmp_file:
            tmp_path = tmp_file.name
        
        save_file(restored_tensors, tmp_path, metadata=metadata)
        
        # 7. Read the file into memory and delete temp file
        with open(tmp_path, 'rb') as f:
            file_data = f.read()
        os.unlink(tmp_path)
        
        # 8. Send file as download
        return send_file(
            io.BytesIO(file_data),
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=original_filename
        )
        
    except Exception as e:
        logHandler.error_handler(e, "download_model_weights")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500