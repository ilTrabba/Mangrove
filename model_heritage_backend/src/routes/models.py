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
from datetime import datetime, time, timezone
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
    Estrae la signature strutturale di un modello safetensors basandosi su: 
    
      - hidden_size:  derivato dalle shapes (dimensione più comune in matrici quadrate ≥128)
      - num_layers: fornito come parametro
      - structural_hash: MD5 troncato basato su (num_layers + hidden_size)
      - total_parameters: somma di tutti i parametri
    
    OTTIMIZZATO:  
      - Usa metadata delle shapes senza caricare i tensori in RAM
      - Memoria O(1) rispetto alla dimensione del modello
    
    Args:
        file_path:  Percorso al file . safetensors
        num_layers: Numero di layer del modello
        
    Returns:
        Dict con num_layers, hidden_size, structural_hash, total_parameters
        
    Raises:
        RuntimeError: Se impossibile dedurre hidden_size
    """
    total_parameters = 0
    square_dims = Counter()  # Conta direttamente le dimensioni delle matrici quadrate
    
    with safe_open(file_path, framework="pt") as f:
        for key in f.keys():
            # ✅ Legge SOLO i metadati, NON carica il tensore in RAM
            shape = f.get_slice(key).get_shape()
            
            # Calcola parametri usando math.prod (più leggibile)
            num_params = 1
            for dim in shape: 
                num_params *= dim
            total_parameters += num_params
            
            # Cerca matrici quadrate significative (≥128)
            if len(shape) == 2 and shape[0] == shape[1] and shape[0] >= 128:
                square_dims[shape[0]] += 1
    
    # Deduzione hidden_size
    if not square_dims:
        raise RuntimeError(
            "Impossibile dedurre hidden_size dal file "
            "(nessuna matrice quadrata ≥128 trovata)."
        )
    
    hidden_size = square_dims.most_common(1)[0][0]
    
    # Hash strutturale (parametri:  num_layers + hidden_size)
    base_string = f"{num_layers}_{hidden_size}"
    structural_hash = hashlib.md5(base_string.encode()).hexdigest()[:16]
    
    return {
        "num_layers":  num_layers,
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
    - Single .safetensors file
    - Multiple sharded .safetensors files
    - Single .bin/.pt/.pth/.ckpt file
    - .zip archives containing model files
    - Files without extensions (validated as PyTorch binaries)
    
    Returns:
        JSON response with model data and processing status
    """
    # Track resources for cleanup
    t_start_clustering_preprocessing = datetime.now()
    tmp_path = None
    extract_tmp_dir = None
    file_path = None
    readme_path = None
    model_id = None
    
    try:
        # ========================================================================
        # 1. INPUT CHECK
        # ========================================================================
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        files_list = request.files.getlist('file')
        
        if not files_list or files_list[0].filename == '':
            return jsonify({'error': 'No file selected'}), 400

        first_file = files_list[0]
        
        # ========================================================================
        # 2. METADATA EXTRACTION FROM THE FORM
        # ========================================================================
        model_id = str(uuid.uuid4())
        name = request.form.get('name', first_file.filename)
        description = request.form.get('description', '')
        license_value = request.form.get('license', '')
        task_value = request.form.get('task', '')
        dataset_url = request. form.get('dataset_url', '')
        is_foundation_model = request.form.get('is_foundation_model', 'false').lower() == 'true'
        
        if dataset_url and not validate_url(dataset_url):
            return jsonify({'error': 'Invalid dataset URL format'}), 400

        # ========================================================================
        # 3. README FILE HANDLING 
        # ========================================================================
        readme_uri = None
        readme_file = request.files.get('readme_file')
        
        if readme_file and readme_file.filename:
            if not allowed_readme_file(readme_file.filename):
                return jsonify({'error': 'README must be . md or .txt file'}), 400
            
            readme_file. seek(0, os.SEEK_END)
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
        # 5. CASE A - UPLOAD OF MULTIPLE FILES
        # ========================================================================
        if len(files_list) > 1:
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

        # ========================================================================
        # 6. CASE B - SINGLE FILE UPLOAD
        # ========================================================================
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
        # 7. BINARY -> SAFETENSORS CONVERSION 
        # ========================================================================
        if loaded_object is not None:
            _convert_to_safetensors(loaded_object, tmp_path)
            del loaded_object
            gc.collect()

        # ========================================================================
        # 8. UPLOAD AND NORMALIZATION
        # ========================================================================
        logger.info("[UPLOAD] Loading safetensors for normalization...")
        tensors_dict = load_safetensors(tmp_path)

        os.unlink(tmp_path)
        tmp_path = None  
        logger.info("[UPLOAD] Starting layer name normalization...")

        tensors_dict, mapping_dict = normalize_safetensors_layers(tensors_dict)

        logger.info(f"[UPLOAD] Normalization completed: {len(tensors_dict)} layers")

        # ========================================================================
        # 9. SAVING FINGERPRINT
        # ========================================================================
        num_layers = save_layer_mapping_json(
            mapping_dict, 
            model_id, 
            first_file.filename
        )
        logger.info(f"[UPLOAD] Fingerprint saved: {num_layers} structural layers")
        
        t_end_clustering_preprocessing = datetime.now()
        t_start_mother_preprocessing = datetime.now()
        # ========================================================================
        # 10. CALCULATING KURTOSIS 
        # ========================================================================
        logger.info("[UPLOAD] Calculating kurtosis...")
        kurtosis = calc_ku(tensors_dict)
        logger.info(f"[UPLOAD] Kurtosis: {kurtosis}")

        t_end_mother_preprocessing = datetime.now()
        t_start_clustering = datetime.now()
        # ========================================================================
        # 11. SAVING FINAL NORMALIZED SAFETENSORS FILE
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
        # 12. POST-PROCESSING:  Checksum, Signature, Neo4j
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
        result = mgmt_system.process_new_model(model_data, t_start_clustering, delta_clustering_preprocessing, delta_mother_preprocessing)

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

import torch
import numpy as np

def _process_binary_file(file, filename: str, extract_tmp_dir: str, tmp_path: str):
    logger.info(f"[UPLOAD] Processing binary file: {filename}")
    bin_path = os.path.join(extract_tmp_dir, filename)
    file.save(bin_path)
    
    loaded_object = None

    # TENTATIVO 1: Caricamento standard (weights_only=True)
    try:
        loaded_object = torch.load(bin_path, map_location="cpu", weights_only=True)
        return loaded_object
    except Exception:
        pass # Se fallisce, passiamo al piano B (Whitelist)

    # PIANO B: Whitelist Aggressiva
    logger.warning(f"⚠️ [UPLOAD] Standard load failed for {filename}. Retrying with Extended Whitelist.")
    
    safe_list = []
    
    # 1. Aggiungiamo NumPy Scalar (Cruciale per il tuo errore)
    # TRUCCO: Recuperiamo il riferimento esatto usato internamente
    try:
        safe_list.append(np.core.multiarray.scalar)
    except AttributeError:
        # Per NumPy 2.0+ o altre versioni dove il percorso è cambiato
        try:
            safe_list.append(np.scalar) 
        except:
            pass

    # Aggiungiamo anche i tipi base di numpy che spesso causano problemi
    safe_list.append(np.dtype)
    
    # 2. Aggiungiamo Lightning (Cruciale per .ckpt)
    try:
        from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
        safe_list.append(ModelCheckpoint)
    except ImportError:
        pass

    # 3. Altri colpevoli comuni nei .ckpt vecchi
    try:
        import collections
        safe_list.append(collections.OrderedDict)
    except:
        pass

    try:
        with torch.serialization.safe_globals(safe_list):
            loaded_object = torch.load(bin_path, map_location="cpu", weights_only=True)
        logger.info("✅ [UPLOAD] Loaded safely with Extended Whitelist.")
        return loaded_object

    except Exception as e:
        # PIANO C: Ultima spiaggia (SOLO se ti fidi dei file e vuoi sbloccare la situazione)
        # Se sei in un ambiente controllato, puoi decommentare queste righe per forzare il caricamento.
        # Altrimenti, rilancia l'errore.
        
        logger.error(f"❌ [UPLOAD] Whitelist failed: {e}")
        
        # SCOMMENTA QUI SOTTO SOLO SE VUOI FORZARE IL CARICAMENTO (RISCHIO SICUREZZA)
        logger.warning("⚠️⚠️ [UPLOAD] FALLBACK: Loading with weights_only=False (UNSAFE MODE)")
        loaded_object = torch.load(bin_path, map_location="cpu", weights_only=False)
        return loaded_object
        
        raise ValueError(f"Unable to load safe checkpoint: {str(e)}")




def _convert_to_safetensors(loaded_object, tmp_path: str):
    """
    Converte un oggetto PyTorch in safetensors.
    Gestisce la priorità EMA e il disaccoppiamento della memoria (clone).
    """
    logger.info("[UPLOAD] Converting PyTorch object to safetensors...")
    
    state_dict = None

    # 1. Logica di estrazione (Priorità EMA senza creare oggetti nuovi)
    if isinstance(loaded_object, dict):
        if "state_dict_ema" in loaded_object:
            logger.info("ℹ️ [CONVERT] Using EMA weights.")
            state_dict = loaded_object["state_dict_ema"]
        elif "state_dict" in loaded_object:
            state_dict = loaded_object["state_dict"]
        elif "model" in loaded_object:
            state_dict = loaded_object["model"]
        else:
            state_dict = loaded_object
            
    elif isinstance(loaded_object, torch.nn.Module):
        state_dict = loaded_object.state_dict()
    else:
        raise ValueError(f'Unexpected model format: {type(loaded_object)}')
    
    # 2. Pulizia: usiamo dict comprehension per evitare loop complessi
    # .detach().clone() è essenziale per evitare errori di shared memory
    clean_state_dict = {
        k: v.detach().clone().contiguous() 
        for k, v in state_dict.items() 
        if isinstance(v, torch.Tensor)
    }
    
    if not clean_state_dict:
        raise ValueError("No valid tensors found to convert.")

    # 3. Salvataggio diretto
    save_file(clean_state_dict, tmp_path, metadata={"format": "pt"})
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