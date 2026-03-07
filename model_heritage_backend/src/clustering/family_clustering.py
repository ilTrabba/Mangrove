"""
FamilyClusteringSystem

This module manages model families and automatic clustering based on weight similarities.
It handles family assignment, centroid calculation, and family management operations.
"""

import logging
import torch
import uuid
import re
import safetensors.torch
import os
import numpy as np

from src.log_handler import logHandler
from src.mother_algorithm.mother_utils import load_model_weights
from src.clustering.distance_calculator import DistanceMetric
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from enum import Enum
from datetime import datetime
from safetensors import safe_open
from ..db_entities.entity import Model
from src.services.neo4j_service import neo4j_service
from .distance_calculator import ModelDistanceCalculator
from src.utils.architecture_filtering import FilteringPatterns
from src.utils.extract_mad_values import extract_mad_values

logger = logging.getLogger(__name__)

#FIXME: capire bene quali soglie settare empiricamente e non
MIN_CONFIDENCE = 0.2

# Costante per soglia chunking automatico
CENTROID_CHUNK_THRESHOLD = 10_000_000  # 40 MB in float32

class ClusteringMethod(Enum):
    """Available clustering methods"""

    DBSCAN = "dbscan"
    KMEANS = "kmeans" 
    THRESHOLD = "threshold"
    AUTO = "auto"


class MetricUtils:
    """Utility per calcoli geometrici efficienti sui pesi per la FamilyGuardian."""
    
    @staticmethod
    def calculate_directional_cosine(
        root_weights: Dict[str, Any],
        centroid_weights: Dict[str, Any],
        new_model_weights: Dict[str, Any]
    ) -> float:
        """
        Calcola il coseno di similarità tra il vettore storia (Centroid - Root) 
        e il vettore nuovo (NewModel - Root).
        
        Opera layer-wise per efficienza di memoria, evitando di concatenare 
        l'intero modello in un unico vettore gigante.
        """
        try:
            dot_product = 0.0
            norm_history_sq = 0.0
            norm_new_sq = 0.0
            
            # Identifica i layer comuni a tutti e tre i set di pesi
            common_layers = set(root_weights.keys()) & set(centroid_weights.keys()) & set(new_model_weights.keys())
            
            if not common_layers:
                return 1.0  # Fallback neutro se non ci sono layer comuni
                
            for layer in common_layers:
                t_root = root_weights[layer]
                t_cent = centroid_weights[layer]
                t_new = new_model_weights[layer]
                
                # Assicuriamoci che siano tensori torch
                if not (isinstance(t_root, torch.Tensor) and 
                        isinstance(t_cent, torch.Tensor) and 
                        isinstance(t_new, torch.Tensor)):
                    continue

                # Normalizzazione tipo e device (CPU float32)
                t_root = t_root.detach().cpu()
                t_cent = t_cent.detach().cpu()
                t_new = t_new.detach().cpu()
                
                if t_root.dtype == torch.bfloat16: t_root = t_root.float()
                if t_cent.dtype == torch.bfloat16: t_cent = t_cent.float()
                if t_new.dtype == torch.bfloat16: t_new = t_new.float()
                
                # Verifica compatibilità shape
                if t_root.shape != t_cent.shape or t_root.shape != t_new.shape:
                    continue

                # Vettori differenza (History e New Candidate)
                diff_history = t_cent - t_root
                diff_new = t_new - t_root
                
                # Calcoli parziali (dot product e norm squares)
                # Flattening implicito nelle operazioni sum
                dot_product += torch.sum(diff_history * diff_new).item()
                norm_history_sq += torch.sum(diff_history ** 2).item()
                norm_new_sq += torch.sum(diff_new ** 2).item()
                
            norm_history = np.sqrt(norm_history_sq)
            norm_new = np.sqrt(norm_new_sq)
            
            # Se la storia è ferma (norm_history=0) o il nuovo è fermo, cos=1 (neutro)
            if norm_history == 0 or norm_new == 0:
                return 1.0 
                
            # Calcolo coseno finale
            cosine_sim = dot_product / (norm_history * norm_new)
            return float(np.clip(cosine_sim, -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating directional cosine: {e}")
            return 1.0 # Fallback neutro in caso di errore

class FamilyGuardian:
    """
    Implementa la soglia adattiva 'Bounded Hybrid'.
    Combina statistica robusta (MAD), geometria (Coseno) e limiti evolutivi (Safe Harbor).
    """
    def __init__(self, distances_history: List[float], min_threshold: float = 1):
        # Convertiamo la history in numpy array per calcoli veloci
        self.distances_history = np.array(distances_history) if distances_history else np.array([])
        # n_members stimato (storia + 1 per il centroide/radice)
        self.n_members = len(self.distances_history) + 1 
        self.min_threshold = min_threshold
        
    def _get_adaptive_k(self) -> float:
        """
        Calcola il fattore k (severità) che decade con la numerosità.
        Rule: Più membri ho (più certezza statistica), più divento severo (k basso).
        """
        k_max = 5.7  # Molto permissivo all'inizio
        k_min = 3.2  # Standard gaussiano a regime
        decay_rate = 0.1
        
        # Formula di decadimento esponenziale inversa
        # Usa max(0, n-2) per ritardare il decadimento ai primissimi membri
        current_k = k_min + (k_max - k_min) * np.exp(-decay_rate * max(0, self.n_members - 2))
        return current_k

    def check_admissibility(self, 
                          dist_to_centroid: float, 
                          dist_to_root: Optional[float] = None, 
                          cosine_sim: float = 1.0,
                          max_family_radius: Optional[float] = None,
                          median: Optional[float] = None,
                          mad_val: Optional[float] = None,
                          max_dist_from_centroid: Optional[float] = None,
                          num_of_nodes: int = 0) -> Tuple[bool, float, float, str]:
        """
        Valuta se accettare il modello nella famiglia.
        
        Args:
            dist_to_centroid: Distanza L2 tra il centroide corrente e il modello candidato
            dist_to_root: Distanza tra la radice e il modello candidato(se disponibile)
            cosine_sim: Similarità coseno direzionale (se disponibile)
            
        Returns:
            Tuple(is_accepted, confidence_score, final_threshold, reason_string)
        """
        # 1. Safe Harbor: Se il modello è molto vicino alla radice, entra sempre.
        #    Questo gestisce la forma a "Stella" (figli ortogonali diretti).
        if dist_to_root is not None:
            # Safe harbor definito come il max tra un valore hard coded e la distanza massima registrata tra i figli diretti della radice
            safe_harbor_limit = max(self.min_threshold * 2.5, 2.75)
            if dist_to_root < safe_harbor_limit:
                return True, 1.0, safe_harbor_limit, "Accepted (Safe Harbor: Near Root)"

        # 2. Penalità Direzionale (Gestione forma a "Verme")
        #    Se il coseno è basso (direzione diversa dalla storia), penalizziamo la distanza.
        #    alpha = 1.0 significa che a 90° (cos=0) la distanza percepita raddoppia.
        #    La penalty potrebbe dover dipendere dalla anzianità della famiglia, troppo penalizzante all'inizio.

        growth_factor = None
        alpha = 1.0
        if cosine_sim >= 0.5:   # 0.4 ?
            growth_factor = 0.85
            penalty_factor = 0.85
        elif cosine_sim >= 0.2:
            growth_factor = 0.6          
            penalty_factor = 1.05
        elif cosine_sim >= 0:             
            penalty_factor = 1 + (alpha * 0.2 * (1 - cosine_sim))      #da 0.2 a 0.1?   
        else:             
            penalty_factor = 1 + alpha * (1 - cosine_sim)

        '''
        if nun_of_nodes < 4:
            alpha = 0.1  
        elif nun_of_nodes < 7:
            alpha = 0.2
        elif nun_of_nodes < 10:
            alpha = 0.3
        else:
            alpha = 0.5
        penalty_factor = 1 + alpha * (1 - cosine_sim)
        '''
        
        dist_penalized = dist_to_centroid * penalty_factor
        
        # 3. Statistica Robusta (Median + MAD) unita al punto 4. Vincolo Evolutivo (Max Cap)
        #    La soglia non può espandersi all'infinito. Limitiamo al max storico + margine. 
        #    Cerchiamo il raggio massimo storico della famiglia
        #    Il cap è il massimo tra il raggio storico e la distanza attuale dalla radice (con margine)
        #    Usiamo dist_to_root come riferimento per "quanto lontano può andare"
        if num_of_nodes <= 3: #minore uguale invece che minore
            # Cold Start: Se abbiamo <= 3 distanze, la statistica è inaffidabile.
            # Usiamo un'euristica basata sul max storico o sul min_threshold.
            stats_threshold = max(self.min_threshold * 5, max_dist_from_centroid * 1.5) 
            evolutionary_cap = stats_threshold
        else:
            if max_family_radius is not None:
                max_history_dist = np.max(self.distances_history) if len(self.distances_history) > 0 else 0.0
                evolutionary_cap = max(max_history_dist, max_family_radius) * 1.5
                # Safety floor per evitare cap troppo stretti all'inizio
                evolutionary_cap = max(evolutionary_cap, self.min_threshold * 2)
                if growth_factor:
                    stats_threshold = max_dist_from_centroid * (1 + growth_factor)
                else:
                    k = self._get_adaptive_k()
                    stats_threshold = max_dist_from_centroid + (k * mad_val)
            else:
                evolutionary_cap = float('inf') 

        # 5. Soglia Finale Bounded
        #    La soglia è statistica, ma "clippata" dal tetto evolutivo.
        #    Mai inferiore al min_threshold globale.
        final_threshold = max(self.min_threshold, min(stats_threshold, evolutionary_cap)) * 1.018 # arrotondamento dell'1%
        
        # Decisione
        is_accepted = dist_penalized <= final_threshold
        
        # Calcolo Confidence (Scala lineare inversa)
        # Se dist_penalized è esattamente sulla soglia -> conf = 0.5
        # Se dist_penalized è 0 -> conf = 1.0
        if dist_penalized > final_threshold:
             # Decrescita rapida fuori soglia
             confidence = max(0.0, 0.5 * (final_threshold / dist_penalized))
        else:
             # Crescita lineare dentro soglia
             confidence = 0.5 + 0.5 * ((final_threshold - dist_penalized) / final_threshold)
        
        reason = (f"Score: {dist_penalized:.4f} (L2: {dist_to_centroid:.4f}, Pen: {penalty_factor:.2f}) "
                  f"vs Thr: {final_threshold:.4f} [Stat: {stats_threshold:.4f}, Cap: {evolutionary_cap:.4f}]")
        
        return is_accepted, float(confidence), final_threshold, reason

class FamilyClusteringSystem:
    """
    Manage model families and automatic clustering based on weight similarities.
    
    Features:
    - Assign new models to existing families or create new ones
    - Maintain family centroids and statistics
    - Use configurable distance thresholds and clustering algorithms
    - Integrate with existing Family database model
    """
    
    def __init__(self,
                 distance_calculator: Optional[ModelDistanceCalculator] = None,
                 family_threshold: float = 0.2,
                 min_family_size: int = 2,
                 clustering_method: ClusteringMethod = ClusteringMethod.THRESHOLD):
        """
        Initialize the family clustering system.
        
        Args:
            distance_calculator: Calculator for model distances
            family_threshold: Distance threshold for family assignment
            min_family_size: Minimum number of models to form a family
            clustering_method: Method to use for clustering
        """
        self.distance_calculator = distance_calculator or ModelDistanceCalculator()
        self.family_threshold = family_threshold
        self.min_family_size = min_family_size
        self.clustering_method = clustering_method
        
        # Centroid storage configuration - use same approach as UPLOAD_FOLDER
        self.centroids_dir = os.path.join('weights', 'centroids')
        os.makedirs(self.centroids_dir, exist_ok=True)
    
    def get_centroid_file_path(self, family_id: str) -> str:
        """Get the file path for a family's centroid file using SafeTensors format."""
        return os.path.join(self.centroids_dir, f"{family_id}.safetensors")
    
    def save_family_centroid(self, family_id: str, centroid: Dict[str, Any]) -> bool:
        """
        Save family centroid to SafeTensors file with enhanced metadata.
        
        Args:
            family_id: ID of the family
            centroid: Dictionary containing centroid weights
            
        Returns:
            True if successfully saved, False otherwise
        """
        logger.info(f"=== ATTEMPTING TO SAVE CENTROID FOR FAMILY {family_id} ===")
        logger.info(f"Absolute centroids path: {os.path.abspath(self.centroids_dir)}")
        
        try:
            if not centroid:
                logger.warning(f"Cannot save empty centroid for family {family_id}")
                return False
            
            centroid_path = self.get_centroid_file_path(family_id)
            logger.info(f"Target path: {centroid_path}")
            logger.info(f"Absolute target path: {os.path.abspath(centroid_path)}")
            
            # Ensure centroids directory exists
            os.makedirs(os.path.dirname(centroid_path), exist_ok=True)
            logger.info(f"✅ Directory created/verified: {os.path.dirname(centroid_path)}")
            
            # Prepare metadata for SafeTensors
            metadata = {
                'family_id': family_id,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'version': '1.0',
                'layer_count': str(len(centroid)),
                'layer_keys': str(list(centroid.keys())),
                'distance_metric': 'L2',
                'format': 'safetensors'
            }
            
            # Save using SafeTensors format
            safetensors.torch.save_file(centroid, centroid_path, metadata=metadata)
            logger.info(f"✅ SUCCESSFULLY SAVED centroid to {centroid_path} (SafeTensors format)")
            
            # Verify that the file was actually created
            if os.path.exists(centroid_path):
                file_size = os.path.getsize(centroid_path)
                logger.info(f"✅ File verified: {centroid_path} ({file_size} bytes)")
            else:
                logger.error(f"❌ File NOT found after save: {centroid_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving centroid for family {family_id}: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def load_family_centroid(self, family_id: str) -> Optional[Dict[str, Any]]:
        """
        Load family centroid from SafeTensors file.
        
        Args:
            family_id: ID of the family
            
        Returns:
            Dictionary containing centroid weights or None if not found
        """
        try:
            centroid_path = self.get_centroid_file_path(family_id)
            
            if not os.path.exists(centroid_path):
                logger.debug(f"No centroid file found for family {family_id} at {centroid_path}")
                return None
            
            # Check if it's a SafeTensors file
            if centroid_path.endswith('.safetensors'):
                try:
                    # Load SafeTensors file
                    centroid_data = safetensors.torch.load_file(centroid_path)
                    logger.info(f"Loaded SafeTensors centroid for family {family_id} from {centroid_path}")
                    return centroid_data
                except Exception as st_error:
                    logger.warning(f"Failed to load as SafeTensors, trying legacy format: {st_error}")
            
            # Fallback to legacy PyTorch format for backward compatibility
            centroid_data = torch.load(centroid_path, map_location='cpu')
            
            # Handle both new format (with metadata) and old format (direct weights)
            if isinstance(centroid_data, dict) and 'weights' in centroid_data:
                logger.info(f"Loaded legacy centroid for family {family_id} from {centroid_path}")
                return centroid_data['weights']
            else:
                # Assume it's the old format - direct weights
                logger.info(f"Loaded legacy centroid for family {family_id} from {centroid_path}")
                return centroid_data
                
        except Exception as e:
            logger.error(f"Error loading centroid for family {family_id}: {e}")
            return None

    def max_distance_root_leaves(self, root_weights: Dict[str, Any], best_family_id: str) -> Any:
        try:
            max_distance = 0.0
            family_leaves = neo4j_service.get_family_leaves(best_family_id)
            
            for model in family_leaves:
                model_weights = load_model_weights(model['file_path'])
                if model_weights is None:
                    continue
                
                distance = self.distance_calculator.calculate_distance(
                    root_weights, model_weights, DistanceMetric.L2_DISTANCE
                )
                
                if distance > max_distance:
                    max_distance = distance
            
            return max_distance
            
        except Exception as e:
            logHandler.error_handler(e, "max_distance_root_leaves")
            return None

    def assign_model_to_family(self, 
                             model: Model,
                             model_weights: Optional[Dict[str, Any]] = None) -> Tuple[str, float]:
        """
        Assign a model to an existing family or create a new one using FamilyGuardian logic.
        Uses adaptive thresholds, robust statistics, and directional analysis.
        """
        try:
            # 1. Caricamento Pesi
            if model_weights is None:
                model_weights = load_model_weights(model.file_path)
                if model_weights is None:
                    raise Exception("Model weights could not be loaded")
            
            logger.info(f"[SEARCHING FAMILY] Model structural_hash:{model.structural_hash}")

            # 2. Identificazione Candidati
            if model.is_foundation_model: 
                candidate_centroids = neo4j_service.get_all_centroids_without_foundation(model.structural_hash)
            else:
                candidate_centroids = neo4j_service.get_all_centroids(model.structural_hash)

            # Caso Nuova Famiglia
            if not candidate_centroids:
                logger.info("No candidates found. Creating new family.")
                family_id = self.create_new_family(model, model_weights)
                confidence = 1.0

            # 3. Best Match L2 sul Centroide
            else:
                best_family_id, best_distance_l2 = self.find_best_family_match(
                model_weights, candidate_centroids
                )

                # =========================================================
                # LOGICA FAMILY GUARDIAN
                # =========================================================
                
                # A. Recupero dati storici (Distanze padre-figlio)
                raw_distances = neo4j_service.get_direct_relationship_distances(best_family_id)
                
                # B. Carica Centroide
                centroid_data = self.load_family_centroid(best_family_id)
            
                # C. Carica Radice
                root_weights = None
                dist_to_root = None
                cosine_sim = 1.0 # default
                
                try:
                    # Recuperiamo la radice (esiste sempre: o flaggata o la più vecchia)
                    root_model = neo4j_service.get_family_root(best_family_id)
                    
                    if root_model:
                        # Carichiamo i pesi se il file esiste
                        root_weights = load_model_weights(root_model['file_path'])
                        if root_weights is None:
                            logger.warning(f"Root model found ({root_model['id']}) but weights file missing.")
                    else:
                        logger.error(f"Logic Error: Family {best_family_id} exists but has no root/members.")

                except Exception as e:
                    logger.warning(f"Error retrieving root weights for family {best_family_id}: {e}")

                # D. Calcoli Geometrici Avanzati (Solo se file radice e centroide esistono)
                if root_weights and centroid_data:
                    # Safe Harbor metric
                    max_family_radius = self.max_distance_root_leaves(root_weights, best_family_id)
                    
                    dist_to_root = self.distance_calculator.calculate_distance(
                        root_weights, model_weights, DistanceMetric.L2_DISTANCE
                    )
                    
                    # Directional Metric
                    cosine_sim = MetricUtils.calculate_directional_cosine(
                        root_weights, centroid_data, model_weights
                    )

                    median, mad_val, max_dist_from_centroid, num_of_nodes = extract_mad_values(best_family_id, centroid_data, self.distance_calculator)

                # E. Interroga il Guardiano
                guardian = FamilyGuardian(raw_distances, min_threshold=self.family_threshold)

                is_accepted, confidence, threshold_used, reason = guardian.check_admissibility(
                    dist_to_centroid=best_distance_l2,
                    dist_to_root=dist_to_root,
                    cosine_sim=cosine_sim,
                    max_family_radius=max_family_radius,
                    median=median,
                    mad_val=mad_val,
                    max_dist_from_centroid=max_dist_from_centroid,
                    num_of_nodes=num_of_nodes
                )
                
                logger.info(f"Family {best_family_id} check: {reason} | Decision: {'ACCEPTED' if is_accepted else 'REJECTED'}")

                # F. Decisione Finale
                if is_accepted and confidence > MIN_CONFIDENCE:
                    family_id = best_family_id
                else:
                    logger.info(f"Model rejected from {best_family_id} (Conf: {confidence:.2f}). Creating new family.")
                    family_id = self.create_new_family(model, model_weights)
                    confidence = 1.0
                
            # G. Aggiornamento DB
            neo4j_service.update_model(model.id, {'family_id': family_id})

            if model.is_foundation_model:
                updates = {
                    'has_foundation_model': True,
                    'updated_at': datetime.now(timezone.utc)
                }
                neo4j_service.update_family(family_id, updates)

            return family_id, confidence

        except Exception as e:
            logHandler.error_handler(e, "assign_model_to_family")
            return self.create_new_family(model, model_weights), 1.0

    def calculate_family_centroid(self, family_id: str, new_model: Model) -> Optional[Dict[str, Any]]:
        """
        Calculate the centroid (average weights) for a family.
        
        Args:
            family_id: ID of the family
            
        Returns:
            Dictionary representing the family centroid weights
        """
        try:
            if not family_id:
                return None
            
            # Load weights for the new model
            new_model_weights = load_model_weights(new_model.file_path)
            if not new_model_weights:
                return None
            
            current_centroid = neo4j_service.get_centroid_by_family_id(family_id)

            # Calculate centroid by averaging weights
            updated_centroid = self.calculate_weights_centroid(current_centroid, new_model_weights)
            
            # Save centroid to file for incremental updates
            if updated_centroid:
                self.save_family_centroid(family_id, updated_centroid)
                
                # Update Neo4j centroid node with actual embedding and metadata
                try:
                    if neo4j_service.is_connected():

                        # Update enhanced Centroid node with metadata
                        neo4j_service.update_centroid_metadata(family_id)
                        
                except Exception as neo4j_error:
                    logHandler.warning_handler(f"Failed to update Neo4j centroid for family {family_id}: {neo4j_error}", "calculate_family_centroid")
            
            logger.info(f"Calculated centroid for {family_id}")
            return updated_centroid
            
        except Exception as e:
            logHandler.error_handler(f"{e}", "calculate_family_centroid")
            return None
    
    def find_best_family_match(self,
                               model_weights: Dict[str, Any],
                               candidate_centroids: List[Dict[str, Any]]) -> Tuple[str, float]:
        """
        Find the best family match for a model.
        
        Returns:
            Tuple of (best_family_id, confidence_score)
        """
        try:
            distance_metric = DistanceMetric.L2_DISTANCE
            best_family_id = None
            best_distance = float('inf')
            
            for centroid in candidate_centroids:
                centroid_weights = None

                # prendere centroide già esistente della famiglia corrente con un if
                centroid_path = centroid["file_path"]
                family_id = centroid["family_id"]

                if os.path.exists(centroid_path):
                    try:
                        with safe_open(centroid_path, framework="pt") as f:

                            # Create a dictionary to hold the tensors
                            centroid_data = {}
                            for key in f.keys():

                                # Load each tensor and add it to the dictionary
                                # .clone() is often good practice to ensure you have an independent copy
                                centroid_data[key] = f.get_tensor(key).clone()

                        # Now, centroid_data is a dictionary with the loaded tensors
                        centroid_weights = centroid_data

                    except Exception as e:
                        logHandler.error_handler(e, "find_best_family_match", "Error loading safetensors file")
                else:
                    logHandler.error_handler(f"Centroid file does not exist for family {family_id}", "find_best_family_match")

                if centroid_weights is None:
                    continue
                
                distance = self.distance_calculator.calculate_distance(
                    model_weights, centroid_weights, distance_metric
                )
                
                if distance < best_distance:
                    best_distance = distance
                    best_family_id = family_id

            if best_family_id is None:
                return "", 0.0
            
            return best_family_id, best_distance
            
        except Exception as e:
            logHandler.error_handler(e, "find_best_family_match")
    
    def create_new_family(self, model: Model, model_weights: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new family for the model.
        
        Args:
            model: Model object to create family for
            model_weights: Pre-loaded model weights to use as initial centroid
        """
        try:
            family_id = f"family_{str(uuid.uuid4())[:8]}"
            
            # Create family in Neo4j
            family_data = {
                'id': family_id,
                'member_count': 1,
                'avg_intra_distance': 0.0,
                'created_at': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc),
                'has_foundation_model': False
            }
            neo4j_service.create_family(family_data)
            
            if model_weights:
                logger.info(f"Creating initial centroid for family {family_id} from model {model.id}")
                
                initial_centroid = {}
                for param_name, tensor in model_weights.items():
                    if isinstance(tensor, torch.Tensor):

                        # Create a copy of the tensor for the centroid
                        initial_centroid[param_name] = tensor.detach().clone()
                
                # Save the initial centroid
                if initial_centroid:
                    self.save_family_centroid(family_id, initial_centroid)
                    
                    # Update Neo4j centroid node
                    try:
                        if neo4j_service.is_connected():
                            
                            # Update enhanced Centroid node with metadata
                            neo4j_service.create_centroid_with_metadata(family_id, model.structural_hash)
                            
                            neo4j_service.create_has_centroid_relationship(family_id)
                    except Exception as neo4j_error:
                        logHandler.error_handler(neo4j_error, "create_new_family", "Failed to update Neo4j centroid for new family")
                    
                    logger.info(f"✅ Initial centroid created and saved for family {family_id}")
                else:
                    logHandler.error_handler(f"No valid weights found to create centroid for family {family_id}", "create_new_family")
            else:
                logHandler.error_handler(f"No model weights provided, family {family_id} could not be created", "create_new_family")

            logger.info(f"✅ Created new family {family_id} for model {model.id}")
            return family_id
            
        except Exception as e:
            logHandler.error_handler(e, "create_new_family", "Generic error creating new family")

    def normalize_safetensors_layers(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizza i nomi dei layer di un singolo file safetensors.

        Gestisce molteplici architetture Hugging Face (ViT, BERT, GPT, LLaMA, T5, 
        Whisper, CLIP, Swin, ecc.) per garantire compatibilità cross-framework.

        Operazioni applicate:
        1. LayerNorm TensorFlow → PyTorch (.beta → .bias, .gamma → .weight)
        2. Estrazione core strutturale (layer.X, block.X, h.X, etc.)
        3. Rimozione prefissi di wrapping del modello
        4. Normalizzazione abbreviazioni comuni
        5. Pulizia finale
        
        Args:
            weights: Dizionario {layer_name: tensor}

        Returns:
            Dizionario normalizzato {normalized_layer_name: tensor}
            
        Raises:
            Nessuna eccezione; logga warning per collisioni
        """
        if not weights:
            logger.warning("Input weights vuoto")
            return {}

        normalized_weights = {}

        # =========================================================================
        # PATTERN STRUTTURALI (ordinati per priorità/specificità)
        # =========================================================================
        structural_patterns = [
            # Encoder-Decoder specifici (più specifici prima)
            r'(encoder\.layer\.\d+)',
            r'(decoder\.layer\.\d+)',
            r'(encoder\.blocks\.\d+)',
            r'(decoder\.blocks\.\d+)',
            r'(encoder_block\.\d+)',
            r'(decoder_block\.\d+)',
            
            # Transformer generici
            r'(transformer\.h\.\d+)',
            r'(transformer\.layers\.\d+)',
            r'(transformer\.blocks\.\d+)',
            r'(transformer_block\.\d+)',
            
            # Layer/Block generici (LLaMA, GPT, Mistral, etc.)
            r'(model\.layers\.\d+)',
            r'(model\.decoder\.layers\.\d+)',
            r'(gpt_neox\.layers\.\d+)',
            r'(layers\.\d+)',
            r'(blocks\.\d+)',
            r'(layer\.\d+)',
            r'(block\.\d+)',
            r'(h\.\d+)',
            
            # Vision Models (CLIP, ViT, Swin, SAM, etc.)
            r'(vision_model\.encoder\.layers\.\d+)',
            r'(visual\.encoder\.blocks\.\d+)',
            r'(vision_tower\.blocks\.\d+)',
            r'(backbone\.stages\.\d+)',
            
            # Altri pattern
            r'(modules\.\d+)',
            r'(block_list\.\d+)',
        ]

        # =========================================================================
        # PREFISSI DA RIMUOVERE (ordinati alfabeticamente per manutenibilità)
        # =========================================================================
        known_prefixes = [
            # Generici
            'backbone.', 'base_model.', 'decoder.', 'discriminator.', 'encoder.',
            'generator.', 'head.', 'model.', 'proj.', 'student.', 'teacher.', 
            'transformer.', 'unet.', 'vae.',
            
            # Text Models
            'bart.', 'bert.', 'bloom.', 'deberta.', 'distilbert.', 'falcon.', 
            'gpt_neox.', 'gptj.', 'llama.', 'marian.', 'mistral.', 'mt5.', 
            'opt.', 'roberta.', 't5.',
            
            # Vision Models
            'beit.', 'clip.', 'clip_model.', 'clip_vision_model.', 'convnext.', 
            'dino.', 'dinov2.', 'open_clip.', 'sam.', 'swin.', 'vit.',
            
            # Multimodal
            'clip.', 'image_encoder.', 'speech_encoder_decoder.', 'text_encoder.', 
            'text_model.', 'vision_model.', 'vision_tower.', 'visual.', 'whisper.',
        ]

        # =========================================================================
        # ABBREVIAZIONI DA NORMALIZZARE
        # =========================================================================
        abbreviation_map = {
            '.ln_f.': '.layernorm_final.',
            '.ln_1.': '.layernorm_before.',
            '.ln_2.': '.layernorm_after.',
            '.ln.': '.layernorm.',
            '.attn.': '.attention.',
            '.mlp.': '.feedforward.',
            '.c_attn.': '.attention.combined.',  
            '.c_proj.': '.attention.projection.',
            '.c_fc.': '.feedforward.fc.',
        }

        # =========================================================================
        # NORMALIZZAZIONE
        # =========================================================================
        collision_count = 0
        
        for original_name, tensor in weights.items():
            normalized_name = original_name

            # Step 1: TensorFlow → PyTorch LayerNorm
            normalized_name = normalized_name.replace('.beta', '.bias')
            normalized_name = normalized_name.replace('.gamma', '.weight')

            # Step 2: Cerca pattern strutturali (dal più specifico al generico)
            core_name_found = False
            for pattern in structural_patterns:
                match = re.search(pattern, normalized_name)
                if match:
                    start_idx = match.start()
                    normalized_name = normalized_name[start_idx:]
                    core_name_found = True
                    break

            # Step 3: Se non trovato pattern, rimuovi prefissi noti
            if not core_name_found:
                for prefix in known_prefixes:
                    if normalized_name.startswith(prefix):
                        normalized_name = normalized_name[len(prefix):]
                        break

            # Step 4: Normalizza abbreviazioni comuni
            for abbrev, expanded in abbreviation_map.items():
                normalized_name = normalized_name.replace(abbrev, expanded)

            # Step 5: Pulizia finale - rimuovi prefissi residui
            # (potrebbe esserci dopo Step 2 se pattern non era all'inizio)
            normalized_name = re.sub(
                r'^(?:model|base_model|transformer)\.',
                '',
                normalized_name
            )

            # Step 6: Gestisci collisioni
            if normalized_name in normalized_weights:
                collision_count += 1
                logger.warning(
                    f"Collisione normalizzazione #{collision_count}: "
                    f"'{original_name}' → '{normalized_name}' (già esistente). "
                    f"Layer scartato."
                )
                continue

            # Step 7: Aggiungi al risultato
            normalized_weights[normalized_name] = tensor

        # =========================================================================
        # LOGGING FINALE
        # =========================================================================
        original_count = len(weights)
        normalized_count = len(normalized_weights)
        
        if collision_count > 0:
            logger.warning(
                f"Normalizzazione completata con {collision_count} collisioni: "
                f"{original_count} layer originali → {normalized_count} layer normalizzati "
                f"({original_count - normalized_count} scartati)"
            )
        else:
            logger.info(
                f"Normalizzazione completata: "
                f"{original_count} layer originali → {normalized_count} layer normalizzati"
            )

        return normalized_weights

    @torch.no_grad()
    def calculate_weights_centroid(
        self, 
        current_centroid: Dict[str, Any], 
        new_model_weights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calcola centroide aggiornato aggiungendo un nuovo modello alla media esistente.
        
        Formula: centroid_new = (centroid_old * n + new_model) / (n + 1)
        
        Caratteristiche:
        - Aggiornamento in-place di current_centroid
        - Calcoli in float32 per massima precisione
        - Chunking automatico per tensori grandi (>10M elementi)
        - Solo layer comuni (nome esatto + shape identica)
        - Layer non comuni nel centroide rimangono invariati
        
        Args:
            current_centroid: Dizionario con centroide corrente (include 'model_count')
            new_model_weights: Dizionario con pesi del nuovo modello
            
        Returns:
            Centroide aggiornato (stesso oggetto di current_centroid, modificato in-place)
            
        Note:
            - I nomi dei layer devono essere già normalizzati
            - Gestisce automaticamente device mismatch
            - Tensori non compatibili vengono esclusi (con logging)
        """
        try:
            # Step 1: Validazione input
            if not new_model_weights:
                logger.warning("Pesi del nuovo modello vuoti, impossibile calcolare centroide")
                return current_centroid
            
            n = current_centroid.get('model_count', 0)
            
            # Non dovrebbe mai accadere
            if n <= 0:
                return logHandler.error_handler(f"model_count invalido nel centroide: {n}", "calculate_weights_centroid")
                 
            
            # Step 2: Trova layer comuni (intersezione per nome esatto)
            current_centroid_weights = load_model_weights(current_centroid['file_path'])

            centroid_layers = set(current_centroid_weights.keys()) 
            new_model_layers = set(new_model_weights.keys())

            common_layers = centroid_layers & new_model_layers
            
            logger.info(
                f"Layer: {len(centroid_layers)} nel centroide, "
                f"{len(new_model_layers)} nel nuovo modello, "
                f"{len(common_layers)} comuni"
            )
            
            if not common_layers:
                logger.warning("Nessun layer comune trovato tra centroide e nuovo modello")
                return current_centroid_weights
            
            # Step 3: Aggiorna layer comuni
            updated_count = 0
            excluded_count = 0
            chunked_layers = []
            
            for layer_name in common_layers:
                centroid_tensor = current_centroid_weights[layer_name]
                new_model_tensor = new_model_weights[layer_name]
                
                # Validazione 1: Entrambi devono essere tensori PyTorch
                if not (isinstance(centroid_tensor, torch.Tensor) and 
                        isinstance(new_model_tensor, torch.Tensor)):
                    logger.warning(
                        f"Layer '{layer_name}' escluso: non entrambi tensori PyTorch "
                        f"(centroid: {type(centroid_tensor)}, new: {type(new_model_tensor)})"
                    )
                    excluded_count += 1
                    continue
                
                # Validazione 2: Shape identiche
                if centroid_tensor.shape != new_model_tensor.shape:
                    logger.warning(
                        f"Layer '{layer_name}' escluso: shape mismatch "
                        f"({centroid_tensor.shape} vs {new_model_tensor.shape})"
                    )
                    excluded_count += 1
                    continue
                
                # Validazione 3: Device matching
                if centroid_tensor.device != new_model_tensor.device:
                    logger.debug(
                        f"Layer '{layer_name}': spostamento device "
                        f"{new_model_tensor.device} → {centroid_tensor.device}"
                    )
                    new_model_tensor = new_model_tensor.to(centroid_tensor.device)
                
                # Conversione a float32 per calcoli (massima precisione)
                original_dtype = centroid_tensor.dtype
                if centroid_tensor.dtype != torch.float32:
                    centroid_tensor = centroid_tensor.float()
                if new_model_tensor.dtype != torch.float32:
                    new_model_tensor = new_model_tensor.float()
                
                # Ottimizzazione: contiguous solo se necessario
                if not centroid_tensor.is_contiguous():
                    centroid_tensor = centroid_tensor.contiguous()
                if not new_model_tensor.is_contiguous():
                    new_model_tensor = new_model_tensor.contiguous()
                
                # Calcolo media con chunking automatico per tensori grandi
                numel = centroid_tensor.numel()
                
                if numel > CENTROID_CHUNK_THRESHOLD:
                    # CHUNKING: suddividi in blocchi per evitare OOM
                    chunk_size = CENTROID_CHUNK_THRESHOLD
                    num_chunks = (numel + chunk_size - 1) // chunk_size
                    
                    centroid_flat = centroid_tensor.view(-1)
                    new_model_flat = new_model_tensor.view(-1)
                    
                    for chunk_old, chunk_new in zip(
                        centroid_flat.split(chunk_size),
                        new_model_flat.split(chunk_size)
                    ):
                        # Formula incrementale: (old * n + new) / (n + 1)
                        chunk_old.mul_(n).add_(chunk_new).div_(n + 1)
                    
                    chunked_layers.append((layer_name, numel, num_chunks))
                    
                else:
                    # NORMALE: aggiornamento diretto (in-place)
                    centroid_tensor.mul_(n).add_(new_model_tensor).div_(n + 1)
                
                # Riconverti al dtype originale se necessario
                if centroid_tensor.dtype != original_dtype:
                    centroid_tensor = centroid_tensor.to(original_dtype)
                
                # Salva il layer aggiornato (in-place su current_centroid)
                current_centroid_weights[layer_name] = centroid_tensor
                updated_count += 1
            
            # Step 4: Logging finale
            logger.info(
                f"✅ Centroide aggiornato: {updated_count} layer modificati, "
                f"{excluded_count} layer esclusi, "
                f"{len(centroid_layers) - len(common_layers)} layer invariati"
            )
            
            if chunked_layers:
                logger.info(f"Chunking applicato a {len(chunked_layers)} layer grandi:")
                for layer_name, numel, num_chunks in chunked_layers[:5]:  # Primi 5
                    logger.info(
                        f"  - {layer_name}: {numel:,} elementi "
                        f"({num_chunks} chunk da {CENTROID_CHUNK_THRESHOLD:,})"
                    )
            
            if updated_count == 0:
                logHandler.warning_handler("Nessun layer aggiornato: centroide invariato", "calculate_weights_centroid")
            
            return current_centroid_weights
            
        except Exception as e:
            logHandler.error_handler(f"❌ Errore durante calcolo centroide: {e}", "calculate_weights_centroid")
            return current_centroid_weights