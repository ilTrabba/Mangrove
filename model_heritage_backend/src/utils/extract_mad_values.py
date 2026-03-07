import numpy as np

from src.clustering.distance_calculator import DistanceMetric, ModelDistanceCalculator
from src.mother_algorithm.mother_utils import load_model_weights
from src.services.neo4j_service import neo4j_service
from src.utils.architecture_filtering import FilteringPatterns
from typing import Dict, Tuple
from typing import Dict, Any
from src.log_handler import logHandler

def extract_mad_values(best_family_id: str, centroid_data: Dict[str,Any], distance_calculator: ModelDistanceCalculator) -> Tuple[float, float, int]:
    """
    Estrae i valori di mediana e MAD (Median Absolute Deviation) dai dati del centroide.
    
    Args:
        best_familiy_id: ID della famiglia del modello
        centroid_data: Dizionario contenente i dati del centroide, inclusi mediana e MAD
        
    Returns:
        Tuple contenente (mediana, MAD)
    """
    try:
        distances_from_centroid = []
        family_models = neo4j_service.get_family_models(best_family_id)

        if not family_models:
            logHandler.warning_handler(f"Nessun modello trovato per la famiglia {best_family_id}", "extract_mad_values")
            return 0.0, 0.0
        
        for model in family_models:
            
            if model:
                # Carichiamo i pesi se il file esiste
                model_weights = load_model_weights(model.file_path)
                if model_weights is None:
                    logHandler.warning_handler(f"Model found ({model.id}) but weights file missing.", "extract_mad_values")
                else:
                    distance = distance_calculator.calculate_distance(
                        model_weights, centroid_data, DistanceMetric.L2_DISTANCE
                    )
                    distances_from_centroid.append(distance)
            else:
                logHandler.error_handler(f"Logic Error: Family {best_family_id} exists but has no root/members.", "extract_mad_values")
        
        median_val = np.median(distances_from_centroid)
        mad_val = np.median(np.abs(distances_from_centroid - median_val)) * 1.4826 # MAD (Median Absolute Deviation) normalizzata con StdDev
        max_dist_from_centroid =np.max(distances_from_centroid)
        
        return median_val, mad_val, max_dist_from_centroid, len(distances_from_centroid)

    except Exception as e:
        logHandler.error_handler(f"Errore nell'estrazione dei valori MAD per la famiglia {best_family_id}: {e}", "extract_mad_values")
        return 0.0, 0.0