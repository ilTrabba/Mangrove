"""
ModelDistanceCalculator

This module provides distance calculation between AI model weights using different metrics
optimized for various model types (fully fine-tuned models vs LoRA models).
"""

import logging
import numpy as np
import torch

from typing import Dict, List, Optional, Any
from src.log_handler import logHandler
from ..db_entities.entity import Model
from enum import Enum
from src.utils.architecture_filtering import FilteringPatterns
from src.mother_algorithm.mother_utils import load_model_weights

logger = logging.getLogger(__name__)

class DistanceMetric(Enum):
    """Available distance metrics for model comparison"""
    L2_DISTANCE = "l2_distance"
    COSINE_SIMILARITY = "cosine_similarity"
    HYBRID_DISTANCE = "hybrid_distance"
    RMS_L2_DISTANCE = "RMS_L2"
    MAE = "MAE"

class ModelType(Enum):
    """Model types for optimized distance calculation"""
    FULL_FINETUNED = "full_finetuned"
    LORA = "lora"
    AUTO = "auto"

class ModelDistanceCalculator:
    """
    Calculate distances between AI model weights using different metrics.
    
    Supports:
    - L2 distance for fully fine-tuned models
    - Matrix rank method for LoRA models  
    - Filtered layer analysis (attention, dense, linear layers)
    """
    
    def __init__(self, 
                 default_metric: DistanceMetric = DistanceMetric.L2_DISTANCE):
        """
        Initialize the distance calculator.
        
        Args:
            default_metric: Default distance metric to use
            layer_filter: List of layer patterns to include. If None, uses default patterns.
        """
        self.default_metric = default_metric

    def calculate_l2_layer_distance(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        """
        Calculate L2 distance between two tensors (single layer).
        
        Args:
            tensor1: First tensor
            tensor2: Second tensor
            
        Returns:
            L2 distance as float
        """

        tensor1 = tensor1.detach().cpu()
        tensor2 = tensor2.detach().cpu()

        if tensor1.dtype == torch.bfloat16:
            tensor1 = tensor1.float()
        if tensor2.dtype == torch.bfloat16:
            tensor2 = tensor2.float()

        diff = tensor1.numpy() - tensor2.numpy()
        l2_dist = np.linalg.norm(diff.flatten())
        return l2_dist

    def calculate_cosine_layer_distance(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        """
        Calculate Cosine distance between two tensors (single layer).
        
        Args:
            tensor1: First tensor
            tensor2: Second tensor
            
        Returns:
            Cosine distance as float, or None if calculation fails (e.g., zero norm)
        """

        tensor1 = tensor1.detach().cpu()
        tensor2 = tensor2.detach().cpu()

        if tensor1.dtype == torch.bfloat16:
            tensor1 = tensor1.float()
        if tensor2.dtype == torch.bfloat16:
            tensor2 = tensor2.float()

        vec1 = tensor1.numpy().flatten()
        vec2 = tensor2.numpy().flatten()
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # Handle edge case: zero vectors
        if norm1 == 0 or norm2 == 0:
            return None
        
        dot_product = np.dot(vec1, vec2)
        cosine_similarity = dot_product / (norm1 * norm2)
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
        cosine_dist = 1.0 - cosine_similarity
        
        return cosine_dist

    def calculate_rms_l2_layer_distance(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        """
        Calculate RMS-L2 distance between two tensors (single layer).
        
        RMS (Root Mean Square) normalizes by the number of elements.
        
        Args:
            tensor1: First tensor
            tensor2: Second tensor
            
        Returns:
            RMS-L2 distance as float
        """

        tensor1 = tensor1.detach().cpu()
        tensor2 = tensor2.detach().cpu()

        if tensor1.dtype == torch.bfloat16:
            tensor1 = tensor1.float()
        if tensor2.dtype == torch.bfloat16:
            tensor2 = tensor2.float()

        diff = tensor1.numpy() - tensor2.numpy()
        diff_flat = diff.flatten()
        rms_dist = np.sqrt(np.mean(diff_flat ** 2))
        return rms_dist

    def calculate_hybrid_layer_distance(self, 
                                   tensor1: torch.Tensor, 
                                   tensor2: torch.Tensor) -> float:
        """
        Calculate Hybrid distance (α·L2 + (1-α)·Cosine) between two tensors (single layer).
        
        Args:
            tensor1: First tensor
            tensor2: Second tensor
            alpha: Weight for L2 distance (0.0 to 1.0)
            
        Returns:
            Hybrid distance as float, or None if calculation fails
        """
        # Variabile per assegnare il giusto "peso" alle 2 metriche (alpha weighting)
        alpha = 0.3

        # Calculate L2 component
        l2_dist = self.calculate_l2_layer_distance(tensor1, tensor2)
        
        # Calculate Cosine component
        cosine_dist = self.calculate_cosine_layer_distance(tensor1, tensor2)
        
        # If cosine calculation failed (zero norm), return None
        if cosine_dist is None:
            return None
        
        # Combine with alpha weighting
        hybrid_dist = alpha * l2_dist + (1 - alpha) * cosine_dist
        
        return hybrid_dist
    
    # excluded_patterns: Optional[frozenset] = None, se si vuole passare i pattern da escludere come argomento
    def calculate_distance(self, 
                      weights1: Dict[str, Any], 
                      weights2: Dict[str, Any], 
                      metric_type: DistanceMetric = None,
                      ) -> float:
        """
        Calculate distance between two sets of model weights using specified metric.
        
        Args:
            weights1: First model's normalized weights
            weights2: Second model's normalized weights
            metric_type: Type of distance metric to use. Options: "l2", "cosine", "rms_l2", "hybrid"
            excluded_patterns: Set of patterns for layers to exclude. If None, uses EXCLUDED_LAYER_PATTERNS
            alpha: Weight for L2 in hybrid metric (only used when metric_type="hybrid")
            
        Returns:
            Average distance across common parameters, or inf if no valid layers
            
        Raises:
            ValueError: If metric_type is not one of the supported metrics
        """
        # Validate metric type
        valid_metrics = [DistanceMetric.L2_DISTANCE, DistanceMetric.COSINE_SIMILARITY, DistanceMetric.RMS_L2_DISTANCE, DistanceMetric.HYBRID_DISTANCE]
        if metric_type not in valid_metrics:
            raise logHandler.error_handler(f"Invalid metric_type '{metric_type}'. Must be one of {valid_metrics}","calculate_distance")
        
        # Use default excluded patterns if none provided
        
        excluded_patterns = FilteringPatterns.FULL_MODEL
        
        try:
            # Get common parameters (intersection)
            common_params = set(weights1.keys()) & set(weights2.keys())
            
            if not common_params:
                logger.warning("No common parameters found between models")
                return float('inf')
            
            logger.debug(f"Found {len(common_params)} common parameters")
            
            total_distance = 0.0
            param_count = 0
            excluded_count = 0
            
            for param_name in common_params:

                # Convert to lowercase once for case-insensitive matching
                param_lower = param_name.lower()
                
                # Exclude layers matching any pattern in blacklist
                if any(pattern in param_lower for pattern in excluded_patterns):
                    excluded_count += 1
                    continue
                
                tensor1 = weights1[param_name]
                tensor2 = weights2[param_name]
                
                # Verify both are tensors
                if not (isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor)):
                    logger.info(f"Skipping {param_name}: not both tensors")
                    continue
                
                # Ensure same shape
                if tensor1.shape != tensor2.shape:
                    """
                    logHandler.warning_handler(
                        f"Shape mismatch for {param_name}: "
                        f"{tensor1.shape} vs {tensor2.shape}", "calculate_distance"
                    )
                    """
                    excluded_count += 1
                    continue
                
                if len(tensor1.shape)!=2 or len(tensor2.shape)!=2 or tensor1.shape[0]!=tensor1.shape[1] or tensor2.shape[0]!=tensor2.shape[1]:
                    """
                    logHandler.warning_handler(
                        f"Shape not quadratic for {param_name}: "
                        f"At least one between tensor1:{tensor1.shape} or tensor2:{tensor2.shape} is not squared", "calculate_distance"
                    )
                    """
                    excluded_count += 1
                    continue
                
                # Calculate layer distance using appropriate metric
                if metric_type == DistanceMetric.L2_DISTANCE:
                    layer_distance = self.calculate_l2_layer_distance(tensor1, tensor2)
                elif metric_type == DistanceMetric.COSINE_SIMILARITY:
                    layer_distance = self.calculate_cosine_layer_distance(tensor1, tensor2)
                elif metric_type == DistanceMetric.RMS_L2_DISTANCE:
                    layer_distance = self.calculate_rms_l2_layer_distance(tensor1, tensor2)
                elif metric_type == DistanceMetric.HYBRID_DISTANCE:
                    layer_distance = self.calculate_hybrid_layer_distance(tensor1, tensor2)
                
                # Skip layer if distance calculation failed (e.g., zero norm for cosine)
                if layer_distance is None:
                    logHandler.warning_handler(f"Distance calculation failed for {param_name}, skipping layer","calculate_distance")
                    continue
                
                total_distance += layer_distance
                param_count += 1
            
            # Log statistics
            #logger.info(
            #    f"{metric_type} distance calculation: {param_count} layers included, "
            #    f"{excluded_count} layers excluded"
            #)
            
            if param_count == 0:
                logHandler.warning_handler("No valid layers found for distance calculation","calculate_distance")
                return float('inf')
            
            # Return average distance
            avg_distance = total_distance / param_count
            #logger.info(f"Average {metric_type} distance: {avg_distance:.6f}")
            
            return avg_distance
            
        except Exception as e:
            logHandler.error_handler(f"Failed to calculate {metric_type} distance: {e}","calculate_distance")
            return float('inf')

    def calculate_intra_family_distance(self, family_models: List[Model]) -> float:
        """
        Calculate average intra-family distance.
        """
        try:
            if len(family_models) < 2:
                return 0.0
            
            # Load weights for all models
            model_weights = {}
            for model in family_models:
                weights = load_model_weights(model.file_path)
                if weights is not None:
                    model_weights[model.id] = weights
            
            if len(model_weights) < 2:
                return 0.0
            
            # Calculate pairwise distances
            distances = []
            model_ids = list(model_weights.keys())
            
            for i in range(len(model_ids)):
                for j in range(i + 1, len(model_ids)):
                    dist = self.calculate_distance(
                        model_weights[model_ids[i]],
                        model_weights[model_ids[j]]
                    )
                    distances.append(dist)
            
            return np.mean(distances) if distances else 0.0
            
        except Exception as e:
            logHandler.error_handler(f"Error calculating intra-family distance: {e}", "calculate_intra_family_distance")
            return 0.0
    
    def calculate_std_intra_distance(self, direct_relationship_distances: List[float], avg_intra_distance: float) -> float:
        """
        Calculate standard deviation of intra-family distances.
        
        Only considers distances from direct parent-child relationships (edges in the family tree),
        not distances between all pairs of family members.
        
        Args:
            direct_relationship_distances: List of distances from all direct parent-child edges in the family
            
        Returns:
            Standard deviation of the distances, or 0. 0 if insufficient data
        """
        if len(direct_relationship_distances) < 2:
            # Need at least 2 relationships to calculate meaningful std
            return 0.0
        
        # Calculate variance
        variance = sum((d - avg_intra_distance) ** 2 for d in direct_relationship_distances) / len(direct_relationship_distances)
        
        # Calculate standard deviation
        std_distance = variance ** 0.5
        
        return std_distance