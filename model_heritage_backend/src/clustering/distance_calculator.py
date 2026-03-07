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
    COSINE_DISTANCE = "cosine_distance"
    REL_FRO_DISTANCE = "rel_fro_distance"
    SPECTRAL_DISTANCE = "spectral_distance"

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

    def calculate_relative_frobenius_layer_distance(self, tensor1: torch.Tensor, tensor2: torch.Tensor, eps: float = 1e-12) -> Optional[float]:
        try:
            t1 = tensor1.detach().cpu()
            t2 = tensor2.detach().cpu()

            if t1.dtype == torch.bfloat16:
                t1 = t1.float()
            if t2.dtype == torch.bfloat16:
                t2 = t2.float()

            if t1.shape != t2.shape:
                return None

            diff_norm = torch.linalg.norm((t1 - t2).reshape(-1), ord=2).item()
            n1 = torch.linalg.norm(t1.reshape(-1), ord=2).item()
            n2 = torch.linalg.norm(t2.reshape(-1), ord=2).item()

            denom = n1 + n2 + eps
            return float(diff_norm / denom)

        except Exception:
            return None

    def to_2d(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2:
            return t
        if t.ndim == 4:
            # Conv: (out, in, kH, kW) -> (out, in*kH*kW)
            return t.reshape(t.shape[0], -1)
        if t.ndim == 1:
            return t.reshape(1, -1)
        # Generic fallback: keep first dim as "rows"
        return t.reshape(t.shape[0], -1)

    def calculate_spectral_layer_distance(self, tensor1: torch.Tensor, tensor2: torch.Tensor, eps: float = 1e-12, topk: Optional[int] = None, relative: bool = True) -> Optional[float]:
        try:
            t1 = tensor1.detach().cpu()
            t2 = tensor2.detach().cpu()

            if t1.dtype == torch.bfloat16:
                t1 = t1.float()
            if t2.dtype == torch.bfloat16:
                t2 = t2.float()

            if t1.shape != t2.shape:
                return None

            m1 = self.to_2d(t1)
            m2 = self.to_2d(t2)

            # Compute singular values
            s1 = torch.linalg.svdvals(m1)
            s2 = torch.linalg.svdvals(m2)

            # Ensure same length (should be, given same shape -> same min(m,n))
            if s1.shape != s2.shape:
                return None

            # Optionally keep only top-k singular values (largest)
            if topk is not None:
                k = min(topk, s1.numel())
                s1 = s1[:k]
                s2 = s2[:k]

            diff = s1 - s2
            dist = torch.linalg.norm(diff, ord=2).item()

            if not relative:
                return float(dist)

            n1 = torch.linalg.norm(s1, ord=2).item()
            n2 = torch.linalg.norm(s2, ord=2).item()
            denom = n1 + n2 + eps
            return float(dist / denom)

        except Exception:
            return None

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
            metric_type: Type of distance metric to use. Options: "l2", "cosine", "rel_fro", "spectral"
            excluded_patterns: Set of patterns for layers to exclude. If None, uses EXCLUDED_LAYER_PATTERNS
            
        Returns:
            Average distance across common parameters, or inf if no valid layers
            
        Raises:
            ValueError: If metric_type is not one of the supported metrics
        """
        # Validate metric type
        valid_metrics = [DistanceMetric.L2_DISTANCE, DistanceMetric.COSINE_DISTANCE, DistanceMetric.REL_FRO_DISTANCE, DistanceMetric.SPECTRAL_DISTANCE]
        if metric_type not in valid_metrics:
            raise logHandler.error_handler(f"Invalid metric_type '{metric_type}'. Must be one of {valid_metrics}","calculate_distance")

        # Use default excluded patterns if none provided
        if excluded_patterns is None:
            excluded_patterns = FilteringPatterns.ATTENTION_ONLY
        
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
                elif metric_type == DistanceMetric.COSINE_DISTANCE:
                    layer_distance = self.calculate_cosine_layer_distance(tensor1, tensor2)
                elif metric_type == DistanceMetric.REL_FRO_DISTANCE:
                    layer_distance = self.calculate_relative_frobenius_layer_distance(tensor1, tensor2)
                elif metric_type == DistanceMetric.SPECTRAL_DISTANCE:
                    layer_distance = self.calculate_spectral_layer_distance(tensor1, tensor2)
                
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