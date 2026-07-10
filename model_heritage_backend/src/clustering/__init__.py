"""
Model Heritage Clustering System

This module provides comprehensive clustering capabilities for AI model heritage detection,
including distance calculation, family clustering, genealogical tree building, and
complete pipeline management.

Components:
- ModelDistanceCalculator: Calculate distances between model weights
- FamilyClusteringSystem: Manage model families and clustering  
- MoTHerTreeBuilder: Build genealogical relationships within families
- ModelManagementSystem: Complete pipeline coordination
"""

from .distance_calculator import ModelDistanceCalculator
from .family_clustering import FamilyClusteringSystem  
from src.mother_algorithm.tree_builder import MoTHerTreeBuilder
from .model_management import ModelManagementSystem

__all__ = [
    'ModelDistanceCalculator',
    'FamilyClusteringSystem', 
    'MoTHerTreeBuilder',
    'ModelManagementSystem'
]

__version__ = '1.0.0'