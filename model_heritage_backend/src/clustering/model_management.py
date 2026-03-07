"""
ModelManagementSystem

This module provides complete pipeline integration for model clustering and heritage detection.
It coordinates all clustering components and manages the complete workflow from model addition
to family assignment and tree reconstruction.
"""

import logging
import numpy as np

from typing import Dict, Any
from datetime import datetime, time, timezone
from src.log_handler import logHandler
from src.db_entities.entity import Model
from src.services.neo4j_service import neo4j_service
from .family_clustering import FamilyClusteringSystem, ClusteringMethod
from src.mother_algorithm.tree_builder import MoTHerTreeBuilder, TreeBuildingMethod

logger = logging.getLogger(__name__)

class ModelManagementSystem:
    """
    Complete pipeline integration for model clustering and heritage detection.
    
    Features:
    - Coordinate all clustering components
    - Manage model and family databases  
    - Provide genealogy queries
    - Handle the complete workflow from upload to heritage detection
    """
    
    def __init__(self,
                 family_threshold: float = 1,
                 clustering_method: ClusteringMethod = ClusteringMethod.THRESHOLD,
                 tree_method: TreeBuildingMethod = TreeBuildingMethod.MOTHER,
                 lambda_param: float = 0.5):
        """
        Initialize the model management system.
        
        Args:
            distance_metric: Default distance metric for calculations
            family_threshold: Threshold for family assignment
            clustering_method: Method for family clustering
            tree_method: Method for tree building
            lambda_param: Balance parameter for MoTHer algorithm
        """
        # Initialize components
        self.family_clustering = FamilyClusteringSystem(
            distance_calculator = None,
            family_threshold=family_threshold,
            clustering_method=clustering_method
        )
        self.tree_builder = MoTHerTreeBuilder(
            lambda_param=lambda_param,
            method=tree_method,
            distance_matrix = None
        )
        
        logger.info("Initialized ModelManagementSystem with components")
    
    def salva_tempo(self, nome_modello, delta_clustering_preprocessing, delta_clustering, delta_mother, delta_mother_preprocessing):
        total_clustering_time = delta_clustering_preprocessing + delta_clustering
        total_mother_time = delta_mother + delta_mother_preprocessing

        with open("tempi.txt", "a") as f:
            f.write(f"{nome_modello}: {total_clustering_time}, {total_mother_time}\n")

    def process_new_model(self, model_data:  Dict[str, Any], t_start_clustering, delta_clustering_preprocessing, delta_mother_preprocessing) -> Dict[str, Any]:
        """
        Complete processing pipeline for a new model.
        
        Pipeline:
        1. Family assignment (existing or new)
        2. Tree reconstruction for entire family
        3. Centroid update
        4. Status finalization
        
        Args: 
            model_data: Dictionary with model attributes
            start_time: Timestamp when processing started
        Returns:
            Dictionary with processing results
        """
        model_proxy = None  
        model_id = model_data. get('id', 'unknown') 
        
        try: 
            # Create proxy object for compatibility with clustering system
            model_proxy = Model(**model_data)
            model_id = model_proxy.id
            
            logger.info(f"[SEARCHING FAMILY] Processing model {model_id}")
            
            # =====================================================================
            # Step 1: Assign to family
            # =====================================================================
            family_id, family_confidence = self.family_clustering.assign_model_to_family(model_proxy)
            logger.info(f"[SEARCHING FAMILY] Assigned model {model_id} to family {family_id} (confidence: {family_confidence:.3f})")
            

            delta_clustering = datetime.now() - t_start_clustering
            t_start_mother = datetime.now()
            # =====================================================================
            # Step 2: Build complete family tree
            # =====================================================================
            existing_family_models = neo4j_service.get_family_models(
                family_id=family_id,
                status='ok'
            )
            
            all_family_models = [*existing_family_models, model_proxy]
            
            family_tree, tree_confidence = self.tree_builder.build_family_tree(
                family_id, 
                all_family_models
            )
            
            num_nodes = family_tree.number_of_nodes()
            num_edges = family_tree.number_of_edges()
            
            # Default values
            parent_id = None
            parent_confidence = 0.0
            
            # =====================================================================
            # Step 3: Atomic tree rebuild (if tree has nodes)
            # =====================================================================
            if num_nodes > 0:
                start_time = datetime.now(timezone.utc)
                
                success = neo4j_service.atomic_rebuild_genealogy(
                    family_id, 
                    family_tree, 
                    tree_confidence
                )
                
                delta_mother = datetime.now()- t_start_mother
                elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                if not success:
                    logger.error(f"Failed to rebuild tree for family {family_id}")
                else:
                    logger.info(f"⚡ Rebuilt family {family_id} tree ({num_edges} edges) in {elapsed_ms:.2f}ms")
                
                # Extract parent info for response (no additional DB queries)
                if family_tree.in_degree(model_id) > 0:
                    parent_id = next(family_tree.predecessors(model_id))  
                    parent_confidence = tree_confidence.get(model_id, 0.0)
                    logger.info(f"✅ Model {model_id} parent:  {parent_id} (confidence:  {parent_confidence:.3f})")
                else:
                    logger.info(f"✅ Model {model_id} is a root node")
            else:
                logger.info(f"✅ Model {model_id} assigned as first root in family {family_id}")
                delta_mother = t_start_mother - t_start_mother
            
            self.salva_tempo(model_proxy.name, delta_clustering_preprocessing, delta_clustering, delta_mother, delta_mother_preprocessing)
            # =====================================================================
            # Step 4: Update centroid statistics 
            # =====================================================================
            if num_nodes > 1:
                self.family_clustering.calculate_family_centroid(family_id, model_proxy)
            
            # =====================================================================
            # Step 5: Finalize status
            # =====================================================================
            neo4j_service.update_model(model_id, {'status': 'ok'})
            
            return {
                'model_id':  model_id,
                'family_id': family_id,
                'family_confidence': family_confidence,
                'parent_id': parent_id,
                'parent_confidence': parent_confidence,
                'tree_nodes': num_nodes,
                'tree_edges': num_edges,
                'status': 'success'
            }
            
        except Exception as e: 
            logHandler.error_handler(e, "process_new_model", f"model_id={model_id}")
            
            # Mark model as error state in DB
            if model_id != 'unknown':
                try:
                    neo4j_service.update_model(model_id, {'status': 'error'})
                except Exception as db_err:
                    logger.error(f"Failed to update error status for {model_id}: {db_err}")
            
            return {
                'model_id': model_id,
                'status': 'error',
                'error':  str(e)
            }

    # =====================================================================
            #Funzioni non utilizzate nel flusso principale
    # =====================================================================

    # Potenzialmente utile per la realizzazione di una box-view della genealogia di un modello (andrà cambiata)
    def get_family_genealogy(self, family_id: str) -> Dict[str, Any]:
        """
        Get complete genealogy information for a family.
        
        Args:
            family_id: ID of the family
            
        Returns:
            Dictionary with genealogy data
        """
        try:
            # Get family info
            family = neo4j_service.get_family_by_id(family_id)
            if not family:
                return {'error': 'Family not found'}
            
            # Get family models
            family_models = neo4j_service.get_family_models(
                family_id=family_id,
                status='ok')
            
            if not family_models:
                return {
                    'family': family.to_dict(),
                    'models': [],
                    'tree': {'nodes': [], 'edges': []},
                    'statistics': {}
                }
            
            # Build current tree
            tree, confidence_scores = self.tree_builder.build_family_tree(family_id, family_models)
            
            # Convert tree to serializable format
            tree_data = {
                'nodes': [
                    {
                        'id': node,
                        'confidence': confidence_scores.get(node, 0.0)
                    }
                    for node in tree.nodes()
                ],
                'edges': [
                    {
                        'source': source,
                        'target': target,
                        'data': tree.get_edge_data(source, target) or {}
                    }
                    for source, target in tree.edges()
                ]
            }
            
            # Get tree statistics
            tree_stats = self.tree_builder.get_tree_statistics(tree)
            
            # Validate tree
            is_valid, issues = self.tree_builder.validate_tree(tree)
            
            return {
                'family': family.to_dict(),
                'models': [model.to_dict() for model in family_models],
                'tree': tree_data,
                'statistics': tree_stats,
                'validation': {
                    'is_valid': is_valid,
                    'issues': issues
                },
                'confidence_scores': confidence_scores
            }
            
        except Exception as e:
            logger.error(f"Error getting family genealogy for {family_id}: {e}")
            return {'error': str(e)}
    
    # Potenzialmente utile per la realizzazione di una box-view della genealogia di un modello (andrà cambiata)
    def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """
        Get complete lineage information for a specific model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dictionary with lineage data
        """
        try:
            # Get model
            model = neo4j_service.get_model_by_id(model_id)
            if not model:
                return {'error': 'Model not found'}
            
            # Get family genealogy
            if model.family_id:
                family_genealogy = self.get_family_genealogy(model.family_id)
                
                if 'error' not in family_genealogy:
                    # Extract model-specific lineage from family tree
                    tree_data = family_genealogy['tree']
                    
                    # Find ancestors
                    ancestors = []
                    current_id = model_id
                    visited = set()
                    
                    while current_id and current_id not in visited:
                        visited.add(current_id)
                        
                        # Find parent in tree
                        parent_edges = [
                            edge for edge in tree_data['edges'] 
                            if edge['target'] == current_id
                        ]
                        
                        if parent_edges:
                            parent_id = parent_edges[0]['source']
                            parent_model = neo4j_service.get_model_by_id(parent_id)
                            if parent_model:
                                ancestors.append(parent_model.to_dict())
                            current_id = parent_id
                        else:
                            break
                    
                    # Find descendants
                    descendants = []
                    
                    def find_children(node_id):
                        child_edges = [
                            edge for edge in tree_data['edges']
                            if edge['source'] == node_id
                        ]
                        children = []
                        for edge in child_edges:
                            child_id = edge['target']
                            child_model = neo4j_service.get_model_by_id(child_id)
                            if child_model:
                                child_dict = child_model.to_dict()
                                child_dict['children'] = find_children(child_id)
                                children.append(child_dict)
                        return children
                    
                    descendants = find_children(model_id)
                    
                    return {
                        'model': model.to_dict(),
                        'family': family_genealogy['family'],
                        'ancestors': ancestors,
                        'descendants': descendants,
                        'confidence_score': family_genealogy['confidence_scores'].get(model_id, 0.0)
                    }
                else:
                    return {
                        'model': model.to_dict(),
                        'family': None,
                        'ancestors': [],
                        'descendants': [],
                        'error': family_genealogy.get('error')
                    }
            else:
                return {
                    'model': model.to_dict(),
                    'family': None,
                    'ancestors': [],
                    'descendants': [],
                    'confidence_score': 0.0
                }
                
        except Exception as e:
            logger.error(f"Error getting model lineage for {model_id}: {e}")
            return {'error': str(e)}
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the clustering system.
        
        Returns:
            Dictionary with system statistics
        """
        try:
            # Basic counts
            total_models = neo4j_service.get_all_models()
            total_models_size = total_models.count()

            processed_models = neo4j_service.filtered_models(total_models, status='ok').count()
            processing_models = neo4j_service.filtered_models(total_models, status='processing').count()
            error_models = neo4j_service.filtered_models(total_models, status='error').count()

            total_families = neo4j_service.get_all_families().count()
            
            # Family statistics
            family_sizes = []
            families = neo4j_service.get_all_families()
            
            for family in families:
                family_model_count = neo4j_service.get_family_models(
                    family_id=family.id,
                    status='ok'
                ).count()
                family_sizes.append(family_model_count)
            
            # Calculate averages
            avg_family_size = np.mean(family_sizes) if family_sizes else 0.0
            max_family_size = max(family_sizes) if family_sizes else 0
            min_family_size = min(family_sizes) if family_sizes else 0
            
            return {
                'models': {
                    'total': total_models_size,
                    'processed': processed_models,
                    'processing': processing_models,
                    'error': error_models
                },
                'families': {
                    'total': total_families,
                    'avg_size': float(avg_family_size),
                    'max_size': max_family_size,
                    'min_size': min_family_size,
                    'size_distribution': family_sizes
                },
                'clustering': {
                    #'distance_metric': str(self.distance_calculator.default_metric),
                    'family_threshold': self.family_clustering.family_threshold,
                    'clustering_method': str(self.family_clustering.clustering_method),
                    'tree_method': str(self.tree_builder.method),
                    'lambda_param': self.tree_builder.lambda_param
                }
            }
            
        except Exception as e:
            return logHandler.error_handler(f"Error getting system statistics: {e}", "get_system_statistics")