"""
Natural Language Query API Endpoint
"""

import logging
from datetime import datetime, date, time
from flask import Blueprint, request, jsonify
from typing import Dict, Any, List

from src.services.nl_to_cypher import nl_to_cypher_service
from src.services.neo4j_service import neo4j_service
from src.utils. cypher_validator import cypher_validator, ValidationResult
from src.config import Config

logger = logging.getLogger(__name__)

nl_query_bp = Blueprint('nl_query', __name__)


def neo4j_to_json_serializable(obj):
    """Convert Neo4j types to JSON serializable Python types."""
    if obj is None:
        return None
    
    # Handle Neo4j DateTime types
    if hasattr(obj, 'isoformat'):
        return obj. isoformat()
    
    # Handle Neo4j Date
    if hasattr(obj, 'year') and hasattr(obj, 'month') and hasattr(obj, 'day') and not hasattr(obj, 'hour'):
        return f"{obj.year:04d}-{obj.month:02d}-{obj.day:02d}"
    
    # Handle Neo4j Time
    if hasattr(obj, 'hour') and hasattr(obj, 'minute') and not hasattr(obj, 'year'):
        return f"{obj.hour:02d}:{obj.minute:02d}:{obj.second:02d}"
    
    # Handle Neo4j Duration
    if hasattr(obj, 'months') and hasattr(obj, 'days') and hasattr(obj, 'seconds'):
        return str(obj)
    
    # Handle lists
    if isinstance(obj, list):
        return [neo4j_to_json_serializable(item) for item in obj]
    
    # Handle dicts
    if isinstance(obj, dict):
        return {k: neo4j_to_json_serializable(v) for k, v in obj.items()}
    
    return obj


class GraphResultFormatter:
    """Formats Neo4j query results for graph visualization."""
    
    NODE_COLORS = {
        'Model': '#4F46E5',
        'Family': '#059669',
        'Centroid': '#D97706',
        'FamilyCentroid': '#D97706'
    }
    
    DEFAULT_COLOR = '#6B7280'
    
    @classmethod
    def format_results(cls, records: List[Dict[str, Any]]) -> Dict[str, Any]: 
        """Format Neo4j query results into nodes and edges for visualization."""
        nodes = {}
        edges = {}
        
        for record in records: 
            for key, value in record.items():
                cls._process_value(value, nodes, edges)
        
        return {
            'nodes': list(nodes.values()),
            'edges':  list(edges.values()),
            'node_count': len(nodes),
            'edge_count': len(edges)
        }
    
    @classmethod
    def _process_value(cls, value:  Any, nodes: Dict, edges: Dict):
        """Process a single value from a Neo4j record."""
        if value is None:
            return
        
        # Handle Node
        if hasattr(value, 'labels') and hasattr(value, 'items'):
            cls._add_node(value, nodes)
        
        # Handle Relationship
        elif hasattr(value, 'type') and hasattr(value, 'start_node'):
            cls._add_relationship(value, nodes, edges)
        
        # Handle Path
        elif hasattr(value, 'nodes') and hasattr(value, 'relationships'):
            for node in value.nodes:
                cls._add_node(node, nodes)
            for rel in value.relationships:
                cls._add_relationship(rel, nodes, edges)
        
        # Handle list
        elif isinstance(value, list):
            for item in value: 
                cls._process_value(item, nodes, edges)
    
    @classmethod
    def _add_node(cls, node: Any, nodes: Dict):
        """Add a node to the nodes dictionary."""
        try:
            props = dict(node.items()) if hasattr(node, 'items') else {}
            # Convert all properties to JSON serializable
            props = neo4j_to_json_serializable(props)
            
            node_id = props.get('id', str(node. element_id if hasattr(node, 'element_id') else id(node)))
            
            labels = list(node.labels) if hasattr(node, 'labels') else []
            primary_label = labels[0] if labels else 'Unknown'
            
            color = cls.NODE_COLORS.get(primary_label, cls.DEFAULT_COLOR)
            
            nodes[node_id] = {
                'id': node_id,
                'label': primary_label,
                'labels': labels,
                'color': color,
                'properties': props,
                'name': props.get('name', props.get('display_name', props.get('id', node_id))),
                'status': props.get('status'),
                'is_foundation_model': props.get('is_foundation_model', False)
            }
        except Exception as e: 
            logger.warning(f"Failed to process node: {e}")
    
    @classmethod
    def _add_relationship(cls, rel: Any, nodes:  Dict, edges: Dict):
        """Add a relationship to the edges dictionary."""
        try:
            props = dict(rel. items()) if hasattr(rel, 'items') else {}
            # Convert all properties to JSON serializable
            props = neo4j_to_json_serializable(props)
            
            start_node = rel.start_node if hasattr(rel, 'start_node') else None
            end_node = rel.end_node if hasattr(rel, 'end_node') else None
            
            if start_node and end_node:
                cls._add_node(start_node, nodes)
                cls._add_node(end_node, nodes)
                
                start_props = dict(start_node.items()) if hasattr(start_node, 'items') else {}
                end_props = dict(end_node. items()) if hasattr(end_node, 'items') else {}
                
                source_id = start_props.get('id', str(start_node.element_id if hasattr(start_node, 'element_id') else id(start_node)))
                target_id = end_props.get('id', str(end_node.element_id if hasattr(end_node, 'element_id') else id(end_node)))
                
                rel_type = rel.type if hasattr(rel, 'type') else 'UNKNOWN'
                
                edge_key = f"{source_id}-{rel_type}->{target_id}"
                
                edges[edge_key] = {
                    'id':  edge_key,
                    'source':  source_id,
                    'target':  target_id,
                    'type':  rel_type,
                    'label': rel_type,
                    'properties': props
                }
        except Exception as e: 
            logger.warning(f"Failed to process relationship: {e}")


def execute_cypher_query(query: str) -> List[Dict[str, Any]]:
    """Execute a Cypher query and return raw results."""
    if not neo4j_service. driver:
        raise Exception("Neo4j not connected")
    
    try:
        with neo4j_service.driver.session(database=Config.NEO4J_DATABASE) as session:
            result = session.run(query)
            records = []
            for record in result: 
                records.append(dict(record))
            return records
    except Exception as e: 
        logger.error(f"Failed to execute query: {e}")
        raise


@nl_query_bp.route('/nl-query', methods=['POST'])
def natural_language_query():
    """Convert natural language to Cypher and execute."""
    try: 
        data = request. get_json(force=True, silent=True)
        
        if not data: 
            return jsonify({
                'success': False,
                'error': 'Request body is required',
                'error_type': 'validation'
            }), 400
        
        question = data.get('question', '').strip()
        execute = data.get('execute', True)
        
        if not question:
            return jsonify({
                'success': False,
                'error': 'Question cannot be empty.',
                'error_type': 'validation'
            }), 400
        
        if not neo4j_service. is_connected():
            return jsonify({
                'success': False,
                'error': 'Neo4j database unavailable.',
                'error_type': 'connection'
            }), 503
        
        # Step 1: Convert NL to Cypher
        logger.info(f"Converting NL query: {question[: 50]}...")
        conversion_success, cypher_query, conversion_error = nl_to_cypher_service.convert(question)
        
        if not conversion_success:
            return jsonify({
                'success': False,
                'error': conversion_error or 'Could not convert the question.',
                'error_type': 'conversion',
                'question': question
            }), 400
        
        # Step 2: Validate the generated query
        logger.info(f"Validating Cypher query: {cypher_query[:50]}...")
        is_valid, validation_result, validation_error = cypher_validator.validate(cypher_query)
        
        if not is_valid:
            return jsonify({
                'success': False,
                'error': validation_error or 'Invalid query generated.',
                'error_type': 'validation',
                'question': question,
                'cypher':  cypher_query
            }), 400
        
        # Step 3: Execute query
        if not execute:
            return jsonify({
                'success': True,
                'question': question,
                'cypher': cypher_query,
                'data': None,
                'message': 'Query generated but not executed'
            })
        
        logger.info(f"Executing Cypher query...")
        try:
            records = execute_cypher_query(cypher_query)
        except Exception as e: 
            logger.error(f"Query execution failed: {e}")
            return jsonify({
                'success': False,
                'error': f'Query execution error: {str(e)}',
                'error_type': 'execution',
                'question': question,
                'cypher': cypher_query
            }), 500
        
        # Step 4: Format results
        graph_data = GraphResultFormatter. format_results(records)
        
        logger.info(f"Query successful: {graph_data['node_count']} nodes, {graph_data['edge_count']} edges")
        
        response_data = {
            'success': True,
            'question':  question,
            'cypher': cypher_query,
            'data': graph_data
        }
        
        if graph_data['node_count'] == 0:
            response_data['message'] = 'Query returned no results.'
        
        return jsonify(response_data)
        
    except Exception as e: 
        logger.error(f"Unexpected error in nl-query endpoint: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error.',
            'error_type': 'internal'
        }), 500


@nl_query_bp.route('/nl-query/schema', methods=['GET'])
def get_schema():
    """Get the current Neo4j schema."""
    try:
        schema = nl_to_cypher_service.get_schema()
        return jsonify({'success': True, 'schema': schema})
    except Exception as e:
        return jsonify({'success':  False, 'error': str(e)}), 500


@nl_query_bp.route('/nl-query/schema/refresh', methods=['POST'])
def refresh_schema():
    """Force refresh the schema cache."""
    try:
        schema = nl_to_cypher_service. refresh_schema()
        return jsonify({'success': True, 'schema': schema, 'message': 'Schema refreshed'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@nl_query_bp.route('/nl-query/examples', methods=['GET'])
def get_query_examples():
    """Get example queries."""
    examples = [
        {"question": "Show all foundation models", "description": "Models not derived from others"},
        {"question": "Show all families", "description": "Display all model families"},
        {"question": "Show all models with their families", "description": "Models and BELONGS_TO relationships"},
        {"question": "Show parent-child relationships", "description": "IS_CHILD_OF relationships"},
        {"question": "Show all centroids", "description":  "Centroids with HAS_CENTROID relationships"},
        {"question": "Show the longest parent-child path", "description": "Longest derivation chain"},
    ]
    return jsonify({'success': True, 'examples': examples})