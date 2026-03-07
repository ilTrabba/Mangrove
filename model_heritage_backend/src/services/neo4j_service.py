import logging

from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
from src.db_entities.entity import Model
from src.log_handler import logHandler
from datetime import datetime, timezone
from ..config import Config

logger = logging.getLogger(__name__)

class Neo4jService:
    """Service class for Neo4j graph operations"""
    
    def __init__(self):
        self.driver = None
        self.connect()
    
    def connect(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                Config.NEO4J_URI,
                auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
            )
            # Test connection
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j database successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def is_connected(self) -> bool:
        """Check if Neo4j is connected and available"""
        if not self.driver:
            return False
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                session.run("RETURN 1")
            return True
        except Exception:
            return False
    
    def create_constraints(self):
        """Create Neo4j constraints and indexes"""
        if not self.driver:
            return False
        
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Model) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Family) REQUIRE f.id IS UNIQUE", 
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:FamilyCentroid) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Centroid) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Model) REQUIRE m.checksum IS UNIQUE"
        ]
        
        indexes = [
            "CREATE INDEX model_family_idx IF NOT EXISTS FOR (m:Model) ON (m.family_id)",
            "CREATE INDEX model_status_idx IF NOT EXISTS FOR (m:Model) ON (m.status)",
            "CREATE INDEX family_pattern_idx IF NOT EXISTS FOR (f:Family) ON (f.structural_pattern_hash)",
            "CREATE INDEX centroid_family_idx IF NOT EXISTS FOR (c:Centroid) ON (c.family_id)",
            "CREATE INDEX dataset_verification_idx IF NOT EXISTS FOR (m:Model) ON (m.dataset_url_verified)"
        ]
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                for constraint in constraints:
                    session.run(constraint)
                for index in indexes:
                    session.run(index)
            logger.info("Neo4j constraints and indexes created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create Neo4j constraints and indexes: {e}")
            return False
    
    ############### MODEL QUERY ############### 
    def create_model(self, model_data: Dict[str, Any]) -> bool:
        """Create a Model node using MERGE (unified method)"""
        if not self.driver:
            logger.error("Neo4j driver not initialized")
            return False
        
        model_id = model_data.get('id', 'UNKNOWN')
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MERGE (m:Model {id: $id})
                SET m.name = $name,
                    m.description = $description,
                    m.file_path = $file_path,
                    m.checksum = $checksum,
                    m.weights_size_MB = $weights_size_MB,
                    m.total_parameters = $total_parameters,
                    m.layer_count = $layer_count,
                    m.structural_hash = $structural_hash,
                    m.status = $status,
                    m.family_id = $family_id,
                    m.parent_id = $parent_id,
                    m.confidence_score = $confidence_score,
                    m.created_at = $created_at,
                    m.distance_from_parent = $distance_from_parent,
                    m.kurtosis = $kurtosis,
                    m.license = $license,
                    m.task = $task,
                    m.dataset_url = $dataset_url,
                    m.dataset_url_verified = $dataset_url_verified,
                    m.readme_uri = $readme_uri,
                    m.is_foundation_model = $is_foundation_model
                RETURN m.id as model_id, labels(m) as labels, properties(m) as props
                """
                
                # Calculate file size in MB (rough estimate)
                weights_size_mb = (model_data.get('total_parameters', 0) * 4) / (1024 * 1024)  # float32 assumption
                
                # Handle task as a list or convert from comma-separated string
                task_value = model_data.get('task')
                if isinstance(task_value, str) and task_value:
                    task_value = [t.strip() for t in task_value.split(',')]
                elif not task_value:
                    task_value = []
                
                params = {
                    'id': model_data['id'],
                    'name': model_data.get('name', ''),
                    'description': model_data.get('description', ''),
                    'file_path': model_data.get('file_path', ''),
                    'checksum': model_data.get('checksum', ''),
                    'weights_size_MB': weights_size_mb,
                    'total_parameters': model_data.get('total_parameters', 0),
                    'layer_count': model_data.get('layer_count', 0),
                    'structural_hash': model_data.get('structural_hash', ''),
                    'status': model_data.get('status', 'processing'),
                    'family_id': model_data.get('family_id'),
                    'parent_id': model_data.get('parent_id'),
                    'confidence_score': model_data.get('confidence_score', 0.0),
                    'created_at': model_data.get('created_at', ''),
                    'distance_from_parent': model_data.get('distance_from_parent'),
                    'kurtosis': model_data.get('kurtosis'),
                    'license': model_data.get('license'),
                    'task': task_value,
                    'dataset_url': model_data.get('dataset_url'),
                    'dataset_url_verified': model_data.get('dataset_url_verified'),
                    'readme_uri': model_data.get('readme_uri'),
                    'is_foundation_model': model_data.get('is_foundation_model', False)
                }
                
                logger.info(f"Attempting to insert Model node with id: {model_id}")
                logger.debug(f"Parameters: {params}")
                
                result = session.run(query, params)
                record = result.single()
                
                if record:
                    logger.info(
                        f"✅ Model node inserted successfully: "
                        f"id={record['model_id']}, "
                        f"labels={record['labels']}, "
                        f"status={record['props'].get('status')}"
                    )
                    return True
                else:
                    logger.error(f"❌ Model node insert returned no record for id: {model_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Failed to insert model node (id={model_id}): {e}", exc_info=True)
            return False
    
    def update_model(self, model_id: str, updates: Dict[str, Any]) -> bool:
        """Update specific fields of a Model node"""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                # Build dynamic SET clause
                set_clauses = []
                params = {'id': model_id}
                
                for key, value in updates.items():
                    set_clauses.append(f"m.{key} = ${key}")
                    params[key] = value
                
                if not set_clauses:
                    return True  # Nothing to update
                
                query = f"""
                MATCH (m:Model {{id: $id}})
                SET {', '.join(set_clauses)}
                RETURN m
                """
                
                result = session.run(query, params)
                return result.single() is not None
                
        except Exception as e:
            logHandler.error_handler(e, "update_model", f"Failed to update model {model_id}: {e}", "update_model")
            return False
    
    def get_all_models(self, search: str = None) -> List[Model]:
        """Get all models with optional search"""
        if not self.driver:
            return []
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                if search:
                    query = """
                    MATCH (m:Model)
                    WHERE toLower(m.name) CONTAINS toLower($search)
                    RETURN m
                    ORDER BY m.name
                    """
                    result = session.run(query, {'search': search})
                else:
                    query = """
                    MATCH (m:Model)
                    RETURN m
                    ORDER BY m.name
                    """
                    result = session.run(query)
                
                models = []
                for record in result:
                    models.append(Model(**record['m']))
                
                return models
                
        except Exception as e:
            logHandler.error_handler(e, "get_all_models")
            return []
    
    def get_model_by_id(self, model_id: str) -> Optional[Model]:
        """Get a single model by ID"""
        if not self.driver:
            return None
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (m:Model {id: $id})
                RETURN m
                """
                result = session.run(query, {'id': model_id})
                record = result.single()
                
                if record:
                    return Model(**record['m'])
                return None
                
        except Exception as e:
            logHandler.error_handler(f"Failed to get model {model_id}: {e}", "get_model_by_id")
            return None
    
    def get_model_by_checksum(self, checksum: str) -> Optional[Dict[str, Any]]:
        """Get a model by checksum (for duplicate detection)"""
        if not self.driver:
            return None
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (m:Model {checksum: $checksum})
                RETURN m
                """
                result = session.run(query, {'checksum': checksum})
                record = result.single()
                
                if record:
                    return dict(record['m'])
                return None
                
        except Exception as e:
            logger.error(f"Failed to get model by checksum {checksum}: {e}")
            return None
    
    def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """Get complete lineage (ancestors and descendants) for a model"""
        if not self.driver:
            return {'parent': None, 'children': []}
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                # Get parent
                parent_query = """
                MATCH (child:Model {id: $id})-[:IS_CHILD_OF]->(parent:Model)
                RETURN parent
                """
                parent_result = session.run(parent_query, {'id': model_id})
                parent_record = parent_result.single()
                parent = dict(parent_record['parent']) if parent_record else None
                
                # Get children
                children_query = """
                MATCH (parent:Model {id: $id})<-[:IS_CHILD_OF]-(child:Model)
                RETURN child
                ORDER BY child.name
                """
                children_result = session.run(children_query, {'id': model_id})
                children = [dict(record['child']) for record in children_result]
                
                return {
                    'parent': parent,
                    'children': children
                }
                
        except Exception as e:
            logger.error(f"Failed to get lineage for model {model_id}: {e}")
            return {'parent': None, 'children': []}

    # tornare una lista di modelli (da cambiare)
    def filtered_models(self, models: List[Dict[str, Any]], status: str) -> List[Dict[str, Any]]:
        """Filter models by status"""
        return [model for model in models if model.get('status') == status]

    ############### FAMILY QUERY ############### 
    def create_family(self, family_data: Dict[str, Any]) -> bool:
        """
        Create a new family node in Neo4j.
        """
        if not self.driver:
            return False
        
        try:
            with self.driver. session(database=Config.NEO4J_DATABASE) as session:
                query = """
                CREATE (f:Family {
                    id: $id,
                    created_at: $created_at,
                    updated_at: $updated_at,
                    member_count: $member_count,
                    avg_intra_distance: $avg_intra_distance,
                    has_foundation_model: $has_foundation_model,
                    display_name: $id
                })
                RETURN f
                """
                
                session.run(query, {
                    'id': family_data['id'],
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'updated_at':  datetime.now(timezone.utc).isoformat(),
                    'member_count': family_data.get('member_count', 0),
                    'avg_intra_distance': family_data.get('avg_intra_distance', 0.0), 
                    'has_foundation_model': family_data.get('has_foundation_model', False),
                    'display_name': family_data['id']
                })
                
            logger.info(f"✅ Created family {family_data['id']}")
            return True
            
        except Exception as e:
            logHandler.error_handler(f"Failed to create family node: {e}","update_family")
            return False

    def update_family(self, family_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update an existing family node in Neo4j.
        Returns:
            bool: True if update successful, False otherwise
        """
        if not self. driver:
            return False
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                # Build SET clause dynamically based on provided fields
                set_clauses = []
                params = {'id': family_id}
                
                # Always update updated_at
                set_clauses.append("f.updated_at = $updated_at")
                params['updated_at'] = update_data.get('updated_at', datetime.now(timezone.utc))
                
                # Optional fields
                if 'name' in update_data:
                    set_clauses.append("f.name = $name")
                    params['name'] = update_data['name']
                
                if 'member_count' in update_data:
                    set_clauses.append("f.member_count = $member_count")
                    params['member_count'] = update_data['member_count']
                
                if 'avg_intra_distance' in update_data:
                    set_clauses.append("f.avg_intra_distance = $avg_intra_distance")
                    params['avg_intra_distance'] = update_data['avg_intra_distance']
                
                if 'has_foundation_model' in update_data:
                    set_clauses.append("f. has_foundation_model = $has_foundation_model")
                    params['has_foundation_model'] = update_data['has_foundation_model']
                
                query = f"""
                MATCH (f:Family {{id: $id}})
                SET {', '.join(set_clauses)}
                RETURN f
                """
                
                result = session.run(query, params)
                
                if result.single():
                    logger.info(f"✅ Updated family {family_id}")
                    return True
                else:
                    logHandler.warning_handler(f"Family {family_id} not found for update","update_family")
                    return False
                    
        except Exception as e:
            logHandler.error_handler(f"Failed to update family node: {e}","update_family")
            return False
     
    def get_direct_relationship_distances(self, best_family_id: str) -> List[float]:
        
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (model:Model {family_id: $family_id})
                WHERE model.distance_from_parent > 0
                RETURN model.distance_from_parent AS distance
                """

                result = session.run(query, family_id=best_family_id)
                distances = [record["distance"] for record in result]
            
            return distances

        except Exception as e:
            logHandler.error_handler(e, "get_direct_relationship_distances" ,f"Failed to get the distances from the family: {best_family_id} in neo4j!")
            return False

    def create_centroid_with_metadata(self, family_id: str, structural_hash: Any) -> bool:
        """Create a Centroid node with enhanced metadata according to requirements"""
        if not self.driver:
            return False
        
        try:
            
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MERGE (c:Centroid {id: $id})
                SET c.family_id = $family_id,
                    c.file_path = $file_path,
                    c.structural_hash = $structural_hash,
                    c.model_count = $model_count,
                    c.updated_at = $updated_at,
                    c.display_name = $id
                RETURN c
                """
                
                centroid_id = f"centroid_{family_id}"
                centroid_path = f"weights/centroids/{family_id}.safetensors"
                
                session.run(query, {
                    'id': centroid_id,
                    'family_id': family_id,
                    'file_path': centroid_path,
                    'structural_hash': structural_hash,
                    'model_count': 1,  # Will be updated when centroid is calculated
                    'updated_at': datetime.now(timezone.utc).isoformat(),
                    'display_name': centroid_id
                })
                
            logger.info(f"Created Centroid node for family {family_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create centroid node for family {family_id}: {e}")
            return False

    def update_centroid_metadata(self, family_id: str, model_count: Optional[int] = None):
        """Update Centroid node metadata with enhanced attributes"""
        try:

            centroid = neo4j_service.get_centroid_by_family_id(family_id)
            
            if model_count is None:
                model_count = centroid.get('model_count', 1) + 1

            # Update the Centroid node with metadata
            with neo4j_service.driver.session(database='neo4j') as session:
                query = """
                MATCH (c:Centroid {family_id: $family_id})
                SET c.model_count = $model_count,
                    c.updated_at = $updated_at
                RETURN c
                """
                
                session.run(query, {
                    'family_id': family_id,
                    'model_count': model_count,
                    'updated_at': datetime.now(timezone.utc).isoformat(), 
                })
                
            logger.info(f"✅ Updated Centroid metadata for family {family_id}, with {model_count} models")
            
        except Exception as e:
            logHandler.error_handler(f"Failed to update centroid metadata for family {family_id}: {e}","update_centroid_metadata")
    
    def delete_family_relationships(self, family_id: str) -> bool:
        """Delete all IS_CHILD_OF relationships for models in a family (batch operation)"""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (m:Model {family_id: $family_id})-[r:IS_CHILD_OF]->()
                DELETE r
                """
                session.run(query, {'family_id': family_id})
                return True
        except Exception as e:
            logHandler.error_handler(f"Failed to delete family relationships: {e}", "delete_family_relationships")
            return False

    def atomic_rebuild_genealogy(self, family_id: str, family_tree, tree_confidence: Dict[str, float]) -> bool:
        """
        Ultra-optimized single-query tree rebuild + distance update. 
        """
        if not self.driver:
            return False
        
        # Handle edge case: no edges
        if family_tree.number_of_edges() == 0:
            try:
                with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                    query = """
                    MATCH (m:Model {family_id: $family_id})-[r:IS_CHILD_OF]->()
                    DELETE r
                    """
                    session.run(query, {'family_id': family_id})
                    return True
            except Exception as e:
                logHandler.error_handler(f"Failed to clear relationships: {e}", "rebuild_family_tree_ultra")
                return False
        
        # Extract edges WITH distance in ONE iteration
        edges_data = [
                {
                    'parent': parent_id,
                    'child': child_id,
                    'confidence': tree_confidence.get(parent_id, 0.0),
                    'distance': family_tree[parent_id][child_id].get('distance', 0.0)  # ← AGGIUNTO
                }
                for parent_id, child_id in family_tree.edges()
            ]

        logger.info(f"🔄 Processing family_id: {family_id}")
        logger.info(f"📊 Total edges to process: {len(edges_data)}")
        for i, edge in enumerate(edges_data):
            logger.debug(f"  Edge {i}: parent={edge['parent']}, child={edge['child']}, "
                        f"confidence={edge['confidence']}, distance={edge['distance']}")  # ← LOG distanza

        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                
                # Main atomic query WITH distance update
                query = """
                // Step 1: Delete all existing IS_CHILD_OF relationships
                MATCH (m:Model {family_id: $family_id})-[r:IS_CHILD_OF]->()
                DELETE r
                
                WITH count(r) as deleted_count
                
                // Step 2: Create relationships AND update distances
                UNWIND $edges AS edge
                MATCH (child:Model {id: edge.child, family_id: $family_id})
                MATCH (parent:Model {id: edge.parent, family_id: $family_id})
                
                // Create relationship
                CREATE (child)-[:IS_CHILD_OF {
                    confidence: edge.confidence,
                    updated_at: datetime()
                }]->(parent)
                
                // Update child node with distance from parent
                SET child.distance_from_parent = edge.distance
                
                RETURN deleted_count, count(*) as created_count
                """
                
                result = session.run(query, {
                    'family_id': family_id,
                    'edges': edges_data
                })
                
                record = result.single()
                if record:
                    deleted = record['deleted_count']
                    created = record['created_count']
                    
                    logger.info(f"✅ Family {family_id}: {deleted} relationships deleted, "
                            f"{created} relationships created in ONE atomic query")
                    
                    if created != len(edges_data):
                        logger.error(f"🚨 MISMATCH: Expected to create {len(edges_data)} relationships, but only {created} were created!")
                        logger.error(f"🚨 Missing: {len(edges_data) - created} relationships were NOT created")
                else:
                    logger.error(f"❌ No result returned from query for family {family_id}")
                
                # Verification after
                verify_after_query = """
                MATCH (child:Model {family_id: $family_id})-[r:IS_CHILD_OF]->(parent:Model)
                RETURN count(r) as total_relationships
                """
                after_result = session.run(verify_after_query, {'family_id': family_id})
                after_record = after_result.single()
                if after_record:
                    logger.info(f"📊 Total IS_CHILD_OF relationships in DB for family {family_id}: {after_record['total_relationships']}")
                
                return True
                
        except Exception as e:
            logHandler.error_handler(f"Ultra rebuild failed for family {family_id}: {e}", "rebuild_family_tree_ultra")
            return False
    
    # funzione potenzialmente utile per la gestione dei centroidi
    def create_or_update_family_centroid(self, family_id: str, centroid_embedding: Optional[List[float]] = None) -> bool:
        """Create or update a FamilyCentroid node with actual centroid data"""
        if not self.driver:
            return False
        
        try:
            # Use provided embedding or placeholder
            embedding = centroid_embedding if centroid_embedding is not None else [0.0]
            
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MERGE (c:FamilyCentroid {id: $id})
                SET c.family_id = $family_id,
                    c.embedding = $embedding,
                    c.color = 'white',
                    c.updated_at = datetime()
                RETURN c
                """
                
                centroid_id = f"centroid_{family_id}"
                session.run(query, {
                    'id': centroid_id,
                    'family_id': family_id,
                    'embedding': embedding
                })
                
            logger.info(f"Updated Neo4j centroid for family {family_id} with embedding size {len(embedding)}")
            return True
        except Exception as e:
            logger.error(f"Failed to create/update family centroid: {e}")
            return False
    
    def get_all_families(self) -> List[Dict[str, Any]]:
        """Get all families"""
        if not self.driver:
            return []
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (f:Family)
                RETURN f
                ORDER BY f.created_at DESC
                """
                result = session.run(query)
                
                families = []
                for record in result:
                    family_props = dict(record['f'])
                    families.append(family_props)
                
                return families
                
        except Exception as e:
            logger.error(f"Failed to get all families: {e}")
            return []
    
    def get_all_centroids(self, structural_hash: Any) -> List[Dict[str, Any]]:
        """Get all centroids"""
        if not self.driver:
            return []
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (c:Centroid)
                WHERE c.structural_hash = $structural_hash
                RETURN c
                """
                result = session.run(query, {'structural_hash': structural_hash})
                
                centroids = []
                for record in result:
                    centroid_props = dict(record['c'])
                    centroids.append(centroid_props)
                
                return centroids
                
        except Exception as e:
            logger.error(f"Failed to get all centroids: {e}")
            return []
    
    # Get all centroids of the families that do not possess a foundation model yet
    def get_all_centroids_without_foundation(self, structural_hash: Any) -> List[Dict[str, Any]]:
        """Get centroids from families without foundation models"""
        if not self.driver:
            return []
        
        try:
            with self.driver. session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (f:Family)-[:HAS_CENTROID]->(c:Centroid)
                WHERE f.has_foundation_model = false
                AND c.structural_hash = $structural_hash
                RETURN c
                ORDER BY c.created_at DESC
                """
                result = session.run(query, {'structural_hash': structural_hash})
                centroids = [dict(record['c']) for record in result]
                return centroids
                
        except Exception as e:
            logHandler.error_handler(f"Failed to get centroids without foundation models: {e}","get_all_centroids_without_foundation")
            return []
    
    def get_centroid_by_family_id(self, family_id: str) -> Optional[Dict]:
        """Get a single centroid from Neo4j by its family's ID."""
        if not self.driver:
            return None
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (f:Family {id: $family_id})-[:HAS_CENTROID]->(c:Centroid)
                RETURN c
                """
                result = session.run(query, {'family_id': family_id})
                record = result.single()
                
                if record:
                    return dict(record['c'])
                return None
                
        except Exception as e:
            logHandler.error_handler(f"Failed to get centroid of: {family_id}: {e}", "get_centroid_by_family_id")
            return None
        return None

    def get_family_by_id(self, family_id: str) -> Optional[Dict]:
        """Get a single family from Neo4j by its ID."""
        if not self.driver:
            return None
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (f:Family {id: $family_id})
                RETURN f
                """
                result = session.run(query, {'family_id': family_id})
                record = result.single()
                
                if record:
                    return dict(record['f'])
                return None
                
        except Exception as e:
            logHandler.error(f"Failed to get family {family_id}: {e}", "get_family_by_id")
            return None

    def get_family_models(self, family_id: str, status: Optional[str] = None) -> List[Model]:
        """Get all models in a specific family (new version)
        
        Args:
            family_id: The family ID to filter by
            status: Optional status filter ('ok', 'failed', etc.)
        """
        if not self.driver:
            return []

        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                # Query dinamica in base alla presenza di status
                if status:
                    query = """
                    MATCH (n)-[:BELONGS_TO]->(f:Family {id: $family_id})
                    WHERE n.status = $status
                    RETURN n
                    """
                    result = session.run(query, {'family_id': family_id, 'status': status})
                else:
                    query = """
                    MATCH (n)-[:BELONGS_TO]->(f:Family {id: $family_id})
                    RETURN n
                    """
                    result = session.run(query, {'family_id': family_id})

                models = []
                for record in result:
                    model = Model(**record['n'])
                    models.append(model)

                return models
        except Exception as e:
            logHandler.error_handler(f"Failed to get models for family {family_id}: {e}", "get_family_models_new")
            return []

    def get_family_root(self, family_id: str) -> Optional[Dict[str, Any]]:
        """Get the root model of a family (model without parent)"""
        if not self.driver:
            return None
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (m:Model)-[:BELONGS_TO]->(f:Family {id: $family_id})
                WHERE NOT (m)-[:IS_CHILD_OF]->(:Model)
                RETURN m
                LIMIT 1
                """
                result = session.run(query, {'family_id': family_id})
                record = result.single()
                
                if record:
                    return dict(record['m'])
                return None
                
        except Exception as e:
            logHandler.error_handler(f"Failed to get root model for family {family_id}: {e}", "get_family_root")
            return None

    def get_stats(self) -> Dict[str, int]:
        """Get system statistics"""
        if not self.driver:
            return {'total_models': 0, 'total_families': 0, 'processing_models': 0}
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                stats_query = """
                MATCH (m:Model)
                OPTIONAL MATCH (f:Family)
                RETURN 
                    count(DISTINCT m) as total_models,
                    count(DISTINCT f) as total_families,
                    count(DISTINCT CASE WHEN m.status = 'processing' THEN m END) as processing_models
                """
                result = session.run(stats_query)
                record = result.single()
                
                if record:
                    return {
                        'total_models': record['total_models'] or 0,
                        'total_families': record['total_families'] or 0,
                        'processing_models': record['processing_models'] or 0
                    }
                
                return {'total_models': 0, 'total_families': 0, 'processing_models': 0}
                
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'total_models': 0, 'total_families': 0, 'processing_models': 0}
    
    def create_belongs_to_relationship(self, model_id: str, family_id: str) -> bool:
        """Create BELONGS_TO relationship between Model and Family"""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (m:Model {id: $model_id})
                MATCH (f:Family {id: $family_id})
                MERGE (m)-[:BELONGS_TO]->(f)
                """
                
                session.run(query, {
                    'model_id': model_id,
                    'family_id': family_id
                })
                
            return True
        except Exception as e:
            logger.error(f"Failed to create BELONGS_TO relationship: {e}")
            return False
    
    def create_has_centroid_relationship(self, family_id: str) -> bool:
        """Create HAS_CENTROID relationship between Family and Centroid"""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (f:Family {id: $family_id})
                MATCH (c:Centroid {family_id: $family_id})
                MERGE (f)-[:HAS_CENTROID]->(c)
                """
                
                session.run(query, {'family_id': family_id})
                
            return True
        except Exception as e:
            logger.error(f"Failed to create HAS_CENTROID relationship: {e}")
            return False
    
    def create_parent_child_relationship(self, parent_id: str, child_id: str, confidence: float = 0.0) -> bool:
        """Create IS_CHILD_OF relationship between Models"""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (parent:Model {id: $parent_id})
                MATCH (child:Model {id: $child_id})
                MERGE (child)-[:IS_CHILD_OF {confidence: $confidence}]->(parent)
                """
                
                session.run(query, {
                    'parent_id': parent_id,
                    'child_id': child_id,
                    'confidence': confidence
                })
                
            return True
        except Exception as e:
            logger.error(f"Failed to create IS_CHILD_OF relationship: {e}")
            return False
    
    # riusabile per box view
    def get_full_graph(self) -> Dict[str, Any]:
        """Get complete graph data for visualization"""
        if not self.driver:
            return {'nodes': [], 'edges': [], 'error': 'Neo4j not connected'}
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                # Get all nodes
                nodes_query = """
                MATCH (n)
                RETURN 
                    id(n) as neo_id,
                    labels(n) as labels,
                    properties(n) as props
                """
                
                # Get all relationships
                edges_query = """
                MATCH (a)-[r]->(b)
                RETURN 
                    id(a) as source_neo_id,
                    id(b) as target_neo_id,
                    type(r) as relationship_type,
                    properties(r) as props,
                    a.id as source_id,
                    b.id as target_id
                """
                
                nodes_result = session.run(nodes_query)
                edges_result = session.run(edges_query)
                
                nodes = []
                for record in nodes_result:
                    node = {
                        'id': record['props'].get('id', f"node_{record['neo_id']}"),
                        'neo_id': record['neo_id'],
                        'label': record['labels'][0] if record['labels'] else 'Unknown',
                        'color': record['props'].get('color', 'gray'),
                        **record['props']
                    }
                    nodes.append(node)
                
                edges = []
                for record in edges_result:
                    edge = {
                        'source': record['source_id'],
                        'target': record['target_id'],
                        'type': record['relationship_type'],
                        **record['props']
                    }
                    edges.append(edge)
                
                return {
                    'nodes': nodes,
                    'edges': edges,
                    'node_count': len(nodes),
                    'edge_count': len(edges)
                }
                
        except Exception as e:
            logger.error(f"Failed to get full graph: {e}")
            return {'nodes': [], 'edges': [], 'error': str(e)}
    
    def get_family_subgraph(self, family_id: str) -> Dict[str, Any]:
        """Get subgraph for a specific family"""
        if not self.driver:
            return {'nodes': [], 'edges': [], 'error': 'Neo4j not connected'}
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                # Get family, its models, and centroid
                query = """
                MATCH (f:Family {id: $family_id})
                OPTIONAL MATCH (m:Model)-[:BELONGS_TO]->(f)
                OPTIONAL MATCH (f)-[:HAS_CENTROID]->(c:FamilyCentroid)
                OPTIONAL MATCH (m1:Model)-[r:IS_CHILD_OF]->(m2:Model)
                WHERE m1.family_id = $family_id AND m2.family_id = $family_id
                
                WITH f, collect(DISTINCT m) as models, c, collect(DISTINCT {source: m1, target: m2, type: type(r), props: properties(r)}) as relationships
                
                RETURN f, models, c, relationships
                """
                
                result = session.run(query, {'family_id': family_id})
                record = result.single()
                
                if not record:
                    return {'nodes': [], 'edges': [], 'error': 'Family not found'}
                
                nodes = []
                edges = []
                
                # Add family node
                family = record['f']
                if family:
                    nodes.append({
                        'id': family['id'],
                        'label': 'Family',
                        'color': 'black',
                        **dict(family)
                    })
                
                # Add centroid node
                centroid = record['c']
                if centroid:
                    nodes.append({
                        'id': centroid['id'],
                        'label': 'FamilyCentroid',
                        'color': 'white',
                        **dict(centroid)
                    })
                    
                    # Add family -> centroid edge
                    edges.append({
                        'source': family_id,
                        'target': centroid['id'],
                        'type': 'HAS_CENTROID'
                    })
                
                # Add model nodes and BELONGS_TO edges
                for model in record['models'] or []:
                    if model:
                        nodes.append({
                            'id': model['id'],
                            'label': 'Model',
                            **dict(model)
                        })
                        
                        # Add model -> family edge
                        edges.append({
                            'source': model['id'],
                            'target': family_id,
                            'type': 'BELONGS_TO'
                        })
                
                # Add parent-child relationships
                for rel in record['relationships'] or []:
                    if rel['source'] and rel['target']:
                        edges.append({
                            'source': rel['source']['id'],
                            'target': rel['target']['id'],
                            'type': rel['type'],
                            **rel['props']
                        })
                
                return {
                    'family_id': family_id,
                    'nodes': nodes,
                    'edges': edges,
                    'node_count': len(nodes),
                    'edge_count': len(edges)
                }
                
        except Exception as e:
            logger.error(f"Failed to get family subgraph: {e}")
            return {'nodes': [], 'edges': [], 'error': str(e)}
        
    def get_family_leaves(self, family_id: str) -> List[Dict[str, Any]]:
        """Get all leaf models in a family (models without children)"""
        if not neo4j_service.driver:
            return []
        
        try:
            with neo4j_service.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (m:Model)-[:BELONGS_TO]->(f:Family {id: $family_id})
                WHERE ((m)-[:IS_CHILD_OF]->(:Model)) AND NOT ((:Model)-[:IS_CHILD_OF]->(m))
                RETURN m
                """
                result = session.run(query, {'family_id': family_id})
                leaves = [dict(record['m']) for record in result]
                return leaves
                
        except Exception as e:
            logHandler.error_handler(f"Failed to get leaf models for family {family_id}: {e}", "get_family_leaves")
            return []

# Global instance
neo4j_service = Neo4jService()