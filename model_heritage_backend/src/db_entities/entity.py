"""
Model and Family proxy classes for Neo4j-only architecture.
These classes provide the same interface as the previous SQLAlchemy models
but use Neo4j as the backend through neo4j_service.
"""
import logging

logger = logging.getLogger(__name__)

class Model:
    """Proxy class for Model operations using Neo4j backend"""
    
    def __init__(self, **kwargs):
        """Initialize model proxy with data"""
        self.data = kwargs
        
    def __getattr__(self, name):
        """Allow attribute access to model properties"""
        return self.data.get(name)
    
    @property
    def id(self):
        return self.data.get('id')
    
    @property
    def name(self):
        return self.data.get('name')
    
    @property
    def family_id(self):
        return self.data.get('family_id')
    
    @property
    def parent_id(self):
        return self.data.get('parent_id')
    
    @property
    def confidence_score(self):
        return self.data.get('confidence_score')
    
    @property
    def status(self):
        return self.data.get('status')
    
    @property
    def checksum(self):
        return self.data.get('checksum')
    
    @property
    def total_parameters(self):
        return self.data.get('total_parameters')
    
    @property
    def layer_count(self):
        return self.data.get('layer_count')
    
    @property
    def structural_hash(self):
        return self.data.get('structural_hash')
    
    @property 
    def file_path(self):
        return self.data.get('file_path')
    
    @property
    def description(self):
        return self.data.get('description')
    
    @property
    def created_at(self):
        return self.data.get('created_at')
    
    @property
    def license(self):
        return self.data.get('license')
    
    @property
    def task(self):
        return self.data.get('task')
    
    @property
    def dataset_url(self):
        return self.data.get('dataset_url')
    
    @property
    def dataset_url_verified(self):
        return self.data.get('dataset_url_verified')
    
    @property
    def readme_uri(self):
        return self.data.get('readme_uri')
    
    @property
    def distance_from_parent(self):
        return self.data.get('distance_from_parent')
    
    @property
    def kurtosis(self):
        return self.data.get('kurtosis')
    
    @property
    def is_foundation_model(self):
        return self.data.get('is_foundation_model', False)
    
    def to_dict(self):
        """Convert to dictionary format"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'total_parameters': self.total_parameters,
            'layer_count': self.layer_count,
            'structural_hash': self.structural_hash,
            'family_id': self.family_id,
            'parent_id': self.parent_id,
            'confidence_score': self.confidence_score,
            'distance_from_parent': self.distance_from_parent,
            'kurtosis': self.kurtosis,
            'status': self.status,
            'checksum': self.checksum,
            'file_path': self.file_path,
            'created_at': self.created_at,
            'license': self.license,
            'task': self.task,
            'dataset_url': self.dataset_url,
            'dataset_url_verified': self.dataset_url_verified,
            'readme_uri': self.readme_uri,
            'is_foundation_model': self.is_foundation_model
        }

class Family:
    """Proxy class for Family operations using Neo4j backend"""
    
    def __init__(self, **kwargs):
        """Initialize family proxy with data"""
        self.data = kwargs
    
    def __getattr__(self, name):
        """Allow attribute access to family properties"""
        return self.data.get(name)
    
    @property
    def id(self):
        return self.data.get('id')
    
    @property
    def structural_pattern_hash(self):
        return self.data.get('structural_pattern_hash')
    
    @property
    def member_count(self):
        return self.data.get('member_count')
    
    @property
    def avg_intra_distance(self):
        return self.data.get('avg_intra_distance')
    
    @property
    def created_at(self):
        return self.data.get('created_at')
    
    @property
    def updated_at(self):
        return self.data.get('updated_at')
    
    @property
    def has_foundation_model(self):
        return self.data.get('has_foundation_model',False)

    @property
    def get_max_distance_root_nodes(self):
        return self.data.get('max_distance_root_nodes')
    
    def to_dict(self):
        """Convert to dictionary format"""
        return {
            'id': self.id,
            'structural_pattern_hash': self.structural_pattern_hash,
            'member_count': self.member_count,
            'avg_intra_distance': self.avg_intra_distance,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'has_foundation_model': self.has_foundation_model
        }