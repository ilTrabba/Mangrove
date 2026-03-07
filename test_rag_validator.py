# test_validator.py
import os
import sys

sys.path.insert(0, os.path.join(os. path.dirname(__file__), "model_heritage_backend"))

from model_heritage_backend.src.utils.cypher_validator import cypher_validator, ValidationResult

print("=" * 60)
print("CYPHER VALIDATOR TESTS")
print("=" * 60)

test_cases = [
    # Valid queries
    ("MATCH (n: Model) RETURN n", True, "Valid - simple match"),
    ("MATCH (a)-[r]->(b) RETURN a, r, b", True, "Valid - with relationship"),
    ("MATCH path = (a)-[*]->(b) RETURN path", True, "Valid - path variable"),
    ("OPTIONAL MATCH (n: Model) RETURN n", True, "Valid - optional match"),
    
    # Write operations (should fail)
    ("CREATE (n:Model {name: 'test'}) RETURN n", False, "Invalid - CREATE"),
    ("MATCH (n) DELETE n", False, "Invalid - DELETE"),
    ("MATCH (n) SET n.name = 'test' RETURN n", False, "Invalid - SET"),
    ("MATCH (n) DETACH DELETE n", False, "Invalid - DETACH DELETE"),
    ("MERGE (n:Model {id: '1'}) RETURN n", False, "Invalid - MERGE"),
    
    # Scalar-only returns (should fail)
    ("MATCH (n) RETURN count(n)", False, "Invalid - count only"),
    ("MATCH (n) RETURN avg(n. value)", False, "Invalid - avg only"),
    
    # Valid with aggregation + nodes
    ("MATCH (n) WITH n, count(*) as c RETURN n", True, "Valid - aggregation with node"),
    
    # Dangerous patterns
    ("CALL dbms.shutdown()", False, "Invalid - dbms call"),
    ("LOAD CSV FROM 'file' AS row RETURN row", False, "Invalid - LOAD CSV"),
    
    # Malformed
    ("", False, "Invalid - empty"),
    ("SELECT * FROM models", False, "Invalid - SQL syntax"),
]

for query, expected_valid, description in test_cases: 
    is_valid, result, error = cypher_validator.validate(query)
    status = "✅" if is_valid == expected_valid else "❌"
    print(f"\n{status} {description}")
    print(f"   Query: {query[: 50]}...")
    print(f"   Valid: {is_valid}, Result: {result. value}")
    if error: 
        print(f"   Error: {error[: 60]}...")