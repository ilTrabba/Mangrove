# test_endpoint.py
import requests
import json

BASE_URL = "http://localhost:5001/api"

def test_nl_query(question: str):
    """Test the NL query endpoint."""
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)
    
    response = requests.post(
        f"{BASE_URL}/nl-query",
        json={"question": question},
        headers={"Content-Type": "application/json"}
    )
    
    data = response.json()
    
    print(f"Status:  {response.status_code}")
    print(f"Success: {data.get('success')}")
    
    if data.get('success'):
        print(f"Cypher:  {data. get('cypher', '')[: 80]}...")
        graph_data = data. get('data', {})
        print(f"Nodes: {graph_data.get('node_count', 0)}")
        print(f"Edges:  {graph_data. get('edge_count', 0)}")
        if data.get('message'):
            print(f"Message: {data. get('message')}")
    else:
        print(f"Error: {data.get('error')}")
        print(f"Error Type: {data.get('error_type')}")
    
    return data

def test_examples():
    """Get example queries."""
    print(f"\n{'='*60}")
    print("Getting example queries...")
    print('='*60)
    
    response = requests.get(f"{BASE_URL}/nl-query/examples")
    data = response.json()
    
    if data.get('success'):
        for ex in data.get('examples', []):
            print(f"  - {ex['question']}")
    
    return data

def test_schema():
    """Get current schema."""
    print(f"\n{'='*60}")
    print("Getting schema...")
    print('='*60)
    
    response = requests.get(f"{BASE_URL}/nl-query/schema")
    data = response.json()
    
    if data.get('success'):
        schema = data.get('schema', {})
        print(f"Nodes: {list(schema.get('nodes', {}).keys())}")
        print(f"Relationships: {schema.get('relationships', [])}")
    
    return data

if __name__ == "__main__":
    print("\n" + "="*60)
    print("NL QUERY ENDPOINT TESTS")
    print("="*60)
    print("\nMake sure the Flask server is running on port 5001!")
    print("Run: python -m flask run --port 5001")
    
    # Test schema
    test_schema()
    
    # Test examples
    test_examples()
    
    # Test queries
    test_nl_query("Mostrami tutti i foundation model")
    test_nl_query("Mostrami tutte le famiglie")
    test_nl_query("Quanti modelli ci sono?")  # Should fail - not visualizable