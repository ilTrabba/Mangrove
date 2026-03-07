# test_rag.py
import os
import sys

# Aggiungi la cartella 'model_heritage_backend' al Python path
sys.path.insert(0, os.path.join(os.path. dirname(__file__), "model_heritage_backend"))

# Imposta la tua API key Groq (USA UNA NUOVA API KEY!)
# os.environ["RAG_API_KEY"] = ""
GROQ_API_KEY = os.environ["GROQ_API_KEY"]


from model_heritage_backend.src.services.nl_to_cypher import nl_to_cypher_service

print("=" * 50)
print("Test 1: Schema retrieval")
print("=" * 50)

schema = nl_to_cypher_service.get_schema()
print(f"Nodes: {list(schema['nodes'].keys())}")
print(f"Relationships: {schema['relationships']}")

print("\n" + "=" * 50)
print("Test 2: Query conversion - Foundation models")
print("=" * 50)

success, result, error = nl_to_cypher_service.convert(
    "Mostrami tutti i foundation model"
)
print(f"Success: {success}")
print(f"Cypher:  {result}")
print(f"Error:  {error}")

print("\n" + "=" * 50)
print("Test 3: Query conversion - Family graph")
print("=" * 50)

success, result, error = nl_to_cypher_service. convert(
    "Mostrami il grafo completo della famiglia test-family"
)
print(f"Success: {success}")
print(f"Cypher: {result}")
print(f"Error: {error}")

print("\n" + "=" * 50)
print("Test 4: Invalid query - Count (should fail)")
print("=" * 50)

success, result, error = nl_to_cypher_service.convert(
    "Quanti modelli ci sono in totale?"
)
print(f"Success: {success}")
print(f"Cypher: {result}")
print(f"Error: {error}")

print("\n" + "=" * 50)
print("Test 5: Query conversion - Longest path")
print("=" * 50)

success, result, error = nl_to_cypher_service.convert(
    "Mostrami il cammino più lungo padre-figlio"
)
print(f"Success:  {success}")
print(f"Cypher: {result}")
print(f"Error: {error}")
