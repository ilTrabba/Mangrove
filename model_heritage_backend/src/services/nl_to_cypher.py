"""
Natural Language to Cypher Query Service
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.services.neo4j_service import neo4j_service
from src.config import Config

logger = logging. getLogger(__name__)


class NLToCypherService: 
    
    def __init__(self):
        self._llm = None
        self._schema_cache = None
        self._schema_text = None
        
    @property
    def llm(self) -> ChatGroq: 
        if self._llm is None: 
            api_key = os. getenv("RAG_API_KEY")
            if not api_key: 
                raise ValueError("RAG_API_KEY environment variable is not set.")
            
            self._llm = ChatGroq(
                api_key=api_key,
                model="llama-3.3-70b-versatile",
                temperature=0,
                max_tokens=1024
            )
            logger.info("Groq LLM initialized with llama-3.3-70b-versatile")
        
        return self._llm
    
    def _execute_read_query(self, query: str, params: Dict = None) -> List[Dict[str, Any]]: 
        if not neo4j_service. driver:
            return []
        
        try: 
            with neo4j_service.driver. session(database=Config.NEO4J_DATABASE) as session:
                result = session. run(query, params or {})
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Failed to execute read query: {e}")
            return []
    
    def get_schema(self, force_refresh: bool = False) -> Dict[str, Any]: 
        if self._schema_cache is not None and not force_refresh:
            return self._schema_cache
        
        if not neo4j_service. is_connected():
            logger.warning("Neo4j not connected, using fallback schema")
            self._schema_cache = self._get_fallback_schema()
            self._schema_text = self._format_schema_for_prompt(self._schema_cache)
            return self._schema_cache
        
        logger.info("Fetching Neo4j schema dynamically...")
        
        schema = {"nodes": {}, "relationships": []}
        
        try:
            labels_result = self._execute_read_query("CALL db.labels() YIELD label RETURN label")
            
            for record in labels_result:
                label = record. get("label")
                if label: 
                    props_query = f"MATCH (n:{label}) WITH n LIMIT 1 RETURN keys(n) as properties"
                    props_result = self._execute_read_query(props_query)
                    properties = props_result[0]. get("properties", []) if props_result else []
                    schema["nodes"][label] = properties
            
            rel_result = self._execute_read_query(
                "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
            )
            
            for record in rel_result:
                rel_type = record. get("relationshipType")
                if rel_type:
                    schema["relationships"].append(rel_type)
            
            # Add fallback relationships if none found
            if not schema["relationships"]: 
                schema["relationships"] = ["IS_CHILD_OF", "BELONGS_TO", "HAS_CENTROID"]
            
            self._schema_cache = schema
            self._schema_text = self._format_schema_for_prompt(schema)
            
            logger.info(f"Schema retrieved: {len(schema['nodes'])} nodes, {len(schema['relationships'])} relationships")
            return schema
            
        except Exception as e:
            logger. error(f"Failed to retrieve schema: {e}")
            self._schema_cache = self._get_fallback_schema()
            self._schema_text = self._format_schema_for_prompt(self._schema_cache)
            return self._schema_cache
    
    def _get_fallback_schema(self) -> Dict[str, Any]:
        return {
            "nodes": {
                "Model": ["id", "name", "description", "total_parameters", "is_foundation_model", "status"],
                "Family": ["id", "display_name", "member_count", "has_foundation_model"],
                "Centroid": ["id", "family_id"]
            },
            "relationships": ["IS_CHILD_OF", "BELONGS_TO", "HAS_CENTROID"]
        }
    
    def _format_schema_for_prompt(self, schema: Dict[str, Any]) -> str:
        lines = ["## Database Schema\n"]
        
        lines.append("### Node Types:")
        for node_label, properties in schema["nodes"].items():
            props_str = ", ". join(properties[: 10]) if properties else "id"
            lines. append(f"- {node_label}:  {props_str}")
        
        lines.append("\n### Relationships:")
        for rel in schema["relationships"]: 
            lines.append(f"- {rel}")
        
        lines.append("\n### Patterns:")
        lines.append("- (Model)-[IS_CHILD_OF]->(Model): child derived from parent")
        lines.append("- (Model)-[BELONGS_TO]->(Family): model belongs to family") 
        lines. append("- (Family)-[HAS_CENTROID]->(Centroid): family has centroid")
        
        return "\n".join(lines)
    
    def _build_prompt(self) -> ChatPromptTemplate:
        system_template = """You are a Neo4j Cypher query generator.  Convert natural language to Cypher. 

{schema}

## CRITICAL RULES: 

1. READ-ONLY:  Only MATCH, RETURN, WITH, WHERE, ORDER BY, LIMIT, OPTIONAL MATCH
2. ALWAYS return relationship variables when querying relationships
3. Return nodes/relationships for graph visualization, NOT scalars

## SYNONYMS (understand these as equivalent):
- "family nodes" / "nodi famiglia" / "families" = Family nodes
- "model nodes" / "nodi modello" / "models" = Model nodes  
- "foundation models" / "root models" / "radici" = Models with no parent
- "show" / "display" / "mostra" / "visualizza" = MATCH ...  RETURN
- "derived from" / "child of" / "derivano da" / "figli di" = IS_CHILD_OF relationship
- "most distant from root" / "più distante dalla radice" = longest path from root

## OUTPUT FORMAT:
Return ONLY the Cypher query.  No explanation, no markdown, no backticks. 
If impossible, return:  ERROR:  <reason>

## EXAMPLES: 

Q: Show all families / Mostrami le famiglie / Show family nodes / Mostrami i nodi famiglia
MATCH (f:Family) RETURN f

Q: Show all models
MATCH (m: Model) RETURN m

Q: Show foundation models
MATCH (m:Model) WHERE NOT (m)-[:IS_CHILD_OF]->() RETURN m

Q: Show models with their families
MATCH (m:Model)-[r: BELONGS_TO]->(f:Family) RETURN m, r, f

Q:  Show parent-child relationships
MATCH (child:Model)-[r:IS_CHILD_OF]->(parent:Model) RETURN child, r, parent

Q:  Show the model most distant from root and the root
MATCH path = (leaf:Model)-[rels:IS_CHILD_OF*]->(root: Model) WHERE NOT (root)-[:IS_CHILD_OF]->() WITH path, leaf, root, length(path) as depth ORDER BY depth DESC LIMIT 1 UNWIND relationships(path) as r UNWIND nodes(path) as n RETURN DISTINCT n, r

Q: Show longest path
MATCH path = (leaf:Model)-[:IS_CHILD_OF*]->(root:Model) WHERE NOT ()-[:IS_CHILD_OF]->(leaf) AND NOT (root)-[:IS_CHILD_OF]->() WITH path ORDER BY length(path) DESC LIMIT 1 RETURN path

Q: Show all centroids
MATCH (f:Family)-[r:HAS_CENTROID]->(c:Centroid) RETURN f, r, c

Q:  How many models?  / Quanti modelli? 
ERROR: This requires a count, not a graph.  Ask to "show all models" instead."""

        return ChatPromptTemplate. from_messages([
            ("system", system_template),
            ("human", "Q: {question}")
        ])
    
    def convert(self, natural_language_query: str) -> Tuple[bool, str, Optional[str]]: 
        if not natural_language_query or not natural_language_query. strip():
            return False, "", "Query cannot be empty."
        
        if len(natural_language_query) > 500:
            return False, "", "Query too long.  Maximum 500 characters."
        
        try: 
            self.get_schema()
            
            prompt = self._build_prompt()
            chain = prompt | self. llm | StrOutputParser()
            
            result = chain.invoke({
                "schema": self._schema_text,
                "question": natural_language_query. strip()
            })
            
            result = result.strip()
            
            # Clean markdown
            if result.startswith("```"):
                lines = [l for l in result.split("\n") if not l.startswith("```")]
                result = "\n".join(lines).strip()
            
            if result.upper().startswith("ERROR: "):
                return False, "", result[6:].strip()
            
            if not self._is_valid_cypher_structure(result):
                return False, "", "Could not generate a valid query."
            
            logger.info(f"Converted to Cypher: {result[: 80]}...")
            return True, result, None
            
        except Exception as e: 
            logger.error(f"Conversion error: {e}")
            return False, "", f"Processing error: {str(e)}"
    
    def _is_valid_cypher_structure(self, query: str) -> bool:
        q = query.upper().strip()
        if not any(q.startswith(s) for s in ["MATCH", "OPTIONAL MATCH", "WITH", "CALL"]):
            return False
        if "RETURN" not in q:
            return False
        return True
    
    def refresh_schema(self) -> Dict[str, Any]: 
        return self.get_schema(force_refresh=True)


nl_to_cypher_service = NLToCypherService()