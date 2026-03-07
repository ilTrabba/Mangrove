"""
Cypher Query Validator

Validates Cypher queries to ensure they are: 
1. Read-only (no write operations)
2. Visualizable (return nodes/relationships, not scalars)
3. Safe (no injection or malicious patterns)
"""

import re
import logging
from typing import Tuple, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Validation result types"""
    VALID = "valid"
    WRITE_OPERATION = "write_operation"
    NOT_VISUALIZABLE = "not_visualizable"
    MALFORMED = "malformed"
    DANGEROUS = "dangerous"


class CypherValidator:
    """
    Validates Cypher queries for safety and visualization compatibility. 
    
    Security Rules:
    - Block all write operations (CREATE, DELETE, SET, MERGE, etc.)
    - Block administrative operations (DROP, GRANT, etc.)
    - Block procedure calls that modify data
    
    Visualization Rules: 
    - Query must return nodes or relationships
    - Scalar-only returns are rejected
    """
    
    # Write operations that MUST be blocked
    WRITE_KEYWORDS = [
        r'\bCREATE\b',
        r'\bDELETE\b',
        r'\bDETACH\b',
        r'\bSET\b',
        r'\bREMOVE\b',
        r'\bMERGE\b',
        r'\bDROP\b',
        r'\bALTER\b',
        r'\bGRANT\b',
        r'\bREVOKE\b',
        r'\bDENY\b',
    ]
    
    # Dangerous patterns
    DANGEROUS_PATTERNS = [
        r'LOAD\s+CSV',
        r'CALL\s+dbms\.',
        r'CALL\s+db\. create',
        r'CALL\s+db\.drop',
        r'CALL\s+apoc\.export',
        r'CALL\s+apoc\.import',
        r'PERIODIC\s+COMMIT',
    ]
    
    # Scalar/aggregation functions that don't return graph elements
    SCALAR_FUNCTIONS = [
        'count', 'sum', 'avg', 'min', 'max', 'collect', 
        'size', 'length', 'toInteger', 'toFloat', 'toString',
        'head', 'last', 'tail', 'range', 'keys', 'properties',
        'id', 'type', 'labels', 'coalesce', 'timestamp'
    ]
    
    # Valid read operation starters
    VALID_READ_STARTERS = [
        r'^\s*MATCH\b',
        r'^\s*OPTIONAL\s+MATCH\b',
        r'^\s*WITH\b',
        r'^\s*CALL\s+db\.labels',
        r'^\s*CALL\s+db\. relationshipTypes',
        r'^\s*CALL\s+db\.schema',
        r'^\s*CALL\s+db\. propertyKeys',
    ]
    
    def __init__(self):
        """Initialize the validator with compiled regex patterns."""
        self._write_patterns = [re.compile(p, re.IGNORECASE) for p in self.WRITE_KEYWORDS]
        self._dangerous_patterns = [re.compile(p, re. IGNORECASE) for p in self. DANGEROUS_PATTERNS]
        self._valid_starters = [re. compile(p, re.IGNORECASE) for p in self.VALID_READ_STARTERS]
        
        # Build scalar function pattern
        scalar_funcs = '|'.join(self.SCALAR_FUNCTIONS)
        self._scalar_func_pattern = re. compile(rf'\b({scalar_funcs})\s*\(', re.IGNORECASE)
    
    def validate(self, query: str) -> Tuple[bool, ValidationResult, Optional[str]]: 
        """
        Validate a Cypher query.
        
        Args:
            query:  The Cypher query string to validate
            
        Returns:
            Tuple of (is_valid, result_type, error_message)
        """
        if not query or not query.strip():
            return False, ValidationResult.MALFORMED, "La query non può essere vuota."
        
        query = query.strip()
        
        # Check 1: Must start with a valid read operation
        if not self._starts_with_read_operation(query):
            return False, ValidationResult. MALFORMED, (
                "La query deve iniziare con MATCH, OPTIONAL MATCH o WITH."
            )
        
        # Check 2: Must contain RETURN
        if not re.search(r'\bRETURN\b', query, re.IGNORECASE):
            return False, ValidationResult. MALFORMED, (
                "La query deve contenere una clausola RETURN."
            )
        
        # Check 3: No write operations
        write_check = self._contains_write_operation(query)
        if write_check: 
            return False, ValidationResult.WRITE_OPERATION, (
                f"Operazione di scrittura non permessa: {write_check}. "
                "Sono ammesse solo query di lettura."
            )
        
        # Check 4: No dangerous patterns
        dangerous_check = self._contains_dangerous_pattern(query)
        if dangerous_check:
            return False, ValidationResult. DANGEROUS, (
                f"Pattern non permesso rilevato: {dangerous_check}. "
                "Questa operazione non è consentita per motivi di sicurezza."
            )
        
        # Check 5: Must return visualizable data (not just scalars)
        scalar_check = self._returns_only_scalars(query)
        if scalar_check:
            return False, ValidationResult.NOT_VISUALIZABLE, (
                f"La query restituisce solo valori scalari ({scalar_check}) "
                "che non possono essere visualizzati come grafo.  "
                "Riformula la domanda chiedendo di mostrare nodi o relazioni."
            )
        
        logger.info(f"Query validated successfully: {query[: 50]}...")
        return True, ValidationResult.VALID, None
    
    def _starts_with_read_operation(self, query: str) -> bool:
        """Check if query starts with a valid read operation."""
        return any(pattern.match(query) for pattern in self._valid_starters)
    
    def _contains_write_operation(self, query: str) -> Optional[str]:
        """Check if query contains any write operation.  Returns the matched keyword."""
        for pattern in self._write_patterns:
            match = pattern.search(query)
            if match:
                return match.group(0).upper()
        return None
    
    def _contains_dangerous_pattern(self, query: str) -> Optional[str]:
        """Check if query contains dangerous patterns.  Returns the matched pattern."""
        for pattern in self._dangerous_patterns:
            match = pattern.search(query)
            if match:
                return match.group(0)
        return None
    
    def _returns_only_scalars(self, query: str) -> Optional[str]:
        """
        Check if the RETURN clause only contains scalar values. 
        Returns the scalar function name if only scalars, None otherwise.
        """
        # Extract the RETURN clause (everything after RETURN until ORDER BY, LIMIT, or end)
        return_match = re.search(
            r'\bRETURN\b\s+(.+?)(?:\s+ORDER\s+BY|\s+LIMIT|\s+SKIP|\s*$)', 
            query, 
            re. IGNORECASE | re.DOTALL
        )
        
        if not return_match: 
            return None
        
        return_clause = return_match. group(1).strip()
        
        # Special case: if RETURN contains 'path', it's visualizable
        if re.search(r'\bpath\b', return_clause, re.IGNORECASE):
            return None
        
        # Split return items by comma (handling nested parentheses)
        return_items = self._split_return_items(return_clause)
        
        # Check each return item
        has_node_or_relationship = False
        scalar_functions_found = []
        
        for item in return_items: 
            item = item. strip()
            
            # Remove alias (AS something)
            item_without_alias = re. sub(r'\s+[Aa][Ss]\s+\w+\s*$', '', item).strip()
            
            # Check if it's a scalar function
            scalar_match = self._scalar_func_pattern.match(item_without_alias)
            if scalar_match: 
                scalar_functions_found.append(scalar_match.group(1))
                continue
            
            # Check if it's a property access (n.property)
            if re.match(r'^[\w]+\.[\w]+$', item_without_alias):
                # This is a property access, which is scalar
                continue
            
            # Check if it's a simple variable (node or relationship)
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', item_without_alias):
                has_node_or_relationship = True
                continue
            
            # Check if it's a string literal or number
            if re.match(r'^["\'].*["\']$', item_without_alias) or re.match(r'^-?\d+\. ?\d*$', item_without_alias):
                continue
            
            # If we can't determine, assume it might be a node/relationship
            has_node_or_relationship = True
        
        # If we found only scalar functions and no nodes/relationships
        if scalar_functions_found and not has_node_or_relationship:
            return ', '.join(set(scalar_functions_found))
        
        return None
    
    def _split_return_items(self, return_clause: str) -> list:
        """
        Split RETURN clause items by comma, handling nested parentheses. 
        """
        items = []
        current_item = ""
        paren_depth = 0
        bracket_depth = 0
        
        for char in return_clause:
            if char == '(': 
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == '[':
                bracket_depth += 1
            elif char == ']':
                bracket_depth -= 1
            elif char == ',' and paren_depth == 0 and bracket_depth == 0:
                items.append(current_item. strip())
                current_item = ""
                continue
            
            current_item += char
        
        if current_item. strip():
            items.append(current_item.strip())
        
        return items
    
    def sanitize(self, query: str) -> str:
        """
        Sanitize a query by removing potentially dangerous content.
        
        Note: This is a secondary defense.  Primary defense is validation.
        """
        # Remove comments
        query = re.sub(r'//.*$', '', query, flags=re. MULTILINE)
        query = re.sub(r'/\*.*?\*/', '', query, flags=re. DOTALL)
        
        # Remove excessive whitespace
        query = ' '.join(query. split())
        
        return query. strip()


# Singleton instance
cypher_validator = CypherValidator()