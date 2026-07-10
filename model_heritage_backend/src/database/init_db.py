#!/usr/bin/env python3
"""
Database initialization script for Model Heritage backend.
Neo4j-only architecture - initializes Neo4j constraints and indexes.
"""
import os
import sys

from src.main import app
from src.services.neo4j_service import neo4j_service

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def init_database():
    """Initialize the Neo4j database constraints and indexes."""
    with app.app_context():
        print("Initializing Neo4j database...")
        if neo4j_service.is_connected():
            print("Creating Neo4j constraints and indexes...")
            neo4j_service.create_constraints()
            print("Neo4j database initialization complete.")
        else:
            print("Error: Could not connect to Neo4j database.")
            print("Please check your Neo4j connection settings.")

if __name__ == '__main__':
    init_database()