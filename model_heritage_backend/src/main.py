import os
import sys
import logging
import atexit

from flask import Flask, send_from_directory
from flask_cors import CORS
from src.config import Config
from src.routes.user import user_bp
from src.routes.models import models_bp
from src.routes.graph import graph_bp
from src.services.neo4j_service import neo4j_service
from src.services.url_verification_service import URLVerificationService
from src.routes.nl_query import nl_query_bp

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Mostra INFO e superiori (INFO, WARNING, ERROR, CRITICAL)
logging.basicConfig(level=logging.INFO)

# Imposta il livello di logging per il driver Neo4j a WARNING per ridurre il rumore nei log
logging.getLogger("neo4j").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config.from_object(Config)

# Enable CORS for all routes
CORS(app)

# Register blueprints
app.register_blueprint(user_bp, url_prefix='/api')
app.register_blueprint(models_bp, url_prefix='/api')
app.register_blueprint(graph_bp, url_prefix='/api')
app.register_blueprint(nl_query_bp, url_prefix='/api')

# Initialize URL verification service
url_verification_service = None

# Initialize Neo4j constraints and URL verification service
with app.app_context():
    try:
        if neo4j_service.is_connected():
            neo4j_service.create_constraints()
            logger.info("Neo4j constraints initialized")
            
            # Initialize and start URL verification service
            url_verification_service = URLVerificationService(neo4j_service)
            url_verification_service.start()
            logger.info("URL verification service started")
        else:
            logger.warning("Neo4j not connected - graph features will be unavailable")
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j: {e}")

# Register graceful shutdown
def shutdown_services():
    """Gracefully shutdown background services"""
    global url_verification_service
    if url_verification_service:
        url_verification_service.stop()
        logger.info("URL verification service stopped on shutdown")

atexit.register(shutdown_services)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
            return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "index.html not found", 404


if __name__ == '__main__':
    print("🔍 DEBUG: Avvio del server") 
    # Per il debug, usa threaded=True e debug=False
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True, use_reloader=False)
