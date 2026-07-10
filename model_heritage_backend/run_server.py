#!/usr/bin/env python3
import os
import sys
import logging

from src.main import app

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Configura il logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)  # Silenzia werkzeug
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s [%(asctime)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)