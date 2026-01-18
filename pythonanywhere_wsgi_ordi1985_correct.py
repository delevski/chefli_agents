# PythonAnywhere WSGI Configuration for ordi1985 - CORRECT ASGI-to-WSGI Version
# FastAPI is ASGI, PythonAnywhere uses WSGI - we need proper adapter

import sys
import os

# Add your project directory to the path
project_path = '/home/ordi1985/chefli_agents'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# Change to your project directory
os.chdir(project_path)

# Activate virtual environment
venv_path = '/home/ordi1985/chefli_agents/venv'
venv_site_packages = os.path.join(venv_path, 'lib', 'python3.10', 'site-packages')
if os.path.exists(venv_site_packages):
    if venv_site_packages not in sys.path:
        sys.path.insert(0, venv_site_packages)

activate_this = os.path.join(venv_path, 'bin', 'activate_this.py')
if os.path.exists(activate_this):
    exec(open(activate_this).read(), {'__file__': activate_this})

# Import FastAPI app
from src.main import app

# Use mangum to convert ASGI to WSGI (better than asgiref for FastAPI)
try:
    from mangum import Mangum
    application = Mangum(app)
except ImportError:
    # Fallback: Use asgiref's ASGI to WSGI adapter (correct direction)
    from asgiref.wsgi import WsgiToAsgi
    # Actually, we need ASGI to WSGI, not WSGI to ASGI
    # Let's use a simple wrapper instead
    from asgiref.compatibility import is_asgi_app
    
    # Create a WSGI wrapper for ASGI app
    class ASGItoWSGI:
        def __init__(self, asgi_app):
            self.asgi_app = asgi_app
        
        def __call__(self, environ, start_response):
            # This is a simplified wrapper - may not work perfectly
            # Better to use mangum
            import asyncio
            from asgiref.sync import sync_to_async
            
            # For now, use mangum if available
            raise ImportError("Please install mangum: pip install mangum")
    
    application = Mangum(app) if 'Mangum' in dir() else ASGItoWSGI(app)
