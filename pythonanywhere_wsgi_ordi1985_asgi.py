# PythonAnywhere WSGI Configuration for ordi1985 - ASGI Compatible Version
# FastAPI is ASGI, so we need to wrap it with ASGI-to-WSGI adapter

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

# Wrap FastAPI (ASGI) app with WSGI adapter
from asgiref.wsgi import WsgiToAsgi

# PythonAnywhere expects a variable called 'application'
application = WsgiToAsgi(app)
