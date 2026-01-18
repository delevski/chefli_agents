# PythonAnywhere WSGI Configuration for ordi1985 - FIXED VERSION
# This version ensures virtual environment is properly activated

import sys
import os

# Add your project directory to the path
project_path = '/home/ordi1985/chefli_agents'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# Change to your project directory
os.chdir(project_path)

# Activate virtual environment - try multiple methods
venv_path = '/home/ordi1985/chefli_agents/venv'

# Method 1: Add venv site-packages to path
venv_site_packages = os.path.join(venv_path, 'lib', 'python3.10', 'site-packages')
if os.path.exists(venv_site_packages):
    if venv_site_packages not in sys.path:
        sys.path.insert(0, venv_site_packages)

# Method 2: Use activate_this.py if available
activate_this = os.path.join(venv_path, 'bin', 'activate_this.py')
if os.path.exists(activate_this):
    exec(open(activate_this).read(), {'__file__': activate_this})

# Import your FastAPI app
from src.main import app

# PythonAnywhere expects a variable called 'application'
application = app
