# PythonAnywhere WSGI Configuration for ordi1985
# Copy this entire content to your WSGI configuration file

import sys
import os

# Add your project directory to the path
path = '/home/ordi1985/chefli_agents'
if path not in sys.path:
    sys.path.insert(0, path)

# Change to your project directory
os.chdir(path)

# Activate virtual environment
activate_this = '/home/ordi1985/chefli_agents/venv/bin/activate_this.py'
if os.path.exists(activate_this):
    exec(open(activate_this).read(), {'__file__': activate_this})

# Import your FastAPI app
from src.main import app

# PythonAnywhere expects a variable called 'application'
application = app
