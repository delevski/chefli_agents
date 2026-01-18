# PythonAnywhere WSGI Configuration for ordi1985 - WORKING VERSION
# Proper ASGI-to-WSGI wrapper for FastAPI

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

# Create proper WSGI wrapper for ASGI app
from asgiref.wsgi import WsgiToAsgi

# WsgiToAsgi converts WSGI to ASGI, but we need the reverse
# So we'll use a custom wrapper
import asyncio
from asgiref.sync import sync_to_async
from io import BytesIO

class ASGItoWSGI:
    """Convert ASGI app to WSGI"""
    def __init__(self, asgi_app):
        self.asgi_app = asgi_app
    
    def __call__(self, environ, start_response):
        # Convert WSGI environ to ASGI scope
        scope = {
            'type': 'http',
            'method': environ['REQUEST_METHOD'],
            'path': environ.get('PATH_INFO', '/'),
            'query_string': environ.get('QUERY_STRING', '').encode(),
            'headers': [
                (k.lower().encode(), v.encode())
                for k, v in environ.items()
                if k.startswith('HTTP_')
            ],
            'server': (environ.get('SERVER_NAME', 'localhost'), int(environ.get('SERVER_PORT', 80))),
            'client': None,
            'scheme': environ.get('wsgi.url_scheme', 'http'),
        }
        
        # Create ASGI message
        async def receive():
            return {
                'type': 'http.request',
                'body': environ.get('wsgi.input').read() if environ.get('CONTENT_LENGTH') else b'',
            }
        
        response_status = None
        response_headers = []
        response_body = BytesIO()
        
        async def send(message):
            nonlocal response_status, response_headers
            if message['type'] == 'http.response.start':
                response_status = f"{message['status']}"
                response_headers = [(k.decode(), v.decode()) for k, v in message.get('headers', [])]
            elif message['type'] == 'http.response.body':
                response_body.write(message.get('body', b''))
        
        # Run ASGI app
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.asgi_app(scope, receive, send))
        finally:
            loop.close()
        
        # Start WSGI response
        start_response(response_status, response_headers)
        response_body.seek(0)
        return [response_body.read()]

# Use the wrapper
application = ASGItoWSGI(app)
