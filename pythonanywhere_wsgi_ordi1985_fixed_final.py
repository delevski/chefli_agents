# PythonAnywhere WSGI Configuration for ordi1985 - FIXED FINAL VERSION
# Handles empty CONTENT_LENGTH properly

import sys
import os

project_path = '/home/ordi1985/chefli_agents'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

os.chdir(project_path)

venv_path = '/home/ordi1985/chefli_agents/venv'
venv_site_packages = os.path.join(venv_path, 'lib', 'python3.10', 'site-packages')
if os.path.exists(venv_site_packages):
    if venv_site_packages not in sys.path:
        sys.path.insert(0, venv_site_packages)

activate_this = os.path.join(venv_path, 'bin', 'activate_this.py')
if os.path.exists(activate_this):
    exec(open(activate_this).read(), {'__file__': activate_this})

from src.main import app

import asyncio

class SimpleASGItoWSGI:
    def __init__(self, asgi_app):
        self.asgi_app = asgi_app
    
    def __call__(self, environ, start_response):
        try:
            scope = {
                'type': 'http',
                'method': environ['REQUEST_METHOD'],
                'path': environ.get('PATH_INFO', '/'),
                'query_string': environ.get('QUERY_STRING', '').encode(),
                'headers': [
                    (k.replace('HTTP_', '').lower().replace('_', '-').encode(), v.encode())
                    for k, v in environ.items()
                    if k.startswith('HTTP_')
                ],
                'server': (environ.get('SERVER_NAME', 'localhost'), int(environ.get('SERVER_PORT', 80))),
                'scheme': environ.get('wsgi.url_scheme', 'http'),
            }
            
            request_queue = asyncio.Queue()
            response_queue = asyncio.Queue()
            
            # Fix: Handle empty CONTENT_LENGTH properly
            content_length_str = environ.get('CONTENT_LENGTH', '0')
            content_length = int(content_length_str) if content_length_str else 0
            
            body = environ['wsgi.input'].read(content_length) if content_length > 0 else b''
            
            async def receive():
                return await request_queue.get()
            
            async def send(message):
                await response_queue.put(message)
            
            request_queue.put_nowait({
                'type': 'http.request',
                'body': body,
                'more_body': False,
            })
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.asgi_app(scope, receive, send))
            finally:
                loop.close()
            
            response_start = None
            response_body_parts = []
            
            while not response_queue.empty():
                message = response_queue.get_nowait()
                if message['type'] == 'http.response.start':
                    response_start = message
                elif message['type'] == 'http.response.body':
                    response_body_parts.append(message.get('body', b''))
            
            if response_start:
                status = response_start['status']
                headers = [(k.decode(), v.decode()) for k, v in response_start.get('headers', [])]
                start_response(f"{status}", headers)
                return response_body_parts
            else:
                start_response('500 Internal Server Error', [('Content-Type', 'text/plain')])
                return [b'Internal Server Error']
        except Exception as e:
            start_response('500 Internal Server Error', [('Content-Type', 'text/plain')])
            return [f'Error: {str(e)}'.encode()]

application = SimpleASGItoWSGI(app)
