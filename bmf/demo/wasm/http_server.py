#! python3
# Usage: sudo SERVER_DIRECTORY=$SERVER_DIRECTORY python3 http_server.py
import http.server
import ssl
import os

directory = os.getenv('SERVER_DIRECTORY')
if directory is None:
    raise "$SERVER_DIRECTORY not exists. Set before run the script"
else:
    print(directory)
class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()

server_address = ('0.0.0.0', 443)
httpd = http.server.HTTPServer(server_address, CustomHTTPRequestHandler)
httpd.socket = ssl.wrap_socket(httpd.socket,
                               keyfile="server.pem",  
                               certfile="server.pem",# Create you own
                               server_side=True)

print(f"Serving on port {server_address[1]}")
httpd.serve_forever()
