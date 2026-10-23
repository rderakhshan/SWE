"""Minimal browser surface for the Irina agent.

Input: CLI arguments ``-d`` workspace and ``--port`` TCP port. Output: a
local HTTP page with a form and rendered agent output. Example:
``python -m irina.web -d . --port 3080`` serves localhost:3080. This is a
small local UI; it does not claim compatibility with the unavailable original
Odysseus SSE/approval browser implementation. Bind to loopback only.
"""
import argparse, html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs
from .harness import Harness
from .security import Policy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--workdir',default='.')
    parser.add_argument('--port',type=int,default=3080)
    args = parser.parse_args()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self): self.render('')
        def do_POST(self):
            n = min(int(self.headers.get('Content-Length','0')), 10000)
            task = parse_qs(self.rfile.read(n).decode()).get('task',[''])[0]
            try:
                result = Harness(args.workdir,policy=Policy('read-only')).run(task)
                output = result['messages'][-1]['content']
            except Exception as exc: output = f'{type(exc).__name__}: {exc}'
            self.render(output)
        def render(self, output):
            body = ('<!doctype html><title>Irina</title><h1>Irina</h1>'
                    '<form method="post"><textarea name="task" rows="7" cols="70"></textarea>'
                    '<button>Run (read-only)</button></form><pre>'+html.escape(output)+'</pre>')
            data=body.encode();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    print(f'Irina web: http://127.0.0.1:{args.port}')
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
if __name__ == '__main__': main()
