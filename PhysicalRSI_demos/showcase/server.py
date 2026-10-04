"""Serve the local demonstration workbench and seekable video files."""
import argparse
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from .catalog import catalog
from .jobs import EvaluationJobs
from .conversation import DemoConversation
from .cycle_status import cycle_status

WEB=Path(__file__).with_name('web')

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def respond(self, data, status=200):
        body=json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path not in ('/api/evaluate', '/api/layouts', '/api/collect', '/api/train', '/api/cycle', '/api/chat'):
            self.send_error(404);return
        origin = self.headers.get('Origin')
        if origin and urlparse(origin).netloc != self.headers.get('Host'):
            self.send_error(403);return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0 < length <= 4096:raise ValueError('Invalid request size')
            request=json.loads(self.rfile.read(length))
            if not isinstance(request, dict):raise ValueError('Expected a JSON object')
            if self.path == '/api/layouts':
                self.respond(self.server.jobs.layouts(request.get('count',2), request.get('seed')),202);return
            if self.path == '/api/cycle':
                if request.get('action') == 'pause':
                    self.respond(self.server.jobs.pause_cycle());return
                self.respond(self.server.jobs.cycle(request.get('rounds', 2)),202);return
            if self.path == '/api/chat':
                message=request.get('message')
                if not isinstance(message,str) or not message.strip():raise ValueError('Enter a message')
                self.respond(self.server.conversation.ask(message));return
            if self.path == '/api/train':
                self.respond(self.server.jobs.train(request.get('steps', 100)),202);return
            if self.path == '/api/collect':
                self.respond(self.server.jobs.collect(request.get('episodes', 2)),202);return
            row=next((r for r in self.server.items if r['id']==request.get('id')),None)
            if row is None and request.get('task'):
                from PhysicalRSI_core.infra.storage import identifier
                task=identifier(request['task'])
                row={'id':'baseline-'+task,'title':task,'group':'baseline','task':task}
            if row is None:raise ValueError('Unknown recording')
            self.respond(self.server.jobs.start(row),202)
        except (ValueError,OSError,RuntimeError) as error:
            self.respond({'error':str(error)},400)

    def do_GET(self):
        route=urlparse(self.path).path
        if route=='/api/health':
            self.respond({'service':'physicalrsi-workbench/v1','workspace':str(self.server.workspace)});return
        if route=='/api/cycle':
            self.respond(cycle_status());return
        if route=='/api/status':
            self.respond(self.server.jobs.latest());return
        if route=='/api/jobs':
            self.respond(self.server.jobs.list());return
        if route.startswith('/api/jobs/'):
            try:self.respond(self.server.jobs.snapshot(route.split('/')[-1]))
            except KeyError:self.send_error(404)
            return
        if route=='/api/catalog':
            self.server.items=catalog()
            items=[{k:v for k,v in row.items() if k!='path'} for row in self.server.items]
            body=json.dumps({'items':items}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
        if route.startswith('/media/'):
            row=next((r for r in self.server.items if r['id']==route.removeprefix('/media/')),None)
            if not row:self.send_error(404);return
            path=Path(row['path'])
        else:
            name={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}.get(route)
            if not name:self.send_error(404);return
            path=WEB/name
        size=path.stat().st_size;start,end=0,size-1
        partial=self.headers.get('Range')
        if partial:
            try:
                interval=partial.removeprefix('bytes=').split(',')[0];left,right=interval.split('-')
                if left:start=int(left);end=min(int(right),end) if right else end
                else:start=max(0,size-int(right))
                if start<0 or start>end:raise ValueError
            except (ValueError,TypeError):self.send_error(416);return
        self.send_response(206 if partial else 200)
        self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes')
        if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers()
        try:
            with path.open('rb') as stream:
                stream.seek(start);remaining=end-start+1
                while remaining:
                    block=stream.read(min(1024*1024,remaining))
                    if not block:break
                    self.wfile.write(block);remaining-=len(block)
        except (BrokenPipeError,ConnectionResetError):pass


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--workspace',type=Path,default=Path('.physicalrsi'))
    args=parser.parse_args()
    workspace=args.workspace.resolve()
    os.environ.setdefault('PHYSICALRSI_SHOWCASE_OUTPUT',str(workspace/'showcase/runs'))
    os.environ.setdefault('PHYSICALRSI_SHOWCASE_MODEL_CONFIG',str(workspace/'model.json'))
    server=ThreadingHTTPServer((args.host,args.port),Handler)
    server.workspace=workspace
    server.items=catalog();server.jobs=EvaluationJobs();server.conversation=DemoConversation(server.jobs)
    print(f'physicalRSI workbench http://{args.host}:{server.server_port}',flush=True)
    server.serve_forever()

if __name__=='__main__':main()
