"""Connect CLI and conversation commands to one persistent workbench."""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PhysicalRSI_core.infra.storage import atomic_json, read_json


def request(url, route, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url+route, data=data, headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        try:
            message = json.load(error).get('error', str(error))
        except ValueError:
            message = str(error)
        raise ValueError(message) from None


def open_workbench(workspace, section='piano'):
    if section not in {'baseline', 'piano', 'dexjoco'}:
        raise ValueError('Choose baseline, piano or dexjoco')
    workspace = Path(workspace).resolve()
    folder = workspace/'showcase'
    state = folder/'server.json'
    configured = os.environ.get('PHYSICALRSI_SHOWCASE_PORT')
    saved = read_json(state) if state.is_file() else {}
    port = int(configured or saved.get('port', 8765))
    if not 1 <= port <= 65535:
        raise ValueError('Preview port must be between 1 and 65535')
    url = f'http://127.0.0.1:{port}'

    def ready():
        try:
            health = request(url, '/api/health')
            return health.get('workspace') == str(workspace) and health.get('service') == 'physicalrsi-workbench/v1'
        except (OSError, ValueError):
            return False

    if not ready():
        with socket.socket() as probe:
            try:
                probe.bind(('127.0.0.1', port))
            except OSError:
                if configured:
                    raise ValueError(f'Port {port} belongs to another service/workspace; choose --port') from None
                probe.bind(('127.0.0.1', 0))
                port = probe.getsockname()[1]
                url = f'http://127.0.0.1:{port}'
        folder.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        model = workspace/'model.json'
        if model.is_file():env['PHYSICALRSI_SHOWCASE_MODEL_CONFIG'] = str(model)
        with (folder/'server.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-m', 'PhysicalRSI_demos.showcase.server',
                '--port', str(port), '--workspace', str(workspace)],
                cwd=Path(__file__).resolve().parents[2], stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env=env)
        for _ in range(50):
            if ready():break
            if process.poll() is not None:
                raise RuntimeError('Workbench failed; see '+str(folder/'server.log'))
            time.sleep(.1)
        else:raise RuntimeError('Workbench startup timed out; inspect '+str(folder/'server.log'))
        atomic_json(state, {'port':port, 'pid':process.pid, 'workspace':str(workspace)})
    return {'workbench':url+'#'+section, 'section':section,
            'scope':'Recorded demonstrations and live jobs are labeled separately'}


def workbench_command(workspace, route, payload=None):
    url = open_workbench(workspace, 'dexjoco')['workbench'].split('#')[0]
    return request(url, route, payload)
