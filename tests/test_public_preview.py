import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

import pytest

from PhysicalRSI.application import Application
from PhysicalRSI_demos.showcase.catalog import catalog
from PhysicalRSI_demos.showcase.jobs import EvaluationJobs
from PhysicalRSI_demos.showcase.server import Handler

ROOT=Path(__file__).resolve().parents[1]


def test_bundled_recordings_match_original_manifest(monkeypatch,tmp_path):
    monkeypatch.setenv('PHYSICALRSI_DATA_ROOT',str(tmp_path))
    monkeypatch.delenv('ROBOPIANIST_ROOT',raising=False)
    monkeypatch.setenv('PHYSICALRSI_SHOWCASE_OUTPUT',str(tmp_path/'runs'))
    rows=catalog()
    assert {row['id'] for row in rows}=={'piano-october','dex-click_mouse','dex-water_plant','dex-hammer_nail'}
    for row in rows:
        assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
    assert rows[0]['sha256']=='c4e73080b24ca0a64c71106be510271bc8b878b4dde251473f50d85c9f32b899'
    assert rows[0]['qualification'] is None
    assert (ROOT/'PhysicalRSI_demos/piano_skill_library/README.md').is_file()


def test_http_preview_supports_seeking_and_reports_missing_runtime(tmp_path,monkeypatch):
    monkeypatch.setenv('PHYSICALRSI_SHOWCASE_OUTPUT',str(tmp_path/'runs'))
    monkeypatch.delenv('DEXJOCO_ROOT',raising=False)
    monkeypatch.delenv('DEXJOCO_PYTHON',raising=False)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    server.workspace=tmp_path;server.items=catalog();server.jobs=EvaluationJobs()
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(base+'/api/catalog') as response:
            rows=json.load(response)['items']
        assert any(row['id']=='piano-october' for row in rows)
        req=Request(base+'/media/piano-october',headers={'Range':'bytes=0-1023'})
        with urlopen(req) as response:
            assert response.status==206 and len(response.read())==1024
            assert response.headers['Content-Range'].startswith('bytes 0-1023/')
        with urlopen(base+'/api/health') as response:
            assert json.load(response)['workspace']==str(tmp_path)
        from urllib.error import HTTPError
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base+'/api/layouts',data=b'{"count":2}',headers={'Content-Type':'application/json'}))
        assert error.value.code==400
        assert 'DEXJOCO_ROOT' in json.load(error.value)['error']
    finally:
        server.shutdown();server.server_close();thread.join()


def test_job_completion_and_orphan_are_distinct(tmp_path,monkeypatch):
    import time
    monkeypatch.setenv('PHYSICALRSI_SHOWCASE_OUTPUT',str(tmp_path/'runs'))
    jobs=EvaluationJobs()
    job=jobs._launch([sys.executable,'-c','print("finished")'],tmp_path,os.environ.copy(),
                     {'id':'software','group':'dexjoco','kind':'test'})
    deadline=time.monotonic()+10
    while time.monotonic()<deadline and jobs.snapshot(job['id'])['status']=='running':
        time.sleep(.02)
    result=jobs.snapshot(job['id'])
    assert result['status']=='completed' and result['returncode']==0
    assert 'finished' in result['lines']
    restored=EvaluationJobs()
    assert restored.latest()['status']=='completed'
    result.update(status='running',pid=999999999,process_start='0',returncode=None)
    restored.jobs['orphan']=result
    assert restored.snapshot('orphan')['status']=='needs_reconciliation'


def test_new_workflows_are_available_to_terminal_and_conversation(tmp_path):
    from PhysicalRSI.conversation import Conversation
    from types import SimpleNamespace
    class Model:
        config=SimpleNamespace(max_tool_rounds=2)
        calls=0
        def complete(self,history,tools):
            self.calls+=1
            if self.calls==1:
                assert 'layouts' in tools[0]['description']
                return '',[{'id':'demo-list','name':'application_command','arguments':'{"command":"/demos"}'}],[]
            assert any(item.get('result') for item in history)
            return 'The piano and Dexjoco recordings are available.',[],[]
        def tool_result(self,call_id,value):return {'result':value}
    app=Application(tmp_path)
    assert 'piano' in Conversation(app,Model()).ask('Show me the available demos.')
    provenance=app.command('/robodojo')
    assert provenance['revision']=='393f730788e7277c7d86829dd5806a924ac31e74'
    assert app.command('/experiment')['outcome']=='success'
