"""Workflow commands shared by the terminal and model tool calls."""
import os
import shlex
from pathlib import Path

from PhysicalRSI_core.infra.storage import read_json


def doctor():
    from PhysicalRSI_demos.showcase.catalog import catalog
    import PhysicalRSI_core
    root = Path(PhysicalRSI_core.__file__).resolve().parent.parent
    fork = root/'PhysicalRSI_baselines/robodojo/XPolicyLab'
    configured = {name:bool(os.environ.get(name)) for name in
                  ('DEXJOCO_ROOT','DEXJOCO_PYTHON','PHYSICALRSI_VLA_PYTHON',
                   'PHYSICALRSI_PI05_PYTHON','PHYSICALRSI_SKILL_CONFIG','ROBOPIANIST_ROOT')}
    return {'core':str(root/'PhysicalRSI_core'), 'media_count':len(catalog()),
            'robodojo_checkout':(fork/'policy/physicalRSI/model.py').is_file(),
            'optional_configuration':configured,
            'preview':'/demo piano or /demo dexjoco; recordings need no simulator or API key',
            'scope':'Installation inventory; use task preflight before live execution'}


def dispatch_workflow(application, command, tail):
    from PhysicalRSI_demos.showcase.launch import workbench_command
    args = shlex.split(tail)
    workspace = application.workspace
    if command in {'doctor','demos','jobs'} and args:
        raise ValueError(f'/{command} takes no arguments')
    if command == 'doctor':
        return doctor()
    if command == 'demos':
        from PhysicalRSI_demos.showcase.catalog import catalog
        return [{k:row[k] for k in ('id','title','group','note')} for row in catalog()]
    if command == 'experiment':
        from PhysicalRSI_demos.experiment import run
        if len(args)>1:raise ValueError('/experiment takes an optional run ID')
        return run(workspace/'experiments', args[0] if args else 'counter-example')
    if command == 'jobs':
        return workbench_command(workspace, '/api/jobs')
    if command == 'logs':
        from PhysicalRSI_core.infra.storage import identifier
        if len(args)>1:raise ValueError('/logs takes an optional job ID')
        return workbench_command(workspace, '/api/jobs/'+identifier(args[0]) if args else '/api/status')
    if command == 'robodojo':
        if len(args)>1:raise ValueError('/robodojo takes an optional task name')
        from PhysicalRSI_baselines.robodojo import __file__ as baseline_file
        if not args:
            return read_json(Path(baseline_file).parent/'provenance.json')
        from PhysicalRSI_core.infra.storage import identifier
        return workbench_command(workspace,'/api/evaluate',{'task':identifier(args[0])})
    if command == 'cycle' and args in (['status'],['pause']):
        return workbench_command(workspace,'/api/cycle',{'action':'pause'} if args==['pause'] else None)
    if len(args) > (2 if command=='layouts' else 1):
        raise ValueError(f'Too many arguments for /{command}')
    count = int(args[0]) if args else (100 if command=='train' else 2)
    key = {'layouts':'count','collect':'episodes','train':'steps','cycle':'rounds'}[command]
    payload = {key:count}
    if command=='layouts' and len(args)==2:payload['seed']=int(args[1])
    return workbench_command(workspace,'/api/'+command,payload)
