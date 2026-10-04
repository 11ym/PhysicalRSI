"""Run installed baseline evaluations and retain their output for the workbench."""
import os
import re
import json
import subprocess
import sys
import threading
import uuid
import time
from pathlib import Path

from PhysicalRSI_core.infra.storage import atomic_json


class EvaluationJobs:
    def __init__(self):
        self.jobs = {}
        self.lock = threading.Lock()
        root = Path(os.environ.get('PHYSICALRSI_SHOWCASE_OUTPUT', '.physicalrsi/showcase/runs')).resolve()
        self.root = root
        self.processes = {}
        for path in sorted(root.glob('*/run.json'), key=lambda p:p.stat().st_mtime)[-20:]:
            try:
                job = json.loads(path.read_text())
                job['output'] = str(path.parent)
                log = path.parent/'execution.log'
                job['lines'] = log.read_text().splitlines()[-2000:] if log.is_file() else []
                self.jobs[job['id']] = job
            except (OSError, ValueError, KeyError):
                continue

    def latest(self):
        with self.lock:
            key = next(reversed(self.jobs), None)
        return self.snapshot(key) if key else None

    def list(self):
        return [{k:v for k,v in self.snapshot(key).items() if k!='lines'} for key in list(self.jobs)]

    def layouts(self, count=2, seed=None):
        if type(count) is not int or not 1 <= count <= 100:
            raise ValueError('Choose 1–100 layouts')
        if seed is None:seed=int(time.time_ns()%1_000_000_000)
        if type(seed) is not int or not 0 <= seed < 2**31-count:
            raise ValueError('Choose a nonnegative seed below 2**31-count')
        root, python = os.environ.get('DEXJOCO_ROOT'), os.environ.get('DEXJOCO_PYTHON')
        if not root or not python:raise ValueError('Configure DEXJOCO_ROOT and DEXJOCO_PYTHON')
        command=[python,str(Path(__file__).with_name('layouts.py')),'--root',root,
                 '--count',str(count),'--seed',str(seed)]
        item=dict(id='dex-layouts',title='New mouse layouts',group='dexjoco',task='click_mouse',kind='layouts')
        return self._launch(command,Path(__file__).resolve().parents[2],os.environ.copy(),item,
                            output_argument='layouts.json')

    def start(self, item):
        if item['group'] != 'baseline':
            raise ValueError('Select a baseline recording to run its task')
        required = ('PHYSICALRSI_XPOLICYLAB_ROOT', 'PHYSICALRSI_SKILL_CONFIG',
                    'PHYSICALRSI_POLICY_PYTHON')
        missing = [name for name in required if not os.environ.get(name)]
        if not os.environ.get('PHYSICALRSI_SIM_PYTHON') and not os.environ.get('ROBODOJO_CONDA_ENV'):
            missing.append('PHYSICALRSI_SIM_PYTHON or ROBODOJO_CONDA_ENV')
        if missing:
            raise ValueError('Configure ' + ', '.join(missing))
        if not any(os.environ.get(name) for name in
                   ('PHYSICALRSI_AGENT_API_KEY', 'OPENAI_API_KEY', 'ARK_API_KEY')):
            raise ValueError('Configure an agent API key in the server environment')
        root = Path(os.environ['PHYSICALRSI_XPOLICYLAB_ROOT']).resolve()
        script = root / 'policy/physicalRSI/eval.sh'
        if not script.is_file():
            raise ValueError('The installed PhysicalRSI evaluation entry is missing')
        env = os.environ.copy()
        env['EVAL_ENV_TYPE'] = 'sim'
        if env.get('PHYSICALRSI_SIM_PYTHON'):
            command = [sys.executable, str(Path(__file__).with_name('evaluate_baseline.py')),
                       '--framework', str(root), '--task', item.get('task', item['id'].removeprefix('baseline-'))]
            return self._launch(command, root, env, item, output_argument=True)
        command = ['bash', str(script), 'RoboDojo', item.get('task', item['id'].removeprefix('baseline-')),
                   'skill-library', 'arx_x5', 'joint', '0',
                   env.get('PHYSICALRSI_POLICY_GPU', '0'), env.get('PHYSICALRSI_ENV_GPU', '0'),
                   env['PHYSICALRSI_POLICY_PYTHON'], env['ROBODOJO_CONDA_ENV']]
        return self._launch(command, root, env, item)

    def collect(self, episodes=2):
        if type(episodes) is not int or not 1 <= episodes <= 100:
            raise ValueError('Choose between 1 and 100 episodes')
        root = os.environ.get('DEXJOCO_ROOT')
        python = os.environ.get('DEXJOCO_PYTHON')
        if not root or not python:
            raise ValueError('Configure DEXJOCO_ROOT and DEXJOCO_PYTHON')
        command = [python, str(Path(__file__).with_name('collect_dexjoco.py')),
                   '--root', root, '--episodes', str(episodes)]
        layouts=[job for job in self.jobs.values() if job.get('kind')=='layouts' and job['status']=='completed']
        if not layouts:
            raise ValueError('Generate layouts first with /layouts '+str(episodes))
        manifest=Path(layouts[-1]['output'])/'layouts.json'
        rows=json.loads(manifest.read_text())['layouts']
        if len(rows)!=episodes:
            raise ValueError('Episode count must match the saved layout count: '+str(len(rows)))
        command += ['--layout-manifest',str(manifest),'--seed',str(rows[0]['seed'])]
        item = dict(id='dex-click_mouse', title='Click a mouse', group='dexjoco', task='click_mouse', kind='collection')
        return self._launch(command, Path(__file__).resolve().parents[2], os.environ.copy(), item,
                            output_argument=True)

    def train(self, steps=100):
        if type(steps) is not int or not 1 <= steps <= 10000:
            raise ValueError('Choose between 1 and 10000 training steps')
        with self.lock:
            datasets = [j for j in self.jobs.values()
                        if j.get('kind') == 'collection' and j['status'] == 'completed']
        if not datasets:
            raise ValueError('Collect successful demonstrations before training')
        python = os.environ.get('PHYSICALRSI_VLA_PYTHON', os.environ.get('DEXJOCO_PYTHON'))
        if not python:
            raise ValueError('Configure PHYSICALRSI_VLA_PYTHON with SmolVLA installed')
        command = [python, str(Path(__file__).with_name('train_dexjoco.py')),
                   '--dataset', datasets[-1]['output'], '--steps', str(steps),
                   '--model', os.environ.get('PHYSICALRSI_VLA_MODEL', 'lerobot/smolvla_base')]
        item = dict(id='dex-training', title='Mouse skill training', group='dexjoco',
                    task='click_mouse', kind='training')
        return self._launch(command, Path(__file__).resolve().parents[2], os.environ.copy(), item,
                            output_argument=True)

    def cycle(self, rounds=2):
        if type(rounds) is not int or not 0 <= rounds <= 20:
            raise ValueError('Choose 1–20 rounds, or zero to continue until paused')
        required = ('DEXJOCO_ROOT', 'DEXJOCO_PYTHON', 'PHYSICALRSI_PI05_PYTHON',
                    'PHYSICALRSI_DSW_SSH_KEY', 'PHYSICALRSI_SHOWCASE_AGENT_CLI')
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise ValueError('Configure '+', '.join(missing))
        source = Path(os.environ['DEXJOCO_ROOT']).resolve()
        cycle = Path(os.environ.get('PHYSICALRSI_PI05_CYCLE', str(source/'runs/pi05-cycle')))
        if not (cycle/'cycle.json').is_file():
            raise ValueError('Configure the pi05 cycle and DSW hosts before starting')
        if (cycle/'pause-requested').exists():
            raise ValueError('Remove the cycle pause-requested file before resuming')
        command = [sys.executable, '-m', 'PhysicalRSI_demos.showcase.pi05_cycle',
                   '--cycle', str(cycle), '--source', str(source),
                   '--collection-python', os.environ['DEXJOCO_PYTHON'],
                   '--training-python', os.environ['PHYSICALRSI_PI05_PYTHON'],
                   '--ssh-key', os.environ['PHYSICALRSI_DSW_SSH_KEY'], '--rounds', str(rounds),
                   '--mirror', str(Path(os.environ.get('PHYSICALRSI_DATA_ROOT', '/mnt/data'))/
                                   'physicalrsi-demo-workbench/pi05-cycle')]
        item = dict(id='dex-pi05-cycle', title='Mouse · pi05 cycle', group='dexjoco',
                    task='click_mouse', kind='pi05-cycle')
        return self._launch(command, Path(__file__).resolve().parents[2], os.environ.copy(), item)

    def pause_cycle(self):
        source=os.environ.get('DEXJOCO_ROOT')
        configured=os.environ.get('PHYSICALRSI_PI05_CYCLE')
        if not source and not configured:raise ValueError('No pi05 cycle configured')
        root=Path(configured) if configured else Path(source)/'runs/pi05-cycle'
        if not (root/'cycle.json').is_file():raise ValueError('No pi05 cycle configured')
        (root/'pause-requested').touch()
        return {'state':'pause_requested','scope':'The current round finishes before the cycle stops'}

    def _launch(self, command, cwd, env, item, output_argument=False):
        with self.lock:
            if any(job['status'] == 'running' for job in self.jobs.values()):
                raise ValueError('A run is already active')
            key = uuid.uuid4().hex
            output = self.root / key
            output.mkdir(parents=True)
            env.update(PHYSICALRSI_EVAL_OUTPUT=str(output), PYTHONUNBUFFERED='1')
            if output_argument:
                command = [*command, '--output', str(output/output_argument if isinstance(output_argument,str) else output)]
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, start_new_session=True)
            job = dict(id=key, kind=item.get('kind','evaluation'), title=item.get('title', item['id']), group=item['group'],
                       task=item.get('task', item['id'].removeprefix('baseline-')),
                       status='running', lines=[], output=str(output), returncode=None,
                       pid=process.pid, process_start=self._process_start(process.pid),command=command)
            self.jobs[key] = job
            self.processes[key] = process
            atomic_json(output/'run.json',{k:v for k,v in job.items() if k!='lines'})
        threading.Thread(target=self._consume, args=(job, process, env), daemon=True).start()
        return self.snapshot(key)

    def _consume(self, job, process, env):
        secrets = [v for k, v in env.items() if v and len(v) >= 8 and
                   any(word in k.upper() for word in ('TOKEN', 'API_KEY', 'PASSWORD', 'SECRET'))]
        with (Path(job['output']) / 'execution.log').open('w') as log:
            for line in process.stdout:
                line = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', line)
                for value in secrets:
                    line = line.replace(value, '[redacted]')
                log.write(line)
                log.flush()
                with self.lock:
                    job['lines'].append(line.rstrip())
                    job['lines'] = job['lines'][-2000:]
            code = process.wait()
            with self.lock:
                job['returncode'] = code
                job['status'] = 'completed' if code == 0 else 'failed'
                metadata = {k:v for k,v in job.items() if k not in ('lines', 'output')}
                atomic_json(Path(job['output'])/'run.json',metadata)

    @staticmethod
    def _process_start(pid):
        try:
            return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
        except (OSError,IndexError):return None

    def snapshot(self, key):
        with self.lock:
            job = self.jobs[key]
            if job['status']=='running' and key not in self.processes:
                start=self._process_start(job.get('pid'))
                if start is None or start != job.get('process_start'):
                    job['status']='needs_reconciliation'
                    job['note']='Process is gone without a confirmed exit receipt; inspect evidence before rerunning'
            return {**job, 'lines': list(job['lines'])}
