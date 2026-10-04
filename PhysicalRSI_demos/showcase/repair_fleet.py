"""Use agent-proposed teachers to recover failed layouts without replacing them."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from .teacher_design import propose
from .pi05_data import successful_episodes


def repair(cycle, number, source, python, ssh_key, memory):
    root = cycle/f'round-{number:04d}'
    expected = {row['seed'] for row in json.loads((root/'layouts.json').read_text())['layouts']}
    initial = [json.loads(path.read_text()) for path in root.glob('worker-*/data/episode-*/result.json')]
    if {row['seed'] for row in initial} != expected:
        raise ValueError('Finish recording every layout before teacher repair')
    failures = [row for row in initial if not row['success']]
    if not failures:
        return
    hosts = json.loads((cycle/'cycle.json').read_text())['hosts']
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-i', str(ssh_key)]
    def remote(host, command, log):
        with log.open('w') as stream:
            subprocess.run(ssh+['root@'+host, shlex.join(command)], stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
    executable = ['env', 'MUJOCO_GL=egl', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
                  'PYTHONUNBUFFERED=1', str(python)]
    for iteration in range(1, 4):
        experiment = root/f'teacher-revision-{iteration}'
        proposal = propose(failures, memory, experiment)
        search = experiment/'teacher_search.py'
        collector = experiment/'collect_dexjoco.py'
        shutil.copyfile(Path(__file__).with_name('teacher_search.py'), search)
        shutil.copyfile(Path(__file__).with_name('collect_dexjoco.py'), collector)
        jobs = [(failure['seed'], index) for failure in failures for index in range(len(proposal['candidates']))]
        def search_worker(rank, host):
            check = subprocess.run(ssh+['root@'+host,'nvidia-smi --query-compute-apps=pid --format=csv,noheader'],
                                   capture_output=True, text=True, check=True, timeout=30)
            if check.stdout.strip():
                raise RuntimeError('GPU is busy: '+host)
            for seed, index in jobs[rank::len(hosts)]:
                target = experiment/f'layout-{seed}-candidate-{index}'
                remote(host, [*executable, str(search), '--root', str(source),
                    '--proposal', str(experiment/'proposal.json'), '--layouts', str(root/'layouts.json'),
                    '--seed', str(seed), '--candidate-index', str(index), '--output', str(target)],
                    experiment/f'layout-{seed}-candidate-{index}.log')
        with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
            list(pool.map(lambda pair: search_worker(*pair), enumerate(hosts)))
        outcomes = [json.loads(path.read_text()) for path in experiment.glob('layout-*/candidate-*.json')]
        chosen = []
        for failure in failures:
            choices = [row for row in outcomes if row['seed'] == failure['seed'] and row['success']]
            if choices:
                chosen.append(min(choices, key=lambda row: (row['steps'], row['candidate_index'])))
        def collect_worker(rank, host):
            for result in chosen[rank::len(hosts)]:
                seed = result['seed']
                target = root/f'repair-{seed}'/f'agent-{iteration}'
                target.mkdir(parents=True, exist_ok=False)
                (target/'teacher.json').write_text(json.dumps(result['candidate'], indent=2))
                remote(host, [*executable, str(collector), '--root', str(source), '--output', str(target/'data'),
                    '--seed', str(seed), '--episodes', '1', '--chunk-size', '30', '--sample-every', '10',
                    '--layout-manifest', str(root/'layouts.json'), '--teacher-parameters', str(target/'teacher.json')],
                    target/'execution.log')
        with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
            list(pool.map(lambda pair: collect_worker(*pair), enumerate(hosts)))
        successful = {row['seed'] for row in initial if row['success']}
        for path in root.glob('repair-*/*/data/episode-*/result.json'):
            row = json.loads(path.read_text())
            if row['success']:
                successful.add(row['seed'])
        failures = [row for row in failures if row['seed'] not in successful]
        memory = {**memory, 'latest_teacher_trials': outcomes}
        (experiment/'outcomes.json').write_text(json.dumps(memory, indent=2))
        print(f'Recovered demonstrations: {len(successful)}/{len(expected)}', flush=True)
        if not failures:
            selected = successful_episodes(root)
            summary = json.loads((root/'collection.json').read_text())
            summary.update(successful=len(selected), missing_successful_layouts=[])
            (root/'collection.json').write_text(json.dumps(summary, indent=2))
            return
    raise RuntimeError('Some unchanged layouts still need successful demonstrations')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycle', type=Path, required=True)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--ssh-key', type=Path, required=True)
    parser.add_argument('--memory', type=Path)
    args = parser.parse_args()
    memory = json.loads(args.memory.read_text()) if args.memory else {}
    repair(args.cycle, args.round, args.source, args.python, args.ssh_key, memory)
