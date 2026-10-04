"""Evaluate a checkpoint across DSWs and aggregate every saved layout."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from .data_splits import check_disjoint


def aggregate(directories, layouts):
    expected = {row['seed'] for row in json.loads(layouts.read_text())['layouts']}
    episodes = {}
    for directory in directories:
        for path in directory.glob('episode-*/result.json'):
            result = json.loads(path.read_text())
            seed = result['seed']
            if seed not in expected:
                raise ValueError('Unexpected evaluation layout: '+str(seed))
            if seed in episodes:
                raise ValueError('Duplicate evaluation layout: '+str(seed))
            episodes[seed] = result
    missing = sorted(expected-episodes.keys())
    success = sum(bool(row['success']) for row in episodes.values())
    return {'layouts': len(expected), 'completed': len(episodes), 'successful': success,
            'success_rate': success/len(episodes) if episodes else None,
            'complete': not missing, 'missing_layouts': missing,
            'episodes': [episodes[seed] for seed in sorted(episodes)]}


def compare(incumbent, candidate):
    if not incumbent['complete'] or not candidate['complete']:
        raise ValueError('Complete both evaluations before choosing a checkpoint')
    before = {row['seed']: bool(row['success']) for row in incumbent['episodes']}
    after = {row['seed']: bool(row['success']) for row in candidate['episodes']}
    if set(before) != set(after):
        raise ValueError('Compare checkpoints on the same saved layouts')
    gained = [seed for seed in sorted(before) if after[seed] and not before[seed]]
    lost = [seed for seed in sorted(before) if before[seed] and not after[seed]]
    return {'layouts': len(before), 'incumbent_success_rate': incumbent['success_rate'],
            'candidate_success_rate': candidate['success_rate'], 'gained': gained, 'lost': lost,
            'promoted': len(gained) > len(lost)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycle', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--layouts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ssh-key', type=Path, required=True)
    parser.add_argument('--base-model', action='store_true')
    args = parser.parse_args()
    training_manifests = list(args.cycle.glob('round-*/layouts.json'))
    if not training_manifests:
        raise ValueError('No training layout manifests found')
    check_disjoint(training_manifests, args.layouts)
    hosts = json.loads((args.cycle/'cycle.json').read_text())['hosts']
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-i', str(args.ssh_key)]
    def check(host):
        result = subprocess.run(ssh+['root@'+host, 'nvidia-smi --query-compute-apps=pid --format=csv,noheader'],
                                capture_output=True, text=True, check=True, timeout=30)
        if result.stdout.strip():
            raise RuntimeError('GPU is busy: '+host)
    with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        list(pool.map(check, hosts))
    args.output.mkdir(parents=True, exist_ok=False)
    script = args.output/'pi05_evaluate.py'
    shutil.copyfile(Path(__file__).with_name('pi05_evaluate.py'), script)
    def worker(rank, host):
        command = ['env', 'MUJOCO_GL=egl', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
                   'PYTHONUNBUFFERED=1', 'XLA_PYTHON_CLIENT_MEM_FRACTION=0.75',
                   'OPENPI_DATA_HOME='+str(args.source/'checkpoints/download-cache'),
                   args.python, str(script), '--root', str(args.source), '--checkpoint', str(args.checkpoint),
                   '--data', str(args.data), '--layouts', str(args.layouts),
                   '--output', str(args.output/f'worker-{rank:02d}'),
                   '--rank', str(rank), '--workers', str(len(hosts))]
        if args.base_model:
            command.append('--base-model')
        state = args.output/f'worker-{rank:02d}.json'
        record = {'rank': rank, 'host': host, 'state': 'running'}
        state.write_text(json.dumps(record))
        try:
            with (args.output/f'worker-{rank:02d}.log').open('w') as log:
                process = subprocess.Popen(ssh+['root@'+host, shlex.join(command)], stdout=log, stderr=subprocess.STDOUT)
                record['pid'] = process.pid
                state.write_text(json.dumps(record))
                code = process.wait()
            record.update(state='completed' if code == 0 else 'failed', returncode=code)
        except Exception as error:
            record.update(state='failed', error=str(error))
        state.write_text(json.dumps(record, indent=2))
        return record
    with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        pending = {pool.submit(worker, rank, host) for rank, host in enumerate(hosts)}
        completed = []
        while pending:
            done, pending = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
            completed.extend(future.result() for future in done)
            try:
                summary = aggregate(list(args.output.glob('worker-*/')), args.layouts)
            except json.JSONDecodeError:
                continue
            (args.output/'evaluation.json').write_text(json.dumps(summary, indent=2))
            print(f"Evaluated {summary['completed']}/{summary['layouts']} layouts · successful {summary['successful']}", flush=True)
    summary = aggregate(list(args.output.glob('worker-*/')), args.layouts)
    (args.output/'evaluation.json').write_text(json.dumps(summary, indent=2))
    if any(row['state'] != 'completed' for row in completed) or not summary['complete']:
        raise SystemExit('Evaluation is incomplete; inspect the per-worker logs')


if __name__ == '__main__':
    main()
