"""Launch and monitor a shared pi05 training run across the configured DSWs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycle', type=Path, required=True)
    parser.add_argument('--round', type=int, default=1)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--ssh-key', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--port', type=int, default=12402)
    args = parser.parse_args()
    hosts = json.loads((args.cycle/'cycle.json').read_text())['hosts']
    root = args.cycle/f'round-{args.round:04d}'
    data = root/'pi05-data'
    manifest = json.loads((data/'dataset.json').read_text())
    if len(manifest['episodes']) != 100:
        raise ValueError('Complete the 100-layout demonstration batch before training')
    run = root/'training-workers'
    run.mkdir(exist_ok=False)
    script = run/'pi05_train.py'
    shutil.copyfile(Path(__file__).with_name('pi05_train.py'), script)
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-i', str(args.ssh_key)]
    # Check the whole fleet before starting the collective, so busy hosts do not leave peers waiting.
    def check(host):
        result = subprocess.run(ssh+['root@'+host,
            'nvidia-smi --query-compute-apps=pid --format=csv,noheader'],
            capture_output=True, text=True, check=True, timeout=30)
        if result.stdout.strip():
            raise RuntimeError('GPU is busy: '+host)
    with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        list(pool.map(check, hosts))

    def worker(rank, host):
        command = ['env', '-u', 'HTTP_PROXY', '-u', 'HTTPS_PROXY', '-u', 'ALL_PROXY',
                   '-u', 'http_proxy', '-u', 'https_proxy', '-u', 'all_proxy',
                   'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'PYTHONUNBUFFERED=1',
                   'XLA_PYTHON_CLIENT_MEM_FRACTION=0.85',
                   'NCCL_SOCKET_IFNAME=eth1', 'NCCL_IB_DISABLE=1',
                   'LD_PRELOAD='+str(Path(args.python).parent.parent/'lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2'),
                   'OPENPI_DATA_HOME='+str(args.source/'checkpoints/download-cache'),
                   args.python, str(script), '--root', str(args.source), '--data', str(data),
                   '--output', str(root/'pi05-training'), '--checkpoint', str(args.checkpoint),
                   '--steps', str(args.steps), '--process-count', str(len(hosts)),
                   '--process-id', str(rank), '--coordinator', hosts[0]+':'+str(args.port)]
        for earlier in range(1, args.round):
            previous = args.cycle/f'round-{earlier:04d}'/'pi05-data'
            if (previous/'dataset.json').is_file():
                command.extend(['--replay-data', str(previous)])
        record = {'rank': rank, 'host': host, 'state': 'running'}
        state = run/f'rank-{rank:02d}.json'
        state.write_text(json.dumps(record))
        try:
            with (run/f'rank-{rank:02d}.log').open('w') as log:
                process = subprocess.Popen(ssh+['root@'+host, shlex.join(command)], stdout=log,
                                           stderr=subprocess.STDOUT)
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
            metrics = root/'pi05-training/metrics.jsonl'
            latest = metrics.read_text().splitlines()[-1] if metrics.exists() and metrics.stat().st_size else 'initializing'
            print(f'Active training workers: {len(pending)} · {latest}', flush=True)
    if any(row['state'] != 'completed' for row in completed):
        raise SystemExit('Training failed; inspect the per-worker logs')


if __name__ == '__main__':
    main()
