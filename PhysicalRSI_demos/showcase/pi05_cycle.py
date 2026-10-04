"""Repeat new-layout collection, pi05 training and paired simulation evaluation."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .evaluate_fleet import aggregate, compare
from .pi05_data import prepare, successful_episodes
from .repair_fleet import repair
from PhysicalRSI.Embodied_Harness.memory.store import MemoryStore, Snapshot


def run(module, *arguments, python=sys.executable, check=True):
    return subprocess.run([str(python), '-m', 'PhysicalRSI_demos.showcase.'+module,
                           *map(str, arguments)], check=check)


def wait_for_training(root):
    """Attach to a locally launched training coordinator without restarting its workers."""
    while True:
        records = [json.loads(path.read_text()) for path in (root/'training-workers').glob('rank-*.json')]
        if not records:
            raise RuntimeError('Training has no worker records')
        if any(row['state'] == 'failed' for row in records):
            raise RuntimeError('A training worker failed; inspect its log before continuing')
        if all(row['state'] == 'completed' for row in records):
            break
        running = [row for row in records if row['state'] == 'running']
        live = False
        for row in running:
            try:
                command = Path(f"/proc/{row['pid']}/cmdline").read_bytes()
                live |= str(root/'training-workers/pi05_train.py').encode() in command
            except FileNotFoundError:
                pass
        if running and not live:
            raise RuntimeError('Recorded training workers are no longer attached to this host')
        metrics = root/'pi05-training/metrics.jsonl'
        latest = metrics.read_text().splitlines()[-1] if metrics.exists() and metrics.stat().st_size else 'initializing'
        print('Following pi05 training · '+latest, flush=True)
        time.sleep(30)
    record = json.loads((root/'pi05-training/training.json').read_text())
    if record['state'] != 'completed' or not (Path(record['checkpoint'])/'params').is_dir():
        raise RuntimeError('Training did not publish a complete checkpoint')
    return Path(record['checkpoint'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycle', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--collection-python', required=True)
    parser.add_argument('--training-python', required=True)
    parser.add_argument('--ssh-key', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=2, help='Number of rounds; zero continues until pause is requested')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--mirror', type=Path)
    args = parser.parse_args()
    args.cycle = args.cycle.resolve()
    args.source = args.source.resolve()
    if args.rounds < 0 or args.steps < 1:
        parser.error('Choose nonnegative rounds and positive training steps')
    config_path = args.cycle/'cycle.json'
    config = json.loads(config_path.read_text())
    rows = config.get('rounds', [])
    number = rows[-1]['round'] if rows and rows[-1]['state'] != 'completed' else len(rows)+1
    incumbent = config.get('incumbent', {'checkpoint': str(args.source/'checkpoints/pi05_base'), 'base_model': True})

    def phase(name, **fields):
        current = json.loads(config_path.read_text())
        existing = next((row for row in current['rounds'] if row['round'] == number), None)
        if existing is None:
            existing = {'round': number, 'directory': f'round-{number:04d}'}
            current['rounds'].append(existing)
        existing.update(state=name, **fields)
        current['state'] = name
        current['incumbent'] = incumbent
        config_path.write_text(json.dumps(current, indent=2))
        print(f'Round {number}: {name}', flush=True)

    completed = 0
    while args.rounds == 0 or completed < args.rounds:
        if (args.cycle/'pause-requested').exists():
            phase('paused')
            return
        root = args.cycle/f'round-{number:04d}'
        root.mkdir(exist_ok=True)
        layouts = root/'layouts.json'
        if not layouts.exists():
            phase('sampling_layouts')
            run('layouts', '--root', args.source, '--output', layouts, '--count', 100,
                '--seed', config['first_seed']+(number-1)*100, python=args.collection_python)
        if not (root/'collection.json').exists():
            phase('collecting')
            run('collect_fleet', '--cycle', args.cycle, '--round', number, '--python', args.collection_python,
                '--source', args.source, '--ssh-key', args.ssh_key, check=False)
        collection = json.loads((root/'collection.json').read_text())
        if collection['successful'] != 100:
            phase('repairing_demonstrations')
            memory_path = args.cycle/'teacher-memory.json'
            pointer = json.loads(memory_path.read_text()) if memory_path.exists() else None
            memory = Snapshot(Path(pointer['snapshot']).parent, pointer['revision']).read() if pointer else {}
            repair(args.cycle, number, args.source, args.collection_python, args.ssh_key, memory)
            if pointer:
                outcomes = sorted(root.glob('teacher-revision-*/outcomes.json'))
                if outcomes:
                    snapshot = MemoryStore(Path(pointer['snapshot']).parent).snapshot({
                        **json.loads(outcomes[-1].read_text()), 'parent_revision': pointer['revision']})
                    memory_path.write_text(json.dumps({'snapshot': str(snapshot.root/(snapshot.revision+'.json')),
                                                       'revision': snapshot.revision}, indent=2))
        episodes = successful_episodes(root)
        if len(episodes) != 100:
            raise RuntimeError('Every round must retain all 100 sampled layouts')
        (root/'successful-demonstrations.json').write_text(json.dumps({'layouts': len(episodes),
            'episodes': [{'seed': seed, 'directory': str(path)} for seed, path in episodes]}, indent=2))
        data = root/'pi05-data'
        if not data.exists():
            phase('preparing_data')
            fixed_stats = args.cycle/'round-0001/pi05-data/norm_stats.json' if number > 1 else None
            prepare(root, data, fixed_stats)
        phase('training', layouts=100, successful_demonstrations=100)
        if not (root/'training-workers').exists():
            run('train_fleet', '--cycle', args.cycle, '--round', number, '--source', args.source,
                '--python', args.training_python, '--checkpoint', Path(incumbent['checkpoint'])/'params',
                '--ssh-key', args.ssh_key, '--steps', args.steps)
        checkpoint = wait_for_training(root)
        evaluation_layouts = root/'evaluation-layouts.json'
        if not evaluation_layouts.exists():
            phase('sampling_evaluation_layouts')
            run('layouts', '--root', args.source, '--output', evaluation_layouts, '--count', 100,
                '--seed', 2000000+(number-1)*100, python=args.collection_python)

        def evaluate(name, checkpoint, base_model=False):
            output = root/name
            summary = output/'evaluation.json'
            if summary.is_file() and json.loads(summary.read_text())['complete']:
                return json.loads(summary.read_text())
            command = ['--cycle', args.cycle, '--source', args.source, '--python', args.training_python,
                       '--checkpoint', checkpoint, '--data', data, '--layouts', evaluation_layouts,
                       '--output', output, '--ssh-key', args.ssh_key]
            if base_model:
                command.append('--base-model')
            run('evaluate_fleet', *command)
            return json.loads(summary.read_text())

        phase('evaluating')
        initial = aggregate([root/'base-evaluation-pilot', root/'base-evaluation-00', root/'base-evaluation-01'], evaluation_layouts)
        before = initial if number == 1 and incumbent['base_model'] and initial['complete'] else evaluate(
            'incumbent-evaluation', Path(incumbent['checkpoint']), incumbent['base_model'])
        after = evaluate('candidate-evaluation', checkpoint)
        comparison = compare(before, after)
        comparison.update(incumbent_checkpoint=incumbent['checkpoint'], candidate_checkpoint=str(checkpoint))
        (root/'comparison.json').write_text(json.dumps(comparison, indent=2))
        if comparison['promoted']:
            incumbent = {'checkpoint': str(checkpoint), 'base_model': False, 'round': number}
        phase('completed', comparison=comparison)
        if args.mirror:
            shutil.copytree(root, args.mirror/root.name, dirs_exist_ok=True, copy_function=shutil.copyfile,
                            ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copyfile(config_path, args.mirror/'cycle.json')
        number += 1
        completed += 1


if __name__ == '__main__':
    main()
