"""Read the recorded progress of a configured Dexjoco pi05 cycle."""
import json
import os
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def cycle_status():
    configured = os.environ.get('PHYSICALRSI_PI05_CYCLE')
    source = os.environ.get('DEXJOCO_ROOT')
    if not configured and not source:
        return None
    root = Path(configured) if configured else Path(source)/'runs/pi05-cycle'
    cycle = read_json(root/'cycle.json')
    if not cycle:
        return None
    rounds = cycle.get('rounds', [])
    current = rounds[-1] if rounds else {'round': 1, 'directory': 'round-0001'}
    folder = root/current['directory']
    collection = read_json(folder/'collection.json')
    training = read_json(folder/'pi05-training/training.json')
    evaluation = read_json(folder/'candidate-evaluation/evaluation.json')
    comparison = read_json(folder/'comparison.json')
    validation = {key: evaluation.get(key) for key in
                  ('layouts', 'completed', 'successful', 'success_rate', 'complete')} if evaluation else None
    if validation and comparison:
        validation.update(incumbent_success_rate=comparison['incumbent_success_rate'],
                          promoted=comparison['promoted'])
    workers = [read_json(path) for path in (folder/'training-workers').glob('rank-*.json')]
    if any(row.get('state') == 'failed' for row in workers):
        training['state'] = 'failed'
    metrics_path = folder/'pi05-training/metrics.jsonl'
    metrics = {}
    if metrics_path.exists():
        for line in reversed(metrics_path.read_text().splitlines()):
            try:
                metrics = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    logs = []
    for path in (root/'collection.log', root/'training-launch.log'):
        if path.exists():
            logs.extend(path.read_text().splitlines()[-12:])
    state = cycle.get('state')
    if state == 'training':
        state = training.get('state', state)
    if training.get('state') == 'failed':
        state = 'failed'
    return {'model': 'pi05', 'round': current['round'], 'state': state,
            'layouts': collection.get('layouts', 100), 'successful': collection.get('successful', 0),
            'devices': len(cycle.get('hosts', [])), 'steps': training.get('steps'),
            'metrics': metrics, 'validation': validation, 'lines': logs[-18:]}
