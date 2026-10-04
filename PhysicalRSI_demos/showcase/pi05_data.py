"""Prepare successful Dexjoco demonstrations for the single-arm pi05 adapter."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def rotation_vector_state(values):
    """Convert TCP xyz, wxyz quaternion and hand joints to xyz, rotvec and joints."""
    values = np.asarray(values)
    if values.shape[-1] != 23:
        raise ValueError('Expected a TCP quaternion and 16 hand joints')
    flat = values.reshape(-1, 23)
    rotation = Rotation.from_quat(flat[:, [4, 5, 6, 3]]).as_rotvec()
    return np.concatenate((flat[:, :3], rotation, flat[:, 7:]), axis=-1).reshape(
        (*values.shape[:-1], 22)).astype(np.float32)


def successful_episodes(round_root):
    """Keep one successful attempt per saved layout, including same-layout repairs."""
    expected = {row['seed'] for row in json.loads((round_root/'layouts.json').read_text())['layouts']}
    selected = {}
    paths = sorted(round_root.glob('worker-*/data/episode-*/result.json'))
    paths += sorted(round_root.glob('repair-*/*/data/episode-*/result.json'))
    for path in paths:
        result = json.loads(path.read_text())
        if result['seed'] in expected and result['success']:
            selected.setdefault(result['seed'], path.parent)
    missing = expected - selected.keys()
    if missing:
        raise ValueError(f'Layouts without successful demonstrations: {sorted(missing)}')
    return sorted(selected.items())


def prepare(round_root, output, norm_stats=None):
    episodes = successful_episodes(round_root)
    output.mkdir(parents=True, exist_ok=False)
    records, states, actions = [], [], []
    for seed, source in episodes:
        target = output/str(seed)
        target.mkdir()
        with np.load(source/'transitions.npz') as data:
            state = rotation_vector_state(data['states'])
            action = rotation_vector_state(data['actions'])
            if action.shape[1] != 30:
                raise ValueError('pi05 demonstrations require 30-step action chunks')
            for name, value in [('state', state), ('actions', action),
                                ('base', data['ego_images']), ('wrist', data['wrist_images'])]:
                np.save(target/(name+'.npy'), value)
            instruction = str(data['instruction'])
        states.append(state)
        actions.append(action.reshape(-1, 22))
        records.append({'seed': seed, 'directory': str(seed), 'samples': len(state),
                        'instruction': instruction, 'source': str(source)})
        print(f'Prepared layout {seed}: {len(state)} samples', flush=True)
    stats = {}
    for name, values in [('state', states), ('actions', actions)]:
        values = np.concatenate(values).astype(np.float64)
        stats[name] = {key: value.tolist() for key, value in {
            'mean': values.mean(axis=0), 'std': values.std(axis=0),
            'q01': np.quantile(values, .01, axis=0), 'q99': np.quantile(values, .99, axis=0)
        }.items()}
    normalization = json.loads(norm_stats.read_text()) if norm_stats else {'norm_stats': stats}
    (output/'norm_stats.json').write_text(json.dumps(normalization, indent=2))
    manifest = {'task': 'click_mouse', 'state_format': 'xyz_rotvec_hand',
                'action_dim': 22, 'action_horizon': 30, 'control_dt': .02,
                'episodes': records, 'samples': sum(row['samples'] for row in records)}
    (output/'dataset.json').write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--norm-stats', type=Path)
    args = parser.parse_args()
    prepare(args.round, args.output, args.norm_stats)
