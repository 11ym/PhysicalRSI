"""Execute proposed teacher revisions on an unchanged training layout."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--proposal', type=Path, required=True)
    parser.add_argument('--layouts', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-index', type=int)
    args = parser.parse_args()
    sys.path[:0] = [str(args.root), str(args.root/'dexjoco')]
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import numpy as np
    from rsi.click_mouse_datagen import Candidate, run_episode
    expected = next(row for row in json.loads(args.layouts.read_text())['layouts'] if row['seed'] == args.seed)
    candidates = list(enumerate(json.loads(args.proposal.read_text())['candidates']))
    if args.candidate_index is not None:
        candidates = [candidates[args.candidate_index]]
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    def verify_layout(env, action):
        if action is None:
            np.testing.assert_allclose(env.data.qpos, expected['qpos'], rtol=0, atol=1e-9)
    for index, parameters in candidates:
        result = run_episode(Candidate(**parameters), args.seed, 0, observe=verify_layout)
        result['candidate_index'] = index
        (args.output/f'candidate-{index:02d}.json').write_text(json.dumps(result, indent=2))
        results.append(result)
        print(f"Candidate {index}: success={result['success']} steps={result['steps']}", flush=True)
    successful = sorted((row for row in results if row['success']), key=lambda row: (row['steps'], row['candidate_index']))
    (args.output/'selection.json').write_text(json.dumps({'seed': args.seed,
        'selected': successful[0] if successful else None, 'attempts': len(results)}, indent=2))


if __name__ == '__main__':
    main()
