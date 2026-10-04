"""Ask the local agent for bounded mouse-teacher revisions after a failed rollout."""
import argparse
import json
import math
from pathlib import Path

from .local_model import LocalModel


LIMITS = {'yaw_scale': (-.5, 1.5), 'translation_scale': (.9, 1.1),
          'reanchor_step': (300, 500), 'reanchor_yaw_scale': (0., 1.5),
          'recovery_start_step': (60, 250), 'recovery_yaw_scale': (0., 1.5)}


def propose(failures, memory, output):
    parameters = {key: {'type': 'integer' if key.endswith('_step') else 'number',
                         'minimum': bounds[0], 'maximum': bounds[1]}
                  for key, bounds in LIMITS.items()}
    schema = {'type': 'object', 'properties': {
        'reason': {'type': 'string'},
        'candidates': {'type': 'array', 'minItems': 1, 'maxItems': 16,
            'items': {'type': 'object', 'properties': parameters,
                      'required': list(parameters), 'additionalProperties': False}}},
        'required': ['reason', 'candidates'], 'additionalProperties': False}
    prompt = (
        'Design small revisions of a simulator-assisted demonstration teacher for click_mouse. '
        'Do not use tools or execute commands. Return the requested JSON. '
        'The goal is to place the mouse on its pad and press the left button. '
        'Keep every failed layout unchanged. This is training-data generation; the student later '
        'receives only RGB, robot state and instruction. '
        'The teacher transforms a recorded absolute TCP trajectory around the initial mouse pose: '
        'yaw_scale scales the relative rotation; translation_scale scales the relative TCP positions. '
        'At reanchor_step it realigns the remaining trajectory to the live mouse pose with '
        'reanchor_yaw_scale. If the first attempt fails, it replays from recovery_start_step '
        'using recovery_yaw_scale. The reference press phase begins at step 450. '
        'Use the observed failure phase and previous results to propose at most 16 distinct candidates. '
        'Prefer small controlled changes. Do not claim success until a candidate is executed.\n'
        + json.dumps({'failures': failures, 'memory': memory}, ensure_ascii=False))
    output.mkdir(parents=True, exist_ok=False)
    (output/'request.txt').write_text(prompt)
    reply = LocalModel().structured(prompt, schema)
    candidates = reply.get('candidates', [])
    if not 1 <= len(candidates) <= 16:
        raise ValueError('Expected one to sixteen teacher candidates')
    for index, candidate in enumerate(candidates):
        if set(candidate) != set(LIMITS):
            raise ValueError('Unexpected teacher parameters')
        for key, value in candidate.items():
            if type(value) not in (int, float) or not math.isfinite(value) or not LIMITS[key][0] <= value <= LIMITS[key][1]:
                raise ValueError('Invalid teacher parameter: '+key)
            if key.endswith('_step') and type(value) is not int:
                raise ValueError('Teacher step must be an integer')
        candidate['name'] = f'candidate_{index:02d}'
        candidate['hold_steps'] = 30
    (output/'proposal.json').write_text(json.dumps(reply, indent=2))
    return reply


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--failures', type=Path, nargs='+', required=True)
    parser.add_argument('--memory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    failures = [json.loads(path.read_text()) for path in args.failures]
    memory = json.loads(args.memory.read_text()) if args.memory else {}
    result = propose(failures, memory, args.output)
    print(f"Agent proposed {len(result['candidates'])} teacher revisions", flush=True)
