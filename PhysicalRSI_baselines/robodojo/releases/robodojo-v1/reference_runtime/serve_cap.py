"""Launch the packaged frozen CAP implementation without extracting shared runtimes."""
import argparse, json, os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', required=True)
    p.add_argument('--seed', default='0')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'runtime/cap_service_registry.json').read_text())[a.task]
    python = os.environ.get('KAI_CAP_PYTHON', config['python'])
    os.environ['KAI_POLICY_ROOT'] = str(root)
    os.execv(python, [python, '-B', '-u', str(root / 'cap' / a.task / 'evaluator/manage_services.py'),
                     '--serve', '--config', a.task, '--root', str(root / 'cap' / a.task / 'bundle/frozen'),
                     '--output', str(a.output.resolve() / 'services'),
                     '--xpolicylab', str(root / 'cap' / a.task / 'XPolicyLab'),
                     '--policy-port', a.port, '--host', '127.0.0.1', '--seed', a.seed, '--python', python])

if __name__ == '__main__':
    main()
