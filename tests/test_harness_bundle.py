import json
import os
from pathlib import Path
import subprocess
import pytest
from PhysicalRSI_baselines.robodojo.agent_planner import AgentLoop, AgentPlanner
from PhysicalRSI_baselines.robodojo.harness_bundle import install, validate

ROOT = Path(__file__).parents[1] / 'PhysicalRSI_baselines/robodojo/releases/robodojo-v1'


def test_bundle_validates_and_installs(tmp_path):
    manifest = validate(ROOT)
    assert manifest['protocol_mode'] == 'task_aware_skill_library'
    assert manifest['components']['skills'] == [f'skills/{name}/descriptor.json' for name in ('pi05', 'pi05-sparse-memory', 'code-policy')]
    assert not (ROOT / 'adapters').exists()
    (tmp_path / 'model_template.py').write_text('')
    target = install(tmp_path)
    assert (target / 'agent_planner.py').read_bytes() == (ROOT.parents[1] / 'agent_planner.py').read_bytes()
    assert not (target / 'compatibility').exists()


def test_draft_cannot_silently_evaluate_an_external_policy():
    env = dict(os.environ)
    env.pop('OPENAI_API_KEY', None)
    env.pop('PHYSICALRSI_AGENT_API_KEY', None)
    result = subprocess.run(['bash', str(ROOT / 'eval.sh')], capture_output=True, text=True, env=env)
    assert result.returncode != 0
    assert 'XPOLICYLAB_ROOT' in result.stderr
    assert json.loads((ROOT / 'manifest.json').read_text())['evaluation_ready'] is False


def test_agent_key_is_mandatory(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('PHYSICALRSI_AGENT_API_KEY', raising=False)
    with pytest.raises(RuntimeError, match='required'):
        AgentPlanner()


class Planner:
    def __init__(self):
        self.calls = []
        self.fail = False

    def plan(self, **context):
        if self.fail:
            raise RuntimeError('unavailable')
        self.calls.append(context)
        return {'subgoal': 'grasp the observed object'}


def test_supervision_preserves_inputs_and_calls_once_per_100_chunks():
    planner = Planner()
    receipts = []
    loop = AgentLoop(planner, {}, receipts.append, replan_interval=100, max_calls_per_episode=3)
    for i in range(201):
        obs = dict(env_idx=0, instruction='place the object', state=[i])
        prepared = loop.prepare([obs])
        assert prepared[0] is obs
        assert obs['instruction'] == 'place the object'
        loop.consume([0])
    assert len(planner.calls) == 3
    assert [call['observation']['state'] for call in planner.calls] == [[0], [100], [200]]
    assert planner.calls[1]['previous_plan'] == {'subgoal': 'grasp the observed object'}
    assert len(receipts) == 3
    with pytest.raises(RuntimeError):
        loop.consume([0])
    loop.reset()
    loop.prepare([dict(env_idx=0, instruction='place the object')])
    assert len(planner.calls) == 4


def test_failed_replan_blocks_actions_and_batch_environments_are_independent():
    planner = Planner()
    loop = AgentLoop(planner, {}, lambda value: None, replan_interval=1, max_calls_per_episode=2)
    rows = [dict(env_idx=i, instruction='place object') for i in (0, 1)]
    loop.prepare(rows)
    loop.consume([1])
    loop.consume([0])
    planner.fail = True
    with pytest.raises(RuntimeError, match='unavailable'):
        loop.prepare(rows)
    with pytest.raises(RuntimeError):
        loop.consume([0])


def test_default_supervision_is_capped_at_one_call_per_episode():
    planner = Planner()
    loop = AgentLoop(planner, {}, lambda receipt: None)
    obs = dict(env_idx=0, instruction="place object")
    for _ in range(501):
        assert loop.prepare([obs])[0] is obs
        loop.consume([0])
    assert len(planner.calls) == 1
    loop.reset()
    loop.prepare([obs])
    assert len(planner.calls) == 2
