import io
import json

import pytest

from PhysicalRSI_baselines.robodojo.agent_planner import AgentPlanner, agent_observation


def test_planner_observation_excludes_evaluation_and_hidden_state():
    observation = {
        'instruction': 'Move the visible object', 'answer': 'hidden', 'reward': 1,
        'layout_seed': 42, 'task_name': 'private',
        'state': {'left_arm_joint_state': [0] * 6, 'object_pose': [1, 2, 3]},
        'vision': {'head': {'color': 'rgb-data', 'depth': 'hidden-depth',
                            'answer': 'hidden-label'}},
    }
    assert agent_observation(observation) == {
        'instruction': 'Move the visible object',
        'state': {'left_arm_joint_state': [0] * 6},
        'vision': {'head': {'color': 'rgb-data'}},
    }
    assert observation['state']['object_pose'] == [1, 2, 3]


@pytest.fixture
def planner_request(monkeypatch):
    monkeypatch.setenv('PHYSICALRSI_AGENT_API_KEY', 'test-key')
    captured = []

    def run(plan):
        def respond(request, **kwargs):
            captured.append(json.loads(request.data))
            return io.BytesIO(json.dumps({'choices': [{'message': {
                'content': json.dumps(plan)}}]}).encode())
        monkeypatch.setattr('urllib.request.urlopen', respond)
        planner = AgentPlanner(endpoint='https://example.invalid', model='test')
        return planner.plan(
            instruction='Move the observed object',
            observation={'task_name': 'excluded', 'vision': {}, 'instruction': 'task data'},
            memory={'capabilities': 'test contracts'}, previous_plan=None,
            operation_program={'input': 'observation', 'output': 'action', 'operations': {
                'perceive': {'input': 'observation', 'output': 'estimate'},
                'control': {'input': 'estimate', 'output': 'action'}}})

    return run, captured


def test_plan_is_executable_contract_sequence(planner_request):
    run, captured = planner_request
    plan = {'operations': ['perceive', 'control'], 'rationale': 'Perceive before control'}
    assert run(plan) == plan
    context = json.loads(captured[0]['messages'][1]['content'][0]['text'])
    assert 'task_name' not in context['observation']
    assert set(context['operation_program']['operations']) == {'perceive', 'control'}


@pytest.mark.parametrize('operations', [
    ['shell'], ['perceive', 'perceive', 'control'], ['control', 'perceive'],
    ['perceive'], [], [None], 'perceive',
])
def test_invalid_sequence_is_rejected(planner_request, operations):
    run, _ = planner_request
    with pytest.raises(ValueError):
        run({'operations': operations, 'rationale': 'test'})
