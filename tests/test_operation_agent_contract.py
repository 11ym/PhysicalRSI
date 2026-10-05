import io
import json

import pytest

from PhysicalRSI_baselines.robodojo import agent_planner


COMPOSITIONS = {
    'library': {'steps': ['memory.snapshot', 'memory-program-library']},
    'vla': {'steps': ['memory.snapshot', 'pi05']},
}
REGISTRY = {'input': 'observation', 'output': 'actions', 'operations': {
    'execute': {'input': 'observation', 'output': 'actions'},
}}


def invoke(monkeypatch, plan, compositions=COMPOSITIONS):
    planner = agent_planner.AgentPlanner.__new__(agent_planner.AgentPlanner)
    planner.api_key = 'local-test'
    planner.endpoint = 'https://example.invalid'
    planner.model = 'test'
    requests = []

    def respond(request, **kwargs):
        requests.append(json.loads(request.data))
        return io.BytesIO(json.dumps({'choices': [{'message': {
            'content': json.dumps(plan)}}]}).encode())

    monkeypatch.setattr(agent_planner.urllib.request, 'urlopen', respond)
    result = planner.plan(instruction='move an object', observation={}, memory={},
                          previous_plan=None, compositions=compositions,
                          operation_program=REGISTRY)
    return result, requests


@pytest.mark.parametrize('composition,operations', [('vla', []), ('library', ['execute'])])
def test_valid_execution_modes(monkeypatch, composition, operations):
    plan = dict(composition=composition, operations=operations, rationale='Declared contract applies.')
    assert invoke(monkeypatch, plan)[0] == plan


@pytest.mark.parametrize('composition,operations', [
    ('library', []), ('unknown', []), ('unknown', ['execute']),
    (' library ', ['execute']), (None, []), ([], []),
])
def test_invalid_agent_choice_is_rejected_without_substitution(monkeypatch, composition, operations):
    plan = dict(composition=composition, operations=operations, rationale='No declared capability matches.')
    with pytest.raises(ValueError):
        invoke(monkeypatch, plan)
    assert plan['composition'] == composition
    assert plan['operations'] == operations


def test_independent_skill_cannot_receive_operations(monkeypatch):
    with pytest.raises(ValueError):
        invoke(monkeypatch, dict(composition='vla', operations=['execute'], rationale='Contract.'))


def test_operation_only_prompt_matches_response_contract(monkeypatch):
    plan = dict(operations=['execute'], rationale='Contract.')
    result, requests = invoke(monkeypatch, plan, compositions=None)
    assert result == plan
    assert 'exactly operations' in requests[0]['messages'][0]['content']
    assert 'exactly composition' not in requests[0]['messages'][0]['content']


def test_operation_prompt_exposes_capability_memory_not_source_binding(monkeypatch):
    plan = dict(composition='library', operations=['execute'], rationale='Contract.')
    _, requests = invoke(monkeypatch, plan)
    prompt = requests[0]['messages'][1]['content'][0]['text']
    assert 'operation_program' in prompt
    assert 'source_name' not in prompt
