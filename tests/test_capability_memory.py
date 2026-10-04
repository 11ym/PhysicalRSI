"""Grounded capability metadata must reach the actual agent request unchanged."""
import copy
import hashlib
import json

import pytest

from PhysicalRSI_baselines.robodojo import capability_memory
from PhysicalRSI_baselines.robodojo.agent_planner import AgentPlanner


def _entry(name):
    return capability_memory.attach_capability_memory(
        {'description': 'Declared execution capability', 'steps': ['memory.snapshot', name]}, name)


def test_library_covers_all_release_skills_and_preserves_unknown_extensions():
    library = capability_memory._library()
    assert len(library) == 24
    assert {'pi05', 'pi05-sparse-memory', 'classify-objects', 'classify-objects-by-language'} <= library.keys()
    custom = {'description': 'Custom skill', 'steps': ['memory.snapshot', 'custom']}
    assert capability_memory.attach_capability_memory(custom, 'custom') == custom


def test_conditions_remain_advisory_and_do_not_filter_catalogue():
    catalogue = {'cover': _entry('cover-blocks')}
    decorated = capability_memory.catalogue_with_observations(catalogue, {})
    assert list(decorated) == ['cover']
    assert decorated['cover']['steps'] == catalogue['cover']['steps']
    assert all(item['status'] == 'unknown' for item in decorated['cover']['observed_conditions'])
    assert all(item['advisory'] for item in decorated['cover']['observed_conditions'])


def test_performance_prior_is_rejected_even_with_recomputed_identity():
    entry = copy.deepcopy(capability_memory._library()['pi05'])
    entry['score_prior'] = 1.0
    payload = {key: value for key, value in entry.items() if key != 'capability_id'}
    entry['capability_id'] = 'cap-' + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    with pytest.raises(ValueError, match='performance priors'):
        capability_memory._validate(entry)


def test_actual_agent_prompt_contains_grounded_binding_and_observation_conditions(monkeypatch):
    monkeypatch.setenv('PHYSICALRSI_AGENT_API_KEY', 'test-only')
    captured = {}
    class Reply:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            return json.dumps({'choices': [{'message': {'content': json.dumps({
                'composition': 'language', 'rationale': 'Explicit destination bindings are required.'})}}]}).encode()
    def respond(request, **kwargs):
        captured.update(json.loads(request.data))
        return Reply()
    monkeypatch.setattr('urllib.request.urlopen', respond)
    planner = AgentPlanner(endpoint='https://example.invalid/test', model='test-model')
    catalogue = {'geometric': _entry('classify-objects'), 'language': _entry('classify-objects-by-language')}
    planner.plan(instruction='Put watches into the left basket.',
                 observation={'instruction': 'Put watches into the left basket.'},
                 memory={}, previous_plan=None, compositions=catalogue)
    context = json.loads(captured['messages'][1]['content'][0]['text'])
    entries = context['available_compositions']
    generic = entries['geometric']['capability_memory']
    language = entries['language']['capability_memory']
    assert 'no language-specified' in generic['instruction_binding']['destination_assignment']
    assert 'Parse explicit category mappings' in language['instruction_binding']['destination_assignment']
    assert generic['operation_memory'] and language['operation_memory']
    assert entries['language']['observed_conditions'][0]['status'] == 'satisfied'
    assert 'score priors' in captured['messages'][0]['content']
    assert list(entries) == list(catalogue)
