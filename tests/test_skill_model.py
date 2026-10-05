import json
import pytest
from PhysicalRSI_core.contracts import Operation
from PhysicalRSI_baselines.robodojo import skill_model
from PhysicalRSI_baselines.robodojo.execution_skills import ExecutionSkill


class Implementation:
    def __init__(self):
        self.observed = []
        self.closed = False
    def update_obs_batch(self, rows):
        self.observed.extend(rows)
    def get_action_batch(self, indices):
        return [[{'value': i}] for i in indices]
    def reset(self):
        self.observed.clear()
    def close(self):
        self.closed = True


def configured(tmp_path, monkeypatch):
    config = {'schema': 'physicalrsi.skill-library/v1', 'memory': {'lessons': []},
              'compositions': ['memory-guided-manipulation'],
              'skills': {name: {'name': name, 'configuration': {}} for name in ['pi05']}}
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    monkeypatch.setenv('PHYSICALRSI_SKILL_CONFIG', str(path))
    monkeypatch.setenv('PHYSICALRSI_EVAL_OUTPUT', str(tmp_path / 'output'))
    decisions = []
    implementations = []
    class Planner:
        def __init__(self, **kwargs):
            pass
        def plan(self, **kwargs):
            decisions.append(kwargs)
            return {'composition': 'memory-guided-manipulation', 'rationale': 'observed manipulation'}
    def loader(descriptor, deployment):
        implementation = Implementation()
        implementations.append(implementation)
        return ExecutionSkill(descriptor['name'], implementation, 'revision')
    monkeypatch.setattr(skill_model, 'AgentPlanner', Planner)
    monkeypatch.setattr(skill_model, 'load_skill', loader)
    return skill_model.Model({}), decisions, implementations


def test_memory_choice_reaches_execution_without_mutating_inputs(tmp_path, monkeypatch):
    model, decisions, implementations = configured(tmp_path, monkeypatch)
    row = {'env_idx': 3, 'instruction': 'move object', 'state': [1, 2]}
    for _ in range(100):
        model.update_obs_batch([row])
        assert model.get_action_batch([3]) == [[{'value': 3}]]
    assert len(decisions) == 1
    assert all(value is row for value in implementations[0].observed)
    assert model.sessions[3]['skill'].name == 'pi05'
    assert isinstance(model.sessions[3]['operation'], Operation)
    assert len(model.sessions[3]['operation'].children) == 2
    assert model.memory.read()['task_specific_policy_memory']['run_context'] == {'lessons': []}
    assert 'capabilities' in model.memory.read()['task_specific_policy_memory']['library']
    model.reset()
    assert not implementations[0].closed
    assert len(implementations) == 1
    model.update_obs_batch([row])
    assert len(decisions) == 2
    model.close()


def test_memory_tampering_blocks_skill_execution(tmp_path, monkeypatch):
    model, decisions, implementations = configured(tmp_path, monkeypatch)
    model.update_obs({'env_idx': 0, 'instruction': 'move'})
    (model.memory.root / (model.memory.revision + '.json')).write_text('{}')
    with pytest.raises(ValueError, match='Memory snapshot changed'):
        model.get_action()
    assert len(implementations[0].observed) == 1
    with pytest.raises(RuntimeError):
        model.get_action()
    model.close()


def test_agent_failure_cannot_fall_back_to_any_skill(tmp_path, monkeypatch):
    model, _, implementations = configured(tmp_path, monkeypatch)
    def fail(**kwargs):
        raise RuntimeError('API unavailable')
    model.planner.plan = fail
    with pytest.raises(RuntimeError, match='API unavailable'):
        model.update_obs({'env_idx': 0, 'instruction': 'move'})
    assert len(implementations) == 1
    assert not implementations[0].observed
    with pytest.raises(RuntimeError):
        model.get_action()
    model.close()


def test_intermediate_observations_reach_skill_without_duplicate_history(tmp_path, monkeypatch):
    model, decisions, implementations = configured(tmp_path, monkeypatch)
    rows = [dict(env_idx=0, instruction='move', state=[i]) for i in range(4)]
    model.update_obs(rows[0])
    model.get_action()
    for row in rows[1:]:
        model.update_obs(row)
    assert [row['state'] for row in implementations[0].observed] == [row['state'] for row in rows]
    model.get_action()
    assert [row['state'] for row in implementations[0].observed] == [row['state'] for row in rows]
    assert len(decisions) == 1
    model.close()


@pytest.mark.parametrize('name,composition', [('pi05-sparse-memory', 'memory-guided-visual-history'), ('code-policy', 'memory-guided-primitive-program')])
def test_memory_driven_choice_loads_skill(tmp_path, monkeypatch, name, composition):
    model, _, _ = configured(tmp_path, monkeypatch)
    model.close()
    path = tmp_path / 'config.json'
    config = json.loads(path.read_text())
    config['compositions'].append(composition)
    config['skills'][name] = {'name': name, 'configuration': {}}
    path.write_text(json.dumps(config))
    model = skill_model.Model({})
    model.planner.plan = lambda **kwargs: {'composition': composition, 'rationale': 'Use the capability described in memory'}
    model.update_obs({'instruction': 'remember the object'})
    assert model.sessions[0]['skill'].name == name
    assert model.get_action() == [{'value': 0}]
    model.close()


def test_unregistered_choice_fails_before_loading(tmp_path, monkeypatch):
    model, _, implementations = configured(tmp_path, monkeypatch)
    model.planner.plan = lambda **kwargs: {'composition': 'unknown', 'rationale': 'test'}
    with pytest.raises(ValueError, match='registered skill'):
        model.update_obs({'instruction': 'move'})
    assert len(implementations) == 1
    assert not implementations[0].observed
    model.close()


def test_named_code_skill_is_selected_from_frozen_catalog(tmp_path, monkeypatch):
    model, _, _ = configured(tmp_path, monkeypatch)
    model.close()
    path = tmp_path / 'config.json'
    config = json.loads(path.read_text())
    config['skills']['align-and-insert'] = {
        'name': 'align-and-insert', 'implementation': 'code-policy', 'configuration': {}}
    config['composition_definitions'] = {'memory-guided-insertion': {
        'description': 'Align observed geometry, then execute insertion primitives.',
        'steps': ['memory.snapshot', 'align-and-insert']}}
    config['compositions'].append('memory-guided-insertion')
    path.write_text(json.dumps(config))
    model = skill_model.Model({})
    model.planner.plan = lambda **kwargs: {
        'composition': 'memory-guided-insertion', 'rationale': 'Observed geometry requires alignment'}
    model.update_obs({'instruction': 'insert the object'})
    assert model.sessions[0]['skill'].name == 'align-and-insert'
    assert model.get_action() == [{'value': 0}]
    model.close()


def test_registered_operation_memory_is_exposed_and_bound_before_load(tmp_path, monkeypatch):
    from PhysicalRSI_core.infra.storage import digest
    from PhysicalRSI.Embodied_Harness.skills.episode_plan import apply_launch_plan
    old, decisions, _ = configured(tmp_path, monkeypatch)
    old.close()
    memory = {'schema': 'physicalrsi.skill-program-memory/v1', 'programs': {
        'advance': {'steps': [{'operation': 'motion.advance', 'revision': 'frozen-v1'}]}}}
    path = tmp_path / 'config.json'
    config = json.loads(path.read_text())
    config['skills']['pi05']['configuration']['operation_memories'] = {digest(memory): memory}
    path.write_text(json.dumps(config))
    loaded = []
    def loader(descriptor, deployment):
        selected = apply_launch_plan(memory, descriptor['configuration']['environment'])
        loaded.append(selected)
        return ExecutionSkill(descriptor['name'], Implementation(), 'revision')
    monkeypatch.setattr(skill_model, 'load_skill', loader)
    model = skill_model.Model({})
    model.update_obs({'instruction': 'move'})
    model.get_action()
    model.update_obs({'instruction': 'move'})
    model.get_action()
    assert len(decisions) == 1
    exposed = decisions[0]['compositions']['memory-guided-manipulation']['operation_memory']
    assert exposed[0]['programs']['advance']['steps'] == memory['programs']['advance']['steps']
    assert loaded[0]['programs'] == memory['programs']
    evidence = next((model.root / 'episodes').glob('*/episode.json'))
    assert json.loads(evidence.read_text())['operation_plan_revision'] == loaded[0]['agent_plan']['revision']
    model.close()


def test_episode_plan_rejects_reordering_even_with_matching_unit_contracts(tmp_path):
    from PhysicalRSI_core.infra.storage import digest
    from PhysicalRSI.Embodied_Harness.skills.episode_plan import freeze_episode_plan, SCHEMA
    steps = [{'operation': name, 'revision': 'v1'} for name in ['clear', 'initialize']]
    memory = {'schema': 'physicalrsi.skill-program-memory/v1',
              'programs': {'reset': {'steps': steps}},
              'bindings': {'reset': [{'operation': name, 'input': 'unit', 'output': 'unit'}
                                    for name in ['clear', 'initialize']]}}
    plan = dict(schema=SCHEMA, episode='test', rationale='test',
                programs={digest(memory): {'reset': list(reversed(steps))}})
    with pytest.raises(ValueError, match='exact registered'):
        freeze_episode_plan(tmp_path / 'plan.json', plan, {digest(memory): memory})
    assert not (tmp_path / 'plan.json').exists()


def test_checkpoint_skill_is_ready_before_first_observation(tmp_path, monkeypatch):
    model, decisions, implementations = configured(tmp_path, monkeypatch)
    assert len(implementations) == 1
    assert not decisions
    assert not implementations[0].observed
    model.close()
    assert implementations[0].closed
