from PhysicalRSI_baselines.robodojo.clean_policy_assets import extract


def test_pure_helpers_survive_but_gt_readers_and_dependents_are_removed():
    source = '''CLOSED = 0

def rotation(x):
    return x * 2

def reader(env):
    return env.scene_manager.layout_manager.get_instance_pose('object')

def privileged_controller(env):
    return rotation(reader(env))

class MixedReader:
    def read(self, env):
        return getattr(env, 'env_seeds')
'''
    clean, blocked, report = extract(source)
    scope = {}
    exec(clean, scope)
    assert scope['rotation'](3) == 6
    assert scope['CLOSED'] == 0
    assert {'reader', 'privileged_controller', 'MixedReader'} <= blocked
    assert all(name not in scope for name in blocked)
    assert len(report) == 3


def test_import_cleanup_keeps_only_pure_geometry_symbols():
    source = '''from bridge import rotation, reader as read_gt

def controller(env):
    return read_gt(env)
'''
    clean, blocked, _ = extract(source, forbidden_imports={'bridge': {'reader'}})
    assert clean.strip() == 'from bridge import rotation'
    assert blocked == {'read_gt', 'controller'}


def test_relative_import_dependency_is_removed():
    source = 'from .bridge import helper, reader\n\ndef bad(env): return reader(env)\n'
    clean, blocked, _ = extract(source, module_name='package.controller',
                               forbidden_imports={'package.bridge': {'reader'}})
    assert 'def bad' not in clean
    assert clean.strip() == 'from .bridge import helper'
    assert blocked == {'reader', 'bad'}


def test_owned_rgb_counter_survives_without_preserving_evaluator_readers():
    from PhysicalRSI_baselines.robodojo.clean_policy_assets import normalize_owned_runtime, extract
    source = '''from collections import defaultdict
from types import SimpleNamespace
class Model:
    def reset(self):
        self.runtime = SimpleNamespace(take_action_cnt=defaultdict(int))
    def advance(self, index):
        self.runtime.take_action_cnt[index] += 1
        return self.runtime.take_action_cnt[index]
def evaluator_reader(task_env):
    return task_env.scene_manager.layout_manager
'''
    normalized = normalize_owned_runtime(source, 'rdj_rgb_adapters.plug_in_charger')
    cleaned, blocked, _ = extract(normalized)
    scope = {}
    exec(cleaned, scope)
    model = scope['Model']()
    model.reset()
    assert model.advance(7) == 1
    assert model.advance(7) == 2
    assert 'evaluator_reader' in blocked
    assert 'Model' not in blocked
    ordinary, blocked, _ = extract(normalize_owned_runtime(source, 'another.module'))
    assert 'Model' in blocked


def test_export_cleans_bundled_evaluators_without_touching_provider_inference(tmp_path):
    from PhysicalRSI_baselines.robodojo.clean_policy_assets import export

    source = tmp_path / 'source'
    source.mkdir()
    (source / 'programs.json').write_text('{}')
    paths = [
        'arrange_rgb/revision/arrange_rgb/deploy.py',
        'pour_by_language/revision/pour_by_language/core/deploy.py',
        'pi05_sparse_mem/revision/pi05_sparse_mem/src/openpi/integrations/rmbench_sparse_memory.py',
    ]
    text = '''def observation_only(obs):
    return obs['state']

def terminal(task):
    return task.take_action_cnt

def old_evaluation(task):
    return terminal(task)
'''
    for relative in paths:
        path = source / 'implementations' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    inference = source / 'implementations/pi05_sparse_mem/revision/inference.py'
    inference.write_text("def action(obs): return obs['state']\n")
    destination = tmp_path / 'clean'
    report = export(source, destination)
    assert len(report['changes']) == 3
    for relative in paths:
        clean = (destination / 'implementations' / relative).read_text()
        scope = {}
        exec(clean, scope)
        assert scope['observation_only']({'state': 3}) == 3
        assert 'terminal' not in scope and 'old_evaluation' not in scope
        assert (source / 'implementations' / relative).read_text() == text
    assert (destination / inference.relative_to(source)).read_bytes() == inference.read_bytes()
