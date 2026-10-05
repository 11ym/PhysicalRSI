import pytest
from PhysicalRSI_baselines.robodojo.skills import program_service


def test_changed_program_is_rejected_before_start(tmp_path, monkeypatch):
    source = tmp_path / 'program.py'
    source.write_text('changed')
    def forbidden(*args, **kwargs):
        pytest.fail('Must not spawn unverified code')
    monkeypatch.setattr(program_service.subprocess, 'Popen', forbidden)
    with pytest.raises(ValueError, match='dependency changed'):
        program_service.Model({'command': ['python', str(source)], 'files': {str(source): '0'*64}})


def test_action_aliases_preserve_coordinates_and_reject_conflicts():
    model = object.__new__(program_service.Model)
    pose = [1, 2, 3, 1, 0, 0, 0]
    model._call = lambda *args: [[{'left_ee_pose': pose, 'left_gripper': [0]}]]
    actions = model.get_action_batch([0])
    assert actions == [[{'left_ee_pose': pose, 'left_ee_joint_state': [0]}]]
    assert actions[0][0]['left_ee_pose'] is pose
    model._call = lambda *args: [[{'left_gripper': [0], 'left_ee_joint_state': [1]}]]
    with pytest.raises(ValueError, match='Conflicting'):
        model.get_action_batch([0])


def test_fixed_base_program_omits_only_mobile_metadata_without_mutating_input():
    model = object.__new__(program_service.Model)
    model._call = lambda method, value: value
    joints = [0.0] * 6
    observation = {'state': {'mobile': {'base_pose': [0, 0, 0]},
                             'left_arm_joint_state': joints, 'unexpected': 1},
                   'vision': {'camera': {'color': object()}}}
    prepared = model.update_obs_batch([observation])[0]
    assert 'mobile' in observation['state']
    assert 'mobile' not in prepared['state']
    assert prepared['state']['left_arm_joint_state'] is joints
    assert prepared['vision'] is observation['vision']
    assert prepared['state']['unexpected'] == 1


def test_private_worker_roundtrip_without_server_or_framework_copy(tmp_path):
    import hashlib
    import json
    import sys
    import numpy as np
    runtime = tmp_path / 'runtime'
    (runtime / 'transport').mkdir(parents=True)
    (runtime / 'transport/__init__.py').write_text('')
    (runtime / 'transport/binding.py').write_text(
        'class Echo:\n'
        '    def reset(self): self.rows = []\n'
        '    def update_obs_batch(self, rows): self.rows = rows\n'
        '    def get_action_batch(self, indices):\n'
        '        return [[{"left_arm_joint_state": self.rows[0]["state"]["joints"]}] for i in indices]\n'
        'def load_bound_model(descriptor): return Echo()\n')
    launcher = runtime / 'launch.py'
    launcher.write_text(
        'import json\n'
        'def expand(value, variables): return value\n'
        'def validate(descriptor): pass\n'
        'def reserve_service_endpoints(descriptor): return []\n'
        'def environment(descriptor, inherited): return inherited\n'
        'def write_json(path, value): path.write_text(json.dumps(value))\n')
    descriptor = runtime / 'descriptor.json'
    descriptor.write_text(json.dumps({'pythonpath': [], 'services': [], 'transport': {}}))
    config = dict(cwd=str(tmp_path), output=str(tmp_path / 'output'),
                  command=[sys.executable, str(launcher), '--descriptor', str(descriptor),
                           '--assets', str(tmp_path), '--output', '{output}',
                           '--framework-root', str(tmp_path / 'framework'),
                           '--python', sys.executable, '--port', '{port}'],
                  files={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (launcher, descriptor)}, startup_timeout_s=10)
    model = program_service.Model(config)
    process = model.process
    try:
        joints = np.arange(6, dtype=np.float64)
        model.update_obs_batch([{'env_idx': 4, 'state': {'joints': joints}}])
        np.testing.assert_array_equal(model.get_action_batch([4])[0][0]['left_arm_joint_state'], joints)
        assert not (model.output / 'XPolicyLab').exists()
        assert not hasattr(model, 'client')
        with pytest.raises(RuntimeError, match='Code skill failed'):
            model._call('unknown')
        with pytest.raises(RuntimeError, match='not running'):
            model.reset()
    finally:
        model.close()
    assert process.poll() is not None
    assert model.pipe.closed
