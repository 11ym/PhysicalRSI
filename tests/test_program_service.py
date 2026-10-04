import pytest
from PhysicalRSI_baselines.robodojo.skills import program_service


def test_materialized_transport_gets_an_isolated_policy_namespace(tmp_path):
    source = tmp_path / 'source'
    (source / 'policy' / 'rdj_rgb').mkdir(parents=True)
    (source / 'policy' / 'rdj_rgb' / 'model.py').write_text('original')
    (source / '.rdj-source-revision').write_text('frozen')
    (source / 'setup_policy_server.py').write_text('server')
    output = tmp_path / 'episode'
    output.mkdir()
    command = ['python', 'manager.py', '--xpolicylab', str(source)]
    actual = program_service.isolate_transport(command, output)
    assert command[-1] == str(source)
    assert actual[-1] == str(output / 'XPolicyLab')
    assert not (output / 'XPolicyLab/policy/rdj_rgb').exists()
    assert (source / 'policy/rdj_rgb/model.py').read_text() == 'original'
    assert (output / 'XPolicyLab/.rdj-source-revision').read_text() == 'frozen'


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
