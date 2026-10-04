import json
from PhysicalRSI_demos.piano import PianoDemo, PIECES
from PhysicalRSI_core.infra.storage import digest


def test_prepared_episode_binds_immutable_memory_to_score_and_skill(tmp_path):
    backend = tmp_path / 'backend'
    for piece in PIECES.values():
        asset = backend / piece['midi']
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(b'MThd fixture')
    source = backend / 'robopianist/rsi/hand_controller.py'
    source.parent.mkdir(parents=True)
    source.write_text('# test controller revision\n')
    demo = PianoDemo({'robopianist_root': str(backend)}, tmp_path / 'workspace')
    first = demo.run({'piece': 'revolutionary'})
    second = demo.run({'piece': 'revolutionary'})
    assert first['state'] == 'prepared' and first['video'] is None
    assert first['memory']['revision'] == second['memory']['revision']
    from pathlib import Path
    memory = json.loads((Path(first['memory']['path']) / (first['memory']['revision'] + '.json')).read_text())
    assert digest(memory) == first['memory']['revision']
    assert memory['midi_sha256'] == first['midi_sha256']
    assert memory['contact_consistency_verified'] is False
    source.write_text('# changed controller\n')
    third = demo.run({'piece': 'revolutionary'})
    assert third['skill_revision'] != first['skill_revision']


def test_renderer_failure_is_persisted(tmp_path, monkeypatch):
    import subprocess
    import pytest
    from pathlib import Path
    backend = tmp_path / 'backend'
    midi = backend / PIECES['revolutionary']['midi']
    midi.parent.mkdir(parents=True)
    midi.write_bytes(b'MThd fixture')
    source = backend / 'robopianist/rsi/hand_controller.py'
    source.parent.mkdir(parents=True)
    source.write_text('# revision')
    workspace = tmp_path / 'workspace'
    def fail(command, **kwargs):
        assert command[0] == '/configured/python'
        assert kwargs['env']['MUJOCO_GL']
        raise subprocess.CalledProcessError(1, command, stderr='renderer failed')
    monkeypatch.setattr(subprocess, 'run', fail)
    demo = PianoDemo({'robopianist_root': str(backend), 'python': '/configured/python'}, workspace)
    with pytest.raises(subprocess.CalledProcessError):
        demo.run({'execute': True})
    episodes = list((workspace / 'episodes').glob('*.json'))
    assert len(episodes) == 1
    result = json.loads(episodes[0].read_text())
    assert result['state'] == 'failed'
    assert result['renderer_stderr'] == 'renderer failed'
    assert (Path(result['memory']['path']) / (result['memory']['revision'] + '.json')).exists()


def test_completed_run_records_result_memory_and_playback(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    from PhysicalRSI.Embodied_Harness.memory.store import Snapshot
    backend = tmp_path / 'backend'
    midi = backend / PIECES['october']['midi']
    midi.parent.mkdir(parents=True)
    midi.write_bytes(b'MThd fixture')
    source = backend / 'robopianist/rsi/hand_controller.py'
    source.parent.mkdir(parents=True)
    source.write_text('# controller')
    config = source.parent / 'configs/passive_ik.json'
    config.parent.mkdir()
    config.write_text('{"reach_z":0.01}')
    def render(command, **kwargs):
        assert command[command.index('--config') + 1] == str(config)
        output = Path(command[command.index('--output') + 1])
        output.write_bytes(b'video fixture')
        output.with_suffix('.json').write_text(json.dumps({
            'f1': .4, 'direct_key_actuation': False,
            'render': {'text_overlay': False}, 'full_song': False}))
        return subprocess.CompletedProcess(command, 0, stdout='rendered')
    monkeypatch.setattr(subprocess, 'run', render)
    result = PianoDemo({'robopianist_root': str(backend)}, tmp_path / 'workspace').run(
        {'piece': 'october', 'execute': True, 'seconds': 1})
    memory = Snapshot(Path(result['memory']['path']), result['memory']['revision']).read()
    parent = Snapshot(Path(result['memory']['path']), result['memory']['parent']).read()
    assert result['state'] == 'completed'
    assert memory['qualified'] is False and memory['metrics']['f1'] == .4
    assert 'metrics' not in parent
    assert memory['skill_revision'] == result['skill_revision']
    assert memory['controller_config'] == {'reach_z': .01}
    assert 'video controls' in Path(result['playback']).read_text()
    assert Path(result['video']).name in Path(result['playback']).read_text()


def test_missing_renderer_outputs_persist_failed_state(tmp_path, monkeypatch):
    import subprocess
    import pytest
    backend = tmp_path / 'backend'
    midi = backend / PIECES['revolutionary']['midi']
    midi.parent.mkdir(parents=True)
    midi.write_bytes(b'MThd fixture')
    source = backend / 'robopianist/rsi/hand_controller.py'
    source.parent.mkdir(parents=True)
    source.write_text('# controller')
    monkeypatch.setattr(subprocess, 'run', lambda cmd, **kw:
        subprocess.CompletedProcess(cmd, 0, stdout=''))
    workspace = tmp_path / 'workspace'
    with pytest.raises(RuntimeError, match='without video'):
        PianoDemo({'robopianist_root': str(backend)}, workspace).run({'execute': True})
    result = json.loads(next((workspace / 'episodes').glob('*.json')).read_text())
    assert result['state'] == 'failed'


def test_live_input_is_not_misreported_as_implemented(tmp_path):
    import pytest
    with pytest.raises(NotImplementedError, match="Live keyboard/MIDI"):
        PianoDemo({}, tmp_path).run({"interactive": True})


def test_user_recorded_midi_is_bound_to_the_selected_skill(tmp_path):
    backend = tmp_path / 'backend'
    user_midi = backend / 'recordings/revolutionary-take.mid'
    user_midi.parent.mkdir(parents=True)
    user_midi.write_bytes(b'MThd user take')
    source = backend / 'robopianist/rsi/hand_controller.py'
    source.parent.mkdir(parents=True)
    source.write_text('# controller')
    result = PianoDemo({'robopianist_root': str(backend)}, tmp_path / 'workspace').run(
        {'piece': 'revolutionary', 'input_midi': 'recordings/revolutionary-take.mid'})
    assert result['midi_role'] == 'user_performance'
    assert result['midi'].endswith('recordings/revolutionary-take.mid')
    assert result['memory']['skill_choice'] == 'piano.revolutionary.long_horizon'
    from pathlib import Path
    memory = json.loads((Path(result['memory']['path']) /
                         (result['memory']['revision'] + '.json')).read_text())
    assert memory['midi_role'] == 'user_performance'


def test_replay_binds_original_controller_and_rejects_wrong_score(tmp_path):
    import pytest
    from PhysicalRSI_core.infra.storage import file_digest
    backend = tmp_path / 'backend'
    midi = backend / PIECES['revolutionary']['midi']
    midi.parent.mkdir(parents=True)
    midi.write_bytes(b'MThd fixture')
    source = backend / 'robopianist/rsi/hand_controller.py'
    source.parent.mkdir(parents=True)
    source.write_text('# newer controller')
    audit = backend / 'audit.json'
    audit.with_suffix('.npz').write_bytes(b'trajectory fixture')
    audit.with_suffix('.midi-events.json').write_text('[]')
    recorded = {'midi_sha256': 'wrong', 'controller_sha256': 'original-controller',
        'controller_config': {'reach_z': .015}, 'direct_key_actuation': False,
        'full_song': False}
    audit.write_text(json.dumps(recorded))
    demo = PianoDemo({'robopianist_root': str(backend)}, tmp_path / 'workspace')
    with pytest.raises(ValueError, match='different score'):
        demo.run({'audit': 'audit.json', 'seconds': 1})
    recorded['midi_sha256'] = file_digest(midi)
    audit.write_text(json.dumps(recorded))
    result = demo.run({'audit': 'audit.json', 'seconds': 1})
    assert result['skill_revision'] == 'original-controller'
    assert result['controller_config'] == {'reach_z': .015}
    assert result['command'][1].endswith('render_piano_contact_replay.py')
    assert '--config' not in result['command']
    assert result['replay_source'][str(audit)] == file_digest(audit)
    with pytest.raises(ValueError, match='Full-song replay'):
        demo.run({'audit': 'audit.json'})
    recorded['control_timestep'] = .05
    audit.write_text(json.dumps(recorded))
    with pytest.raises(ValueError, match='recorded physics substeps'):
        demo.run({'audit': 'audit.json', 'seconds': 1, 'fps': 100})
    recorded.update(recorded_physics_substeps=True, physics_timestep=.005)
    audit.write_text(json.dumps(recorded))
    result = demo.run({'audit': 'audit.json', 'seconds': 1, 'fps': 100})
    assert result['command'][-2:] == ['--fps', '100']
    with pytest.raises(ValueError, match='preserve control boundaries'):
        demo.run({'audit': 'audit.json', 'seconds': 1, 'fps': 60})
