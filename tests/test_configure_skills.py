import hashlib
import json
import sys
from pathlib import Path
import pytest
from PhysicalRSI_baselines.robodojo.configure_skills import configure


def test_installed_assets_produce_complete_executable_catalog(tmp_path):
    assets = tmp_path / 'assets'
    for name in ('pi05', 'pi05-sparse-memory'):
        (assets / 'checkpoints' / name / 'params').mkdir(parents=True)
    programs = assets / 'implementations/code-skills'
    programs.mkdir(parents=True)
    source = programs / 'program.py'
    source.write_text('print("program")\n')
    (programs / 'programs.json').write_text(json.dumps({
        'schema': 'physicalrsi.frozen-program-library/v1',
        'programs': [{'name': 'alignment', 'description': 'Align observed geometry.',
                      'action_type': 'ee', 'configuration': {
            'runtime': 'frozen-program-service', 'cwd': '{root}',
            'command': ['{python}', '{root}/program.py'], 'output': '{evidence}',
            'files': {'{root}/program.py': hashlib.sha256(source.read_bytes()).hexdigest()}}}]}))
    interpreter = tmp_path / 'venv-python'
    interpreter.symlink_to(sys.executable)
    args = dict(assets=assets, python=interpreter, output=tmp_path/'skills.json',
                evidence=tmp_path/'evidence', endpoint='https://example.invalid/v1/chat/completions', model='test')
    config = configure(**args)
    assert set(config['skills']) == {'pi05', 'pi05-sparse-memory', 'alignment'}
    assert config['compositions'][-1] == 'memory-guided-program-library'
    library = config['composition_definitions']['memory-guided-program-library']
    assert library['action_types'] == ['ee']
    assert 'action_type' not in library
    assert list(config['program_registry']) == ['alignment']
    assert config['program_registry']['alignment']['source_name'] == 'alignment'
    assert config['skills']['alignment']['configuration']['command'] == [str(interpreter),str(source)]
    with pytest.raises(FileExistsError):
        configure(**args)
    source.write_text('changed')
    with pytest.raises(ValueError, match='dependency changed'):
        configure(**dict(args, output=tmp_path/'another.json'))


def test_program_relocation_preserves_shared_template(tmp_path):
    import os
    from PhysicalRSI_baselines.robodojo.prepare_code_programs import prepare
    root = tmp_path / 'installed'
    root.mkdir()
    original = tmp_path / 'shared-template.py'
    original.write_bytes(b'ROOT = "/PHYSICALRSI_PROGRAM_ASSETS"\r\n')
    path = root / 'program.py'
    os.link(original, path)
    os.link(original, root / 'duplicate.py')
    template = {'schema': 'physicalrsi.program-templates/v1',
                'text_files': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for name in ('program.py', 'duplicate.py')},
                'programs': [{'name': 'alignment', 'description': 'Align.', 'action_type': 'joint',
                              'configuration': {'command': ['{python}', '{root}/program.py']},
                              'dependency_files': ['program.py']}]}
    (root/'program-templates.json').write_text(json.dumps(template))
    assert prepare(root) == 1
    assert str(root) in path.read_text()
    assert '/PHYSICALRSI_PROGRAM_ASSETS' in original.read_text()
    assert path.read_bytes().endswith(b'\r\n')
    assert (root / 'duplicate.py').read_bytes() == path.read_bytes()
    result = json.loads((root/'programs.json').read_text())
    assert result['programs'][0]['configuration']['files']['{root}/program.py'] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        prepare(root)


@pytest.mark.parametrize('explicit_framework', [True, False])
def test_unified_worker_uses_shared_framework_root(tmp_path, monkeypatch, explicit_framework):
    assets = tmp_path / 'assets'
    for name in ('pi05', 'pi05-sparse-memory'):
        (assets / 'checkpoints' / name / 'params').mkdir(parents=True)
    programs = assets / 'implementations/code-skills'
    programs.mkdir(parents=True)
    source = programs / 'launch.py'
    source.write_text('# launcher fixture\n')
    (programs / 'programs.json').write_text(json.dumps({
        'schema': 'physicalrsi.frozen-program-library/v1', 'programs': [{
            'name': 'test-skill', 'description': 'Test composition', 'action_type': 'joint',
            'configuration': {'command': ['{python}', '{root}/launch.py', '--framework-root', '{framework}'],
                              'files': {'{root}/launch.py': hashlib.sha256(source.read_bytes()).hexdigest()}}}]}))
    framework = tmp_path / 'XPolicyLab'
    framework.mkdir()
    (framework / 'model_template.py').write_text('class ModelTemplate: pass\n')
    (framework / 'client_server').mkdir()
    monkeypatch.chdir(framework)
    kwargs = {'framework': framework} if explicit_framework else {}
    result = configure(assets, python=sys.executable, output=tmp_path / 'configured.json',
                       evidence=tmp_path / 'evidence', endpoint='https://example.invalid', model='test', **kwargs)
    command = result['skills']['test-skill']['configuration']['command']
    assert command[-2:] == ['--framework-root', str(framework)]
    assert '--xpolicylab' not in command
