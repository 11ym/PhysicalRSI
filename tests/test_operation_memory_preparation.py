import hashlib
import json
from pathlib import Path
import subprocess
import sys

from PhysicalRSI_core.infra.storage import digest


def test_preparation_collects_verified_memories_and_deduplicates(tmp_path):
    memory = {'schema': 'physicalrsi.skill-program-memory/v1', 'programs': {
        'advance': {'steps': [{'operation': 'motion.advance', 'revision': 'v1'}]}}}
    text = json.dumps(memory)
    files = ['first.memory.json', 'nested.memory.json']
    for name in files:
        (tmp_path / name).write_text(text)
    template = {'schema': 'physicalrsi.program-templates/v1',
                'text_files': {name: hashlib.sha256(text.encode()).hexdigest() for name in files},
                'programs': [{'name': 'test', 'description': 'Test fixture',
                              'action_type': 'joint', 'configuration': {},
                              'dependency_files': files, 'operation_memory_files': files}]}
    (tmp_path / 'program-templates.json').write_text(json.dumps(template))
    script = Path(__file__).resolve().parents[1] / 'PhysicalRSI_baselines/robodojo/prepare_code_programs.py'
    process = subprocess.run([sys.executable, '-I', '-S', str(script), str(tmp_path)],
                             cwd=tmp_path, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert 'Prepared executable programs: 1' in process.stdout
    result = json.loads((tmp_path / 'programs.json').read_text())
    assert result['programs'][0]['configuration']['operation_memories'] == {digest(memory): memory}


def test_preparation_preserves_every_nested_class_memory(tmp_path):
    from PhysicalRSI_baselines.robodojo.prepare_code_programs import prepare
    memories = {name: {'schema': 'physicalrsi.skill-program-memory/v1', 'programs': {
        'advance': {'steps': [{'operation': name + '.advance', 'revision': 'v1'}]}}}
        for name in ['BatchModel', 'EpisodeModel']}
    document = {'schema': 'physicalrsi.structured-module/v1', 'classes': memories}
    text = json.dumps(document)
    (tmp_path / 'model.memory.json').write_text(text)
    template = {'schema': 'physicalrsi.program-templates/v1',
                'text_files': {'model.memory.json': hashlib.sha256(text.encode()).hexdigest()},
                'programs': [{'name': 'test', 'description': 'Test fixture',
                              'action_type': 'joint', 'configuration': {},
                              'dependency_files': ['model.memory.json'],
                              'operation_memory_files': ['model.memory.json']}]}
    (tmp_path / 'program-templates.json').write_text(json.dumps(template))
    assert prepare(tmp_path) == 1
    result = json.loads((tmp_path / 'programs.json').read_text())
    assert result['programs'][0]['configuration']['operation_memories'] == {
        digest(memory): memory for memory in memories.values()}


def test_preparation_refreshes_verified_worker_identity_after_relocation(tmp_path):
    from PhysicalRSI_baselines.robodojo.prepare_code_programs import prepare
    source = tmp_path / 'model.py'
    source.write_text('ASSETS = "/PHYSICALRSI_PROGRAM_ASSETS"\n')
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    worker = {'schema': 'physicalrsi.skill-worker/v1', 'implementation': {
        'source': '{assets}/model.py', 'sha256': original_hash, 'module': 'model', 'class': 'Model'},
        'transport': {'initialization': {'controller_source': '{assets}/model.py', 'controller_sha256': original_hash}}}
    descriptor = tmp_path / 'worker.json'
    descriptor.write_text(json.dumps(worker))
    template = {'schema': 'physicalrsi.program-templates/v1',
                'text_files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, descriptor]},
                'programs': [{'name': 'test', 'description': 'Test fixture', 'action_type': 'joint',
                              'configuration': {}, 'dependency_files': ['model.py', 'worker.json'],
                              'worker_descriptor': 'worker.json'}]}
    (tmp_path / 'program-templates.json').write_text(json.dumps(template))
    assert prepare(tmp_path) == 1
    after = hashlib.sha256(source.read_bytes()).hexdigest()
    assert after != original_hash
    assert json.loads(descriptor.read_text())['implementation']['sha256'] == after
    assert json.loads(descriptor.read_text())['transport']['initialization']['controller_sha256'] == after
    installed = json.loads((tmp_path / 'programs.json').read_text())['programs'][0]
    assert installed['configuration']['files']['{root}/worker.json'] == hashlib.sha256(descriptor.read_bytes()).hexdigest()
