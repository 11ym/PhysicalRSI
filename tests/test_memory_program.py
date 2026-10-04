import pytest

from PhysicalRSI.Embodied_Harness.memory.store import MemoryStore
from PhysicalRSI.Embodied_Harness.skills.memory_program import compile_program
from PhysicalRSI_core.contracts import Context, Contract, Operation


def test_memory_order_drives_actual_operation_calls(tmp_path):
    store = MemoryStore(tmp_path)
    calls = []
    def op(name, call):
        def execute(value, context):
            calls.append(name)
            return call(value)
        return Operation(name, 'v1', Contract('number'), Contract('number'), execute)
    registry = {'add': op('add', lambda x: x + 2),
                'multiply': op('multiply', lambda x: x * 3)}
    def program(names):
        snapshot = store.snapshot({'schema': 'physicalrsi.skill-program-memory/v1',
            'programs': {'calculate': {'steps': [
                {'operation': name, 'revision': 'v1'} for name in names]}}})
        return compile_program(snapshot, 'calculate', registry)
    first = program(['add', 'multiply'])
    second = program(['multiply', 'add'])
    assert first(1, Context('first')) == 9
    assert second(1, Context('second')) == 5
    assert calls == ['add', 'multiply', 'multiply', 'add']
    assert first.revision != second.revision


def test_invalid_binding_and_tampered_memory_fail_before_execution(tmp_path):
    store = MemoryStore(tmp_path)
    calls = []
    operation = Operation('run', 'v1', Contract('in'), Contract('out'),
                          lambda value, context: calls.append(value))
    snapshot = store.snapshot({'schema': 'physicalrsi.skill-program-memory/v1',
        'programs': {'program': {'steps': [{'operation': 'run', 'revision': 'v1'}]}}})
    with pytest.raises(ValueError, match='identity'):
        compile_program(snapshot, 'program', {'run': Operation(
            'run', 'v2', operation.input, operation.output, operation.call)})
    program = compile_program(snapshot, 'program', {'run': operation})
    (snapshot.root / (snapshot.revision + '.json')).write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        program('input', Context('test'))
    assert calls == []


def test_incompatible_contracts_fail_before_execution(tmp_path):
    first = Operation('first', 'v1', Contract('a'), Contract('b'), lambda x, c: x)
    second = Operation('second', 'v1', Contract('c'), Contract('d'), lambda x, c: x)
    snapshot = MemoryStore(tmp_path).snapshot({
        'schema': 'physicalrsi.skill-program-memory/v1',
        'programs': {'program': {'steps': [
            {'operation': 'first', 'revision': 'v1'},
            {'operation': 'second', 'revision': 'v1'}]}}})
    with pytest.raises(ValueError, match='Incompatible'):
        compile_program(snapshot, 'program', {'first': first, 'second': second})
