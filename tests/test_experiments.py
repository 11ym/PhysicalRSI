import json

import pytest

from PhysicalRSI_core.experiments import Budget, ExperimentRuntime
from PhysicalRSI_core.self_harness import ReconciliationRequired
from PhysicalRSI_demos.experiment import CounterEnvironment, CounterPolicy, CounterVerifier


def run(runtime, **changes):
    options = dict(task='counter', case={'initial':0,'target':3}, scope='software',
                   environment=CounterEnvironment(), policy=CounterPolicy(),
                   verifier=CounterVerifier(), budget=Budget(max_steps=5,seconds=5))
    options.update(changes)
    return runtime.run('trial', **options)


def test_completed_receipt_is_read_without_reexecution(tmp_path):
    class Counting(CounterEnvironment):
        resets = 0
        def reset(self, case, context):
            self.resets += 1
            return super().reset(case, context)
    runtime=ExperimentRuntime(tmp_path)
    env=Counting()
    first=run(runtime,environment=env)
    assert first['outcome']=='success'
    assert run(runtime,environment=env)==first
    assert env.resets==1
    assert first['steps']==3
    with pytest.raises(ValueError,match='different experiment'):
        run(runtime,budget=Budget(max_steps=10,seconds=5))


def test_termination_does_not_override_independent_verification(tmp_path):
    result=run(ExperimentRuntime(tmp_path),policy=CounterPolicy(2))
    assert result['state']=='completed'
    assert result['outcome']=='failure'
    assert result['verdict']['measurements']['position']==4
    assert result['qualification'] is None


def test_invalid_reset_is_not_policy_failure(tmp_path):
    class Unready(CounterEnvironment):
        def reset(self,case,context):return {'ready':False}
        def step(self,*args):raise AssertionError('Must not execute')
    result=run(ExperimentRuntime(tmp_path),environment=Unready())
    assert result['outcome']=='invalid' and result['steps']==0


def test_unknown_verdict_is_preserved(tmp_path):
    class Unknown(CounterVerifier):
        def verify(self,case,trace):
            return {'outcome':'uncertain','reason':'Independent sensor unavailable','measurements':{}}
    assert run(ExperimentRuntime(tmp_path),verifier=Unknown())['outcome']=='uncertain'


def test_interrupted_effect_is_never_retried(tmp_path):
    class Interrupted(CounterEnvironment):
        calls=0
        def step(self,action,context):
            self.calls+=1
            raise RuntimeError('Connection lost after dispatch')
    env=Interrupted();runtime=ExperimentRuntime(tmp_path)
    with pytest.raises(RuntimeError,match='Connection lost'):
        run(runtime,environment=env)
    with pytest.raises(ReconciliationRequired):
        run(runtime,environment=env)
    assert env.calls==1
    receipt=json.loads((tmp_path/'trial/receipt.json').read_text())
    assert receipt['state']=='needs_reconciliation'
    assert (tmp_path/'trial/pending-action.json').is_file()


def test_modified_evidence_cannot_be_reused(tmp_path):
    runtime=ExperimentRuntime(tmp_path);run(runtime)
    (tmp_path/'trial/trajectory.json').write_text('[]')
    with pytest.raises(ValueError,match='evidence changed'):
        run(runtime)
