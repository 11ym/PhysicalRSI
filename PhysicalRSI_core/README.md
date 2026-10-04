# PhysicalRSI-core

Core makes a proposed improvement inspectable: the experiment records its conditions, executes a frozen candidate, asks an independent verifier for an outcome, and preserves the evidence used to select a revision.

## Experiment runtime

`experiments.py` provides `ExperimentRuntime`, `Budget`, and the `Environment`, `Policy`, and `Verifier` ports. An environment verifies reset readiness and reports observations/termination; a policy returns actions; a verifier compares the observations with the declared task requirement.

```python
from PhysicalRSI_demos.experiment import run
receipt = run('/tmp/physicalrsi-experiments', 'trial-1')
print(receipt['outcome'], receipt['evidence'])
```

An experiment directory contains `experiment.json`, `reset.json`, `trajectory.json`, `verdict.json` and `receipt.json`. Adapter identities, budget, case and evidence hashes are bound to the receipt. `success`, `failure`, `uncertain` and `invalid` remain separate. A reset that is not ready cannot become a failed policy trial. An external effect interrupted before completion requires reconciliation; rerunning the same ID never silently repeats it.

Deadlines are cooperative. Adapters must bound device calls and honor `Context.check()`. The runtime verifies local evidence and identity consistency; it cannot guarantee sensor accuracy, verifier accuracy, physical reproducibility or process isolation. Physical adapters must coordinate ownership of their device resources.

## Self-Harness

The original preview release's `self_harness/` is retained. Its injected proposal, evaluation and selection ports perform:

1. Development feedback, including failures.
2. Candidate materialization with parent and dependency identities.
3. Admission and a frozen comparison pool/profile.
4. Fresh validation cases and paired parent/candidate evaluation.
5. Selection with regression limits and evidence checks.
6. Compare-and-swap inheritance through `lineage.HarnessState`.

`/evolve` runs the CPU service example end to end. Completed stages resume from receipts. Unknown external effects require reconciliation. The current implementation fixes the foundation component during an iteration; it does not automatically improve foundation-model weights.

The experiment runtime is an adapter building block. Task-specific Self-Harness evaluators decide how to turn measured trials into comparison results. `self_harness/selection.py` checks paired cases, evaluator identity, native outcomes, evidence digests and per-task regression limits. It does not infer physical qualification from passing software checks.

## Other modules

- `contracts.py`: typed, versioned operations and cooperative execution context.
- `infra/`: atomic storage, events, resource pools, process/service lifecycle, RPC and trajectory evidence.
- `lineage/`: current revision, historical evidence and explicit rollback records.

Core imports no CLI, baseline or concrete task implementation. Application adapters provide those dependencies. See [architecture](../docs/architecture.md) for System 1 and System 2.
