"""Versioned reset → execute → verify experiments, independent of any policy.

Adapters own physical resets and outcome measurement. Deadlines are cooperative:
an adapter must honor Context.check() and bound its own device/network calls.
The journal prevents silently repeating an interrupted physical experiment.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic
from typing import Protocol

from .contracts import Context
from .infra.storage import atomic_json, digest, file_digest, identifier, locked, read_json
from .self_harness import ReconciliationRequired


@dataclass(frozen=True)
class Budget:
    max_steps: int = 100
    seconds: float = 60

    def __post_init__(self):
        import math
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if isinstance(self.seconds, bool) or not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("seconds must be finite and positive")


class Environment(Protocol):
    def identity(self) -> dict: ...
    def reset(self, case: dict, context: Context) -> dict:
        """Return {ready: bool, observation: JSON value}; verify reset readiness."""
        ...
    def step(self, action, context: Context) -> dict:
        """Return {observation: JSON value, terminated: bool}."""
        ...


class Policy(Protocol):
    def identity(self) -> dict: ...
    def act(self, observation, context: Context): ...


class Verifier(Protocol):
    def identity(self) -> dict: ...
    def verify(self, case: dict, trace: list[dict]) -> dict:
        """Return outcome (success/failure/uncertain), reason and measurements."""
        ...


class ExperimentRuntime:
    """One receipt binds the case, budget, adapters, raw trajectory and verdict.

    A matching completed run is read without executing again. Interrupted runs
    require reconciliation; use a new ID only after checking the physical state.
    Identity/evidence integrity is verified here; verifier accuracy is a separate
    empirical property of each task adapter.
    """
    def __init__(self, root):
        self.root = Path(root).resolve()

    def read(self, run_id):
        folder = self.root / identifier(run_id)
        receipt = read_json(folder / "receipt.json")
        if receipt["state"] != "completed":
            raise ReconciliationRequired(f"Inspect interrupted experiment: {folder}")
        for name, expected in receipt["evidence"].items():
            path = (folder / name).resolve()
            if not path.is_relative_to(folder) or file_digest(path) != expected:
                raise ValueError("Experiment evidence changed: " + name)
        if digest(read_json(folder / "experiment.json")) != receipt["experiment_sha256"]:
            raise ValueError("Experiment specification changed")
        verdict = read_json(folder / "verdict.json")
        if receipt["verdict"] != verdict or receipt["outcome"] != verdict["outcome"]:
            raise ValueError("Receipt disagrees with the recorded verdict")
        return receipt

    def run(self, run_id, *, task, case, scope, environment, policy, verifier, budget=None):
        budget = budget or Budget()
        folder = self.root / identifier(run_id)
        if not scope or not task:
            raise ValueError("Declare the task and evidence scope")
        identities = lambda: dict(environment=environment.identity(), policy=policy.identity(),
                                  verifier=verifier.identity(), runtime=file_digest(Path(__file__)))
        specification = dict(schema="physicalrsi.experiment/v1", task=task, case=deepcopy(case),
                             scope=scope, budget=asdict(budget), adapters=deepcopy(identities()))
        frozen = digest(specification)
        with locked(folder / ".lock"):
            if (folder / "receipt.json").exists():
                if read_json(folder / "receipt.json")["experiment_sha256"] != frozen:
                    raise ValueError("Run ID belongs to a different experiment")
                return self.read(run_id)
            atomic_json(folder / "experiment.json", specification)
            receipt = dict(schema="physicalrsi.experiment-receipt/v1", id=run_id,
                           state="started", experiment_sha256=frozen, scope=scope,
                           qualification=None)
            atomic_json(folder / "receipt.json", receipt)
            start = monotonic()
            context = Context(run_id, deadline=start + budget.seconds)
            trace = []
            try:
                context.check()
                initial = environment.reset(deepcopy(case), context)
                atomic_json(folder / "reset.json", initial)
                context.check()
                if type(initial.get("ready")) is not bool:
                    raise ValueError("Reset must declare readiness")
                if not initial["ready"]:
                    verdict = dict(outcome="invalid", reason="reset_not_ready", measurements={})
                else:
                    trace.append(dict(observation=initial["observation"]))
                    for step in range(budget.max_steps):
                        context.check()
                        action = policy.act(deepcopy(trace[-1]["observation"]), context)
                        context.check()
                        # Persist the intended effect before entering the environment.
                        atomic_json(folder / "pending-action.json", dict(step=step, action=action))
                        result = environment.step(action, context)
                        context.check()
                        if type(result.get("terminated")) is not bool:
                            raise ValueError("Environment must declare termination")
                        row = dict(step=step, action=action, **result)
                        trace.append(row)
                        atomic_json(folder / "trajectory.json", trace)
                        if result["terminated"]:
                            break
                    verdict = verifier.verify(deepcopy(case), deepcopy(trace))
                    context.check()
                    if verdict.get("outcome") not in {"success", "failure", "uncertain"}:
                        raise ValueError("Verifier must return success, failure or uncertain")
                    if not verdict.get("reason") or not isinstance(verdict.get("measurements"), dict):
                        raise ValueError("Verifier must supply a reason and measured evidence")
                if identities() != specification["adapters"]:
                    raise ValueError("An adapter changed during the experiment")
                atomic_json(folder / "trajectory.json", trace)
                atomic_json(folder / "verdict.json", verdict)
                receipt.update(state="completed", outcome=verdict["outcome"], verdict=verdict,
                               steps=max(0, len(trace)-1), elapsed_seconds=monotonic()-start,
                               evidence={name: file_digest(folder/name) for name in
                                         ("experiment.json", "reset.json", "trajectory.json", "verdict.json")})
                atomic_json(folder / "receipt.json", receipt)
                return self.read(run_id)
            except BaseException as error:
                receipt.update(state="needs_reconciliation", error=type(error).__name__ + ": " + str(error))
                atomic_json(folder / "receipt.json", receipt)
                raise
