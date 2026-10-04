"""Memory → remote model/environment skill → trajectory and versioned evidence."""

import sys
import uuid
from pathlib import Path

from PhysicalRSI_demos.runtime_service import ENVIRONMENT, MODEL
from PhysicalRSI.Embodied_Harness.memory.store import MemoryStore
from PhysicalRSI.Embodied_Harness.skills.composition import sequence
from PhysicalRSI.Embodied_Harness.skills.remote import policy_skill
from PhysicalRSI_core.contracts import Context, Contract
from PhysicalRSI_core.infra.clients import EnvironmentClient, ModelClient
from PhysicalRSI_core.infra.execution import Execution
from PhysicalRSI_core.infra.resources import ResourcePool
from PhysicalRSI_core.infra.services import Services, ServiceSpec
from PhysicalRSI_core.infra.storage import file_digest
from PhysicalRSI_core.infra.trajectory import EpisodeWriter, validate_episode

SPEC = {
    "arrays": {
        "state": {"dtype": "float32", "shape": (1,)},
        "actions": {"dtype": "float32", "shape": (1,)},
    },
    "success_mask": lambda data: data["terminated"],
}


def run(root):
    root = Path(root).resolve()
    pool = ResourcePool({"cpu": 2})
    specs = [
        ServiceSpec(
            kind,
            meta,
            command=(
                sys.executable,
                "-m",
                "PhysicalRSI_demos.runtime_service",
                kind,
                "--journal",
                str(root / kind / uuid.uuid4().hex),
            ),
            resources={"cpu": 1},
            sessions=kind == "model",
        )
        for kind, meta in [("environment", ENVIRONMENT), ("model", MODEL)]
    ]
    with Services(root / "services", pool=pool) as services:
        clients = services.start(specs)
        environment = EnvironmentClient(clients["environment"], expected=ENVIRONMENT)
        model = ModelClient(clients["model"], expected=MODEL)
        initial = environment.reset(uuid.uuid4().hex)
        snapshot = MemoryStore(root / "memory").snapshot({"goal": {"target": 2}})
        recorder = EpisodeWriter(
            root / "episodes",
            metadata={
                "memory_revision": snapshot.revision,
                "implementation_revision": file_digest(Path(__file__)),
                "model": MODEL,
                "environment": ENVIRONMENT,
            },
            spec=SPEC,
            initial_observation=initial,
        )
        skill = policy_skill(
            "skill.remote",
            file_digest(Path(__file__)),
            environment=environment,
            model=model,
            recorder=recorder,
        )
        composed = sequence(
            "remember_and_act", snapshot.reader("goal", Contract("goal")), skill
        )
        execution = Execution(
            root / "steps",
            observe=lambda op, ctx: {"state": environment.observe()["state"].tolist()},
        )
        result = composed(
            None,
            Context(
                uuid.uuid4().hex,
                execution=execution,
                harness_revision=composed.revision,
            ),
        )
        metadata = validate_episode(recorder.path, spec=SPEC)
    return dict(
        result=result,
        episode=str(recorder.path),
        metadata=metadata,
        resources=pool.status(),
        scope="local CPU service contract; no robot qualification",
    )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    print(json.dumps(run(parser.parse_args().workspace), indent=2))
