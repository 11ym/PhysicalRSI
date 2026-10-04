"""Deterministic task adapter for verifying the release CLI end to end.

This is a software baseline.  It exercises task loading, the selected RSI
plan, preflight and evidence persistence without pretending to qualify a robot
or simulator.
"""

import uuid
from pathlib import Path

from PhysicalRSI_core.infra.storage import atomic_json


class DemoBaseline:
    protocol_version = 1

    def __init__(self, configuration, workspace, *, rsi_plan=None, harness=None, **_):
        self.configuration = dict(configuration)
        self.workspace = Path(workspace).resolve()
        self.rsi_plan = rsi_plan or {"scheme": "hybrid"}
        self.harness = harness
        self.version = 1

    def check(self):
        return {
            "ready": True,
            "scope": "software baseline preflight",
            "qualification": False,
        }

    def run(self, value=None):
        request = dict(value or {})
        expected_goal = self.configuration.get("goal", "place_block")
        actual_goal = request.get("goal", expected_goal)
        success = actual_goal == expected_goal
        result = {
            "schema": "physicalrsi.demo-baseline/v1",
            "episode_id": uuid.uuid4().hex,
            "goal": actual_goal,
            "episode_score": 1.0 if success else 0.0,
            "success": success,
            "natural_terminal": True,
            "rsi_scheme": self.rsi_plan["scheme"],
            "harness": self.harness,
            "qualification": None,
            "scope": "software baseline; no physical qualification",
        }
        path = self.workspace / "baseline-runs" / (result["episode_id"] + ".json")
        atomic_json(path, result)
        return dict(result, evidence=str(path))

    def evolve(self):
        return {
            "state": "available",
            "scope": "demo adapter exposes baseline only; use a task adapter with Self-Harness for evolution",
            "qualification": None,
        }

    def status(self):
        return {
            "version": self.version,
            "goal": self.configuration.get("goal", "place_block"),
            "scope": "software baseline; no physical qualification",
        }


def create(*, configuration, workspace, source_directory, language_model, rsi_plan=None, harness=None):
    return DemoBaseline(
        configuration,
        workspace,
        rsi_plan=rsi_plan,
        harness=harness,
    )


create.configuration_schema = {
    "type": "object",
    "properties": {"goal": {"type": "string", "minLength": 1}},
    "additionalProperties": False,
}
