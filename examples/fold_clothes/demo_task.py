"""A deterministic fold-clothes software baseline.

The task mirrors the public phase structure of the physical task while using
an abstract garment state. It is useful for checking task loading, phase
ordering, memory, evidence, and baseline reporting without a simulator.
"""

import uuid
from pathlib import Path

from PhysicalRSI_core.infra.storage import atomic_json


PHASES = (
    "observe_garment",
    "grasp_left_edge",
    "fold_left_sleeve",
    "grasp_right_edge",
    "fold_right_sleeve",
    "fold_body",
    "place_folded_garment",
    "return_home",
)


class FoldClothesBaseline:
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
            "task": "fold_clothes",
            "scope": "software fold-clothes baseline",
            "qualification": False,
            "external_backend_required": False,
        }

    def _plan(self, request):
        requested = request.get("phases", list(PHASES))
        if tuple(requested) != PHASES:
            missing = [phase for phase in PHASES if phase not in requested]
            extra = [phase for phase in requested if phase not in PHASES]
            raise ValueError(
                "Fold plan must follow the declared phase order; "
                f"missing={missing}, extra={extra}"
            )
        return list(PHASES)

    def run(self, value=None):
        request = dict(value or {})
        garment = request.get("garment", self.configuration.get("garment", "shirt"))
        if not isinstance(garment, str) or not garment.strip():
            raise ValueError("garment must be a non-empty string")
        phases = self._plan(request)
        episode_id = uuid.uuid4().hex
        transitions = []
        state = {"garment": garment, "fold_count": 0, "home": False}
        for index, phase in enumerate(phases):
            if phase.startswith("fold_"):
                state["fold_count"] += 1
            if phase == "return_home":
                state["home"] = True
            transitions.append(
                {
                    "step": index,
                    "phase": phase,
                    "state": dict(state),
                    "source": "software_baseline",
                }
            )
        success = state["fold_count"] == 3 and state["home"]
        result = {
            "schema": "physicalrsi.fold-clothes-baseline/v1",
            "episode_id": episode_id,
            "task": "fold_clothes",
            "garment": garment,
            "phases": phases,
            "transitions": transitions,
            "success": success,
            "episode_score": 1.0 if success else 0.0,
            "natural_terminal": True,
            "memory": {
                "phase_count": len(phases),
                "fold_count": state["fold_count"],
                "return_home": state["home"],
            },
            "rsi_scheme": self.rsi_plan["scheme"],
            "harness": self.harness,
            "qualification": None,
            "scope": "software baseline; no physical qualification",
        }
        path = self.workspace / "baseline-runs" / (episode_id + ".json")
        atomic_json(path, result)
        return dict(result, evidence=str(path))

    def evolve(self):
        return {
            "state": "available",
            "task": "fold_clothes",
            "scope": "task baseline exposes a Self-Harness-compatible boundary",
            "qualification": None,
        }

    def status(self):
        return {
            "version": self.version,
            "task": "fold_clothes",
            "garment": self.configuration.get("garment", "shirt"),
            "phases": list(PHASES),
            "scope": "software baseline; no physical qualification",
        }


def create(*, configuration, workspace, source_directory, language_model, rsi_plan=None, harness=None):
    return FoldClothesBaseline(
        configuration,
        workspace,
        rsi_plan=rsi_plan,
        harness=harness,
    )


create.configuration_schema = {
    "type": "object",
    "properties": {
        "garment": {"type": "string", "minLength": 1},
        "variant": {"enum": ["standard", "random"]},
    },
    "additionalProperties": False,
}
