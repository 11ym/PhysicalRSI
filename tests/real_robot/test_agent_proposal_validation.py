"""Code-only validation feedback, without any robot or private judge data."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PhysicalRSI_demos.real_robot.models import VisualCodeModel


def response(source):
    return json.dumps(dict(source=source, memory={}, rationale="repair")), [], []


class Model:
    config = SimpleNamespace(protocol="responses", public=lambda: {"model": "offline-test"})

    def __init__(self, replies):
        self.replies, self.histories = iter(replies), []

    def complete(self, history, tools):
        self.histories.append(history)
        assert tools == []
        return next(self.replies)


HISTORY = [dict(role="system", content="Generate policy"),
           dict(role="user", content=json.dumps(dict(development_feedback=[dict(split="development")])))]
BAD = "def policy(robot, memory):\n    robot.yield_control(type(robot).__name__)\n"
GOOD = "def policy(robot, memory):\n    robot.yield_control('Need current geometry')\n"


def test_invalid_source_is_returned_to_model_with_exact_validation_reason(tmp_path):
    model = Model([response(BAD), response(GOOD)])
    result = VisualCodeModel(model, output=tmp_path).complete(HISTORY, [])
    assert json.loads(result[0])["source"] == GOOD
    assert len(model.histories) == 2
    repair = json.loads(model.histories[1][-1]["content"])
    assert repair["validation_error"] == "Code-policy source may not access dunder attributes"
    assert repair["rejected_source"] == BAD
    assert repair["executed"] is False
    assert len(list(tmp_path.glob("source-validation-*.json"))) == 2
    assert json.loads((tmp_path / "source-validation-01.json").read_text())["valid"] is False
    assert json.loads((tmp_path / "source-validation-02.json").read_text())["valid"] is True


def test_repair_budget_never_admits_repeated_invalid_code(tmp_path):
    model = Model([response(BAD)] * 3)
    with pytest.raises(ValueError, match="dunder"):
        VisualCodeModel(model, output=tmp_path).complete(HISTORY, [])
    assert len(model.histories) == 3
    assert len(list(tmp_path.glob("source-validation-*.json"))) == 3


def test_valid_source_uses_one_model_request(tmp_path):
    model = Model([response(GOOD)])
    assert VisualCodeModel(model, output=tmp_path).complete(HISTORY, [])[0] == response(GOOD)[0]
    assert len(model.histories) == 1


def test_malformed_response_can_be_repaired_without_execution(tmp_path):
    model = Model([("not JSON", [], []), response(GOOD)])
    result = VisualCodeModel(model, output=tmp_path).complete(HISTORY, [])
    assert json.loads(result[0])["source"] == GOOD
    assert len(model.histories) == 2


def test_existing_parent_does_not_remove_reusable_skill_source(tmp_path):
    from PhysicalRSI_demos.real_robot.contracts import Task
    from PhysicalRSI_demos.real_robot.library import PolicyLibrary, dependency_files
    task = Task("pour-water-bottle-into-cup", "Pour into cup", "Visible water")
    source = "def side_enclose_grasp(robot, item):\n    robot.relate(item)\n"
    library = PolicyLibrary(tmp_path / "library", foundation={"model": "offline-test"},
        reusable_skills=[dict(name="side_enclose_grasp", entry="side_enclose_grasp(robot, item)",
                             source=source, qualification="unvalidated-development-candidate")])
    parent = library.freeze(task=task, source=GOOD, memory={})
    model = Model([response(GOOD + "\n")])
    library.propose(task=task, parent=parent,
                    feedback=[dict(task=task.id, split="development", observations=[], execution=[])],
                    model=model, output=tmp_path / "proposal")
    request = json.loads(model.histories[0][1]["content"][0]["text"])
    skill = request["reusable_skills"][0]
    assert skill["entry"] == "side_enclose_grasp(robot, item)"
    assert "def side_enclose_grasp(robot, item):" in skill["source"]
    assert "robot.relate" in skill["source"]
    assert skill["qualification"] == "unvalidated-development-candidate"
    frozen = json.loads((Path(parent["root"]) / "prompts.json").read_text())
    assert frozen["reusable_skills"][0]["source"] == source
    use = json.loads((tmp_path / "proposal/reusable-skill-use.json").read_text())
    assert use["side_enclose_grasp"]["source_supplied"] is True
    # Providing code alone is never reported as actual adoption/execution.
    assert use["side_enclose_grasp"]["defined"] is False
    assert use["side_enclose_grasp"]["referenced_by_call"] is False
