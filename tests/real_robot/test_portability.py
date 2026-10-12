"""The extracted core binds host dependencies explicitly, without local SDKs."""
from functools import partial
import json

import pytest

from PhysicalRSI_baselines.real_robot.contracts import Task
from PhysicalRSI_baselines.real_robot.evolution import PairedEpisodeEvaluator
from PhysicalRSI_baselines.real_robot.host_environment import (
    capture_environment, verify_trial_environment,
)
from PhysicalRSI_baselines.real_robot.library import PolicyLibrary


def test_physical_evaluation_requires_host_capture(tmp_path):
    config = tmp_path / "robot.json"
    config.write_text('{}')
    capture = partial(capture_environment, files={"config": config},
                      package_versions={"test-host": "1"})
    args = dict(task=Task("test", "Align", "Visible alignment"),
                scene_provider=lambda *_: [], trial_runner=lambda *_: None,
                identity={"host": "test"}, episodes=1, qualification="physical",
                environment=capture())
    with pytest.raises(ValueError, match="host environment capture"):
        PairedEpisodeEvaluator(**args)
    evaluator = PairedEpisodeEvaluator(**args, environment_capture=capture)
    assert evaluator.identity()["environment"] == capture()


def test_all_environment_receipts_must_match_and_be_committed(tmp_path):
    config = tmp_path / "robot.json"
    config.write_text('{}')
    capture = partial(capture_environment, files={"config": config},
                      package_versions={"test-host": "1"})
    expected = capture()
    for name in ("evaluation/environment-before.json", "hardware-owner/environment.json",
                 "evaluation/environment-after.json"):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(expected))
    verify_trial_environment(tmp_path, expected)
    with pytest.raises(ValueError, match="committed"):
        verify_trial_environment(tmp_path, expected, evidence={})
    config.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="changed"):
        verify_trial_environment(tmp_path, capture())


def test_explicit_skill_source_is_immutable_candidate_input(tmp_path):
    task = Task("test", "Align", "Visible alignment")
    skill = dict(name="align", source="def align(robot):\n    robot.reobserve()\n",
                 qualification="unvalidated-development-candidate")
    library = PolicyLibrary(tmp_path / "library", foundation={"model": "fixture"},
                            reusable_skills=[skill])
    skill["source"] = "caller mutation"
    parent = library.freeze(task=task, source="def policy(robot, memory):\n    robot.reobserve()\n", memory={})
    from pathlib import Path
    prompts = json.loads((Path(parent["root"]) / "prompts.json").read_text())
    assert prompts["reusable_skills"][0]["source"] != "caller mutation"
    library.reusable_skills[0]["source"] += "\n"
    child = library.freeze(task=task, source="def policy(robot, memory):\n    robot.reobserve()\n", memory={})
    assert child["id"] != parent["id"]
