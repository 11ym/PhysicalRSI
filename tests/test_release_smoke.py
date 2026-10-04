import json
import subprocess
import sys
from pathlib import Path

from PhysicalRSI.application import Application


def test_one_command_baseline_and_persisted_plan(tmp_path):
    root = Path(__file__).resolve().parents[1] / "configs"
    app = Application(tmp_path / "workspace")
    app.command("/scheme learned_skill")
    app.command('/harness design {"name":"smoke","success_signal":"success"}')
    app.command("/task " + str(root / "quickstart.task.json"))
    result = app.command('/baseline {"goal":"place_block"}')
    assert result["result"]["success"] is True
    assert result["result"]["rsi_scheme"] == "learned_skill"
    assert (tmp_path / "workspace" / "rsi-plan.json").exists()


def test_play_alias_uses_the_same_task_boundary(tmp_path):
    root = Path(__file__).resolve().parents[1] / "configs"
    app = Application(tmp_path / "workspace")
    app.command("/task " + str(root / "quickstart.task.json"))
    result = app.command('/play {"goal":"place_block"}')
    assert result["result"]["success"] is True


def test_cli_starts_without_api_configuration(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "PhysicalRSI",
            "--plain",
            "--workspace",
            str(tmp_path / "workspace"),
            "--command",
            "/help",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    commands = json.loads(result.stdout)
    assert "/baseline" in commands
    assert "/play" in commands
    assert "/harness" in commands
    assert "/exit" in commands
    assert "/quit" not in commands


def test_hello_world_explains_model_configuration_without_an_api_key(tmp_path):
    result = Application(tmp_path / "workspace").command("hello world")
    assert result["message"] == "Hello world!"
    assert result["model"] is None
    assert result["configuration"]["command"] == "/model configs/model.example.json"


def test_fold_clothes_baseline_is_ready_out_of_the_box(tmp_path):
    task = Path(__file__).resolve().parents[1] / "configs" / "fold_clothes.task.json"
    app = Application(tmp_path / "workspace")
    app.command("/task " + str(task))
    result = app.command('/baseline {"garment":"shirt"}')
    assert result["check"]["ready"] is True
    assert result["result"]["task"] == "fold_clothes"
    assert result["result"]["success"] is True
    assert result["result"]["phases"][-1] == "return_home"


def test_evaluation_status_keeps_candidate_runs_separate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".release-staging" / "archive-dsw-eval"
    root.mkdir(parents=True)
    rows = {
        "earlier": {"results": [{"task": "example", "status": "complete", "score": 10}]},
        "candidate": {"results": [{"task": "example", "status": "missing_result"}]},
    }
    (root / "LIVE_COMPARISON.json").write_text(json.dumps({"runs": rows}))
    result = Application(tmp_path / "workspace").command("/eval-status")
    assert result["runs"]["earlier"]["completed"] == 1
    assert result["runs"]["candidate"]["completed"] == 0
    assert result["runs"]["candidate"]["unresolved"] == ["example"]
    assert "score" not in result["runs"]["candidate"]["results"][0]
