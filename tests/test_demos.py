from pathlib import Path

from PhysicalRSI_demos.runtime import run as run_runtime
from PhysicalRSI_demos.runtime_evolution import run as run_evolution


def test_runtime_demo_records_a_valid_episode(tmp_path):
    result = run_runtime(tmp_path / "runtime")
    assert result["result"]["environment_success"] is True
    assert result["metadata"]["is_success"] is True
    assert Path(result["episode"]).is_dir()


def test_self_harness_demo_selects_and_persists_candidate(tmp_path):
    result = run_evolution(tmp_path / "evolution")
    assert result["state"] == "selection"
    assert result["harness"] == "memory_candidate"
    assert (tmp_path / "evolution" / "state" / "current.json").exists()
