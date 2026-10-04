from pathlib import Path

from PhysicalRSI_baselines.robodojo.provider import describe_provider, register


def test_external_provider_manifest_pins_code_and_checkpoint_without_copying(tmp_path):
    provider = tmp_path / "Pi_05"
    provider.mkdir()
    for name in ("model.py", "process_data.sh", "train.sh", "eval.sh", "deploy.yml"):
        (provider / name).write_text(name + "\n")
    checkpoint = provider / "checkpoints" / "run"
    checkpoint.mkdir(parents=True)
    (checkpoint / "service.json").write_text("{}\n")
    output = tmp_path / "provider.json"

    manifest = register(provider, output, policy="pi05", checkpoint=checkpoint)

    assert manifest["weights"] == "external; not copied into this release"
    assert str(checkpoint / "service.json") in manifest["files"]
    assert manifest["checkpoint_files"][str(checkpoint / "service.json")]
    assert Path(manifest["provider_root"]) == provider
    assert output.is_file()
