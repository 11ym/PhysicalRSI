from PhysicalRSI_demos.skill_memory import build_catalog
from PhysicalRSI.application import Application


def test_skill_memory_catalog_binds_core_skills_to_one_memory_revision(tmp_path):
    catalog = build_catalog(tmp_path / "workspace")

    assert set(catalog["skills"]) == {"pi05", "pi05-sparse-memory", "code-policy"}
    assert catalog["skill_choices"]["stack_bowls"]["skill"] == "pi05-sparse-memory"
    assert catalog["skill_choices"]["fold_clothes"]["skill"] == "code-policy"
    assert catalog["systems"]["system1"]["skills"] == [
        "pi05",
        "pi05-sparse-memory",
        "code-policy",
    ]
    assert "selection" in catalog["systems"]["system2"]["components"]
    revisions = {skill_choice["memory_revision"] for skill_choice in catalog["skill_choices"].values()}
    assert revisions == {catalog["memory"]["revision"]}


def test_cli_exposes_skill_memory_demo(tmp_path):
    result = Application(tmp_path / "workspace").command("/skill-memory")
    assert result["schema"] == "physicalrsi.skill-memory-demo/v1"
    assert (tmp_path / "workspace" / "skill-memory" / "memory").is_dir()
