# PhysicalRSI-baselines

The RoboDojo baseline imports **PhysicalRSI's XPolicyLab PR #147** as a pinned git submodule. This organization-owned submission carries forward yanming03's original PR #144. It preserves the complete fork, its adapter runtime, evaluation scripts, source and license notices.

- PR: https://github.com/XPolicyLab/XPolicyLab/pull/147
- Fork: https://github.com/PhysicalRSI/XPolicyLab
- Commit: `393f730788e7277c7d86829dd5806a924ac31e74`
- Adapter: `robodojo/XPolicyLab/policy/physicalRSI/`
- Machine-readable provenance: [`robodojo/provenance.json`](robodojo/provenance.json)

The commit is pinned regardless of later PR updates. The surrounding `robodojo/` modules retain the preview release's task integration and frozen bundle tools; the submodule is the authoritative PR snapshot. Installation does not replace either source tree with the other.

## Initialize

From the PhysicalRSI repository root:

```bash
git submodule update --init PhysicalRSI_baselines/robodojo/XPolicyLab
physicalrsi --command /robodojo
```

Follow the adapter's [installation instructions](robodojo/XPolicyLab/policy/physicalRSI/README.md) to download its checksum-verified assets, prepare code programs and create a skill configuration. Retain the XPolicyLab and dependency licenses. The adapter requires its model assets, simulator, policy interpreter and an image-capable agent endpoint; these are not bundled with the lightweight CLI.

## Run from the terminal or workbench

Configure before opening the workbench:

```bash
export PHYSICALRSI_XPOLICYLAB_ROOT="$PWD/PhysicalRSI_baselines/robodojo/XPolicyLab"
export PHYSICALRSI_SKILL_CONFIG=/path/to/skills.json
export PHYSICALRSI_POLICY_PYTHON=/path/to/policy/python
export ROBODOJO_CONDA_ENV=your_robodojo_environment
# Set PHYSICALRSI_AGENT_API_KEY through your environment/secret manager.
physicalrsi
```

```text
/robodojo general_pickup
/jobs
/logs
/baselines
```

For a simulator in a virtual environment, use `PHYSICALRSI_SIM_PYTHON` instead of `ROBODOJO_CONDA_ENV`. `PHYSICALRSI_POLICY_GPU` and `PHYSICALRSI_ENV_GPU` select devices. Jobs record their actual process status, logs and result directory. Archived videos are optional and are discovered under `PHYSICALRSI_DATA_ROOT`; importing the PR does not assert a leaderboard result.

The baseline is evaluation-only. Dexjoco data generation and continual training are implemented separately under `PhysicalRSI_demos/showcase/`.
