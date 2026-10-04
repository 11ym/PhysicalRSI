# Portable executable skills

This directory contains the common skill descriptors, registered compositions,
and a complete PhysicalRSI runtime source export under `source/`. Each descriptor
points to its executable Python source. The source checksum manifest covers the
skill backends, memory implementation, primitive assembly runtime, and shared
contracts. The release installer can use this export without the original repo.

Large dependencies are listed in `../assets.json`. `../download_assets.py` checks
every part before assembling and extracting it. It refuses corrupt archives,
path traversal, and replacement of an existing installation.

```bash
python download_assets.py --output /path/to/skill-assets
python -m pip install /path/to/skill-assets/implementations/openpi/packages/openpi-client
python -m pip install /path/to/skill-assets/implementations/openpi
```

Run these commands with the inference environment's Python. Install matching
accelerator dependencies for that environment. Use the downloaded checkpoint
paths in your skill library configuration and run `skill_preflight` before eval.
The source tree supports inference only; no training or dataset processing entry
is provided by this submission.

Prepared Release URLs are not proof of publication. Check `publication_status`
in the asset manifest. A private repository or draft release cannot provide
anonymous public downloads. Weights are never committed to Git.

The code-policy implementation is included, but its particular primitive program,
frozen harness, and simulation services must be configured. A generic program
example does not constitute a verified policy for every benchmark task.
