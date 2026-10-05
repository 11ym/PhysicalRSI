# PhysicalRSI skill harness bundle

Install the adapter with `install_xpolicylab.sh /path/to/XPolicyLab`. This copies
the execution runtime and harness descriptors. The installed adapter README
explains skill configuration, API credentials, and environment preparation.

Set `XPOLICYLAB_ROOT` and use `eval.sh` with XPolicyLab's ten evaluation arguments.
This script delegates to the installed adapter; it does not fabricate a result.

The runtime exposes one skill library containing pi05, pi05-sparse-memory and
code-policy implementations, with separate weight identities recorded explicitly.
At episode start an API-backed agent reads instructions, observations and immutable
task-aware memory, then chooses a registered, configured skill composition. That
choice determines the actual execution skill. The composition remains stable until
reset; observations, policy instructions and actions are not rewritten. Missing
credentials or an invalid choice stop execution without a fallback.

## Evaluation-only scope

This integration performs inference and evaluation only. Data processing and
training are unsupported; `process_data.sh` and `train.sh` are intentionally absent.
Checkpoint architecture/configuration metadata is used solely for model loading.
There is no training-release commitment. Official acceptance still depends on the
benchmark's review of the complete evaluated system and its evidence.

## Downloaded skill assets

The adapter downloads the asset manifest together with the library and verifies
its checksum. The default library matches [XPolicyLab PR #147](https://github.com/XPolicyLab/XPolicyLab/pull/147).
Use the installed adapter's `download_assets.py --output skill-assets` command.
An existing asset directory and generated skill configuration are not upgraded
automatically; install into a fresh directory and regenerate the configuration.
Code sources remain readable. The library still contains task-specific geometry
assumptions; the release does not claim that every code skill is layout-independent.
