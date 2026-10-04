#!/usr/bin/env bash
set -euo pipefail
checkout=${1:?usage: install_xpolicylab.sh /path/to/XPolicyLab}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -d "$root/skills/source/PhysicalRSI_baselines" ]]; then
  repo_root="$root/skills/source"
else
  repo_root=$(cd "$root/../../../.." && pwd)
fi
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
python -m PhysicalRSI_baselines.robodojo.xpolicy_package "$checkout"
python -m PhysicalRSI_baselines.robodojo.harness_bundle "$checkout" --bundle "$root"
printf 'Installed PhysicalRSI executable skill adapter and harness descriptors (run preflight before evaluation) under %s/policy/physicalRSI\n' "$checkout"
