#!/usr/bin/env bash
set -euo pipefail
: "${XPOLICYLAB_ROOT:?Set XPOLICYLAB_ROOT to the installed XPolicyLab checkout}"
exec bash "${XPOLICYLAB_ROOT}/policy/physicalRSI/eval.sh" "$@"
