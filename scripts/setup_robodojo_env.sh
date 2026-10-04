#!/usr/bin/env bash
set -euo pipefail

# Create the lightweight policy environment and install this preview's adapter.
# Isaac Sim and benchmark assets remain external because their licenses and GPU
# requirements are controlled by the benchmark repositories.
usage() {
  cat >&2 <<'EOF'
Usage: setup_robodojo_env.sh [options]

Options:
  --source DIR          PhysicalRSI checkout (default: this checkout)
  --xpolicylab DIR      existing XPolicyLab checkout to receive policy/physicalRSI
  --policy-venv DIR     policy virtualenv (default: <source>/.venv-robodojo-policy)
  --install-upstreams   install this checkout and the XPolicyLab checkout in the venv
  --help

This script does not download Isaac Sim, benchmark assets, checkpoints, or
model weights. Those remain governed by RoboDojo/XPolicyLab installation rules.
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XPL=""
VENV="${ROOT}/.venv-robodojo-policy"
INSTALL_UPSTREAMS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) ROOT="$(cd "$2" && pwd)"; shift 2 ;;
    --xpolicylab) XPL="$(cd "$2" && pwd)"; shift 2 ;;
    --policy-venv) VENV="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"; shift 2 ;;
    --install-upstreams) INSTALL_UPSTREAMS=1; shift ;;
    --help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

[[ -f "${ROOT}/pyproject.toml" ]] || { echo "Not a PhysicalRSI checkout: ${ROOT}" >&2; exit 2; }
if [[ -n "${XPL}" && ! -f "${XPL}/model_template.py" ]]; then
  echo "Not an XPolicyLab checkout (model_template.py missing): ${XPL}" >&2
  exit 2
fi

mkdir -p "$(dirname "${VENV}")"
python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip
"${VENV}/bin/python" -m pip install -e "${ROOT}[robodojo]"
if [[ "${INSTALL_UPSTREAMS}" == 1 && -n "${XPL}" ]]; then
  "${VENV}/bin/python" -m pip install -e "${XPL}"
fi
if [[ -n "${XPL}" ]]; then
  "${VENV}/bin/python" -m PhysicalRSI_baselines.robodojo.xpolicy_package "${XPL}"
  "${ROOT}/PhysicalRSI_baselines/robodojo/xpolicylab_policy/install.sh" "${ROOT}" "${VENV}"
fi

if [[ -n "${XPL}" ]]; then
  ADAPTER="${XPL}/policy/physicalRSI"
else
  ADAPTER="${ROOT}/PhysicalRSI_baselines/robodojo/xpolicylab_policy"
fi

cat <<EOF
Environment ready:
  policy python: ${VENV}/bin/python
  adapter: ${XPL:-not installed; pass --xpolicylab DIR to install it}

Next checks:
  PHYSICALRSI_EVAL_OUTPUT=/tmp/physicalrsi-debug-\$(date +%s) \\
    EVAL_ENV_TYPE=debug bash ${ADAPTER}/eval.sh \\
    RoboDojo stack_bowls <checkpoint> arx_x5 joint 0 0 0 ${VENV} base
EOF
