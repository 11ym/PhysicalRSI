#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: check_xpolicylab_adapter.sh XPolicyLab_checkout [policy_venv]' >&2
  exit 2
fi
XPL="$(cd "$1" && pwd)"
POLICY="${XPL}/policy/physicalRSI"
[[ -d "${POLICY}" ]] || { echo "Missing ${POLICY}" >&2; exit 1; }

required=(README.md __init__.py install.sh eval.sh setup_eval_policy_server.sh
  setup_eval_env_client.sh deploy.yml deploy.py model.py)
for name in "${required[@]}"; do
  [[ -f "${POLICY}/${name}" ]] || { echo "Missing adapter file: ${name}" >&2; exit 1; }
done

bash -n "${POLICY}"/*.sh
python -m py_compile "${POLICY}/model.py" "${POLICY}/deploy.py" "${POLICY}/code_policy_demo.py"
if rg -n 'cv2\.imdecode|np\.frombuffer|Image\.open|cv2\.imencode|COLOR_BGR2RGB|COLOR_RGB2BGR' "${POLICY}" --glob '*.py'; then
  echo 'Adapter image conversion must use XPolicyLab process_data helpers.' >&2
  exit 1
fi

if [[ $# == 2 ]]; then
  VENV="$(cd "$2" && pwd)"
  [[ -x "${VENV}/bin/python" ]] || { echo "Missing policy interpreter: ${VENV}" >&2; exit 1; }
  echo "Static adapter checks passed. Run the official debug loop with a release:"
  echo "PHYSICALRSI_EVAL_OUTPUT=/tmp/physicalrsi-debug EVAL_ENV_TYPE=debug bash ${POLICY}/eval.sh RoboDojo stack_bowls <checkpoint> arx_x5 joint 0 0 0 ${VENV} base"
else
  echo "Static adapter checks passed. Supply the policy venv to print the debug command."
fi
