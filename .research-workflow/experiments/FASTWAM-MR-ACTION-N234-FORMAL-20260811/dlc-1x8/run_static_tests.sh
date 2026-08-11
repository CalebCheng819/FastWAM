#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
/usr/bin/python3 -m py_compile "${SCRIPT_DIR}/controller.py" "${SCRIPT_DIR}/test_static.py"
/bin/bash -n "${SCRIPT_DIR}/runtime.sh" "${SCRIPT_DIR}/submit_from_ssh970.sh"
/usr/bin/python3 "${SCRIPT_DIR}/test_static.py"
