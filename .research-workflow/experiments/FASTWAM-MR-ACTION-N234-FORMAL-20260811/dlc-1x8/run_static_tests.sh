#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
/usr/bin/python3 -B -I -S - "${SCRIPT_DIR}/controller.py" "${SCRIPT_DIR}/test_static.py" <<'PY'
import sys
from pathlib import Path

for literal in sys.argv[1:]:
    path = Path(literal)
    compile(path.read_bytes(), str(path), "exec")
PY
/bin/bash -n "${SCRIPT_DIR}/runtime.sh" "${SCRIPT_DIR}/submit_from_ssh970.sh"
/usr/bin/python3 -B -I -S "${SCRIPT_DIR}/test_static.py"
