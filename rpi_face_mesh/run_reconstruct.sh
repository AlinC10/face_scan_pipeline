#!/usr/bin/env bash
# Python: $PYTHON, else ./.venv, else the scanner's venv (../.venv), else python3.
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
if [ -n "$PYTHON" ]; then PY="$PYTHON"
elif [ -x "$DIR/.venv/bin/python" ]; then PY="$DIR/.venv/bin/python"
elif [ -x "$DIR/../.venv/bin/python" ]; then PY="$DIR/../.venv/bin/python"
else PY=python3; fi
cd "$DIR"
exec "$PY" "$DIR/reconstruct_3d.py" "$@"
