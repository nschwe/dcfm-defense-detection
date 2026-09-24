#!/usr/bin/env bash
# fp_importance.sh -- rank the single-feature AUC of every observable in the
# reported FPNT campaign. A thin wrapper around fp_importance.py.
#
# Both paths are resolved at run time so a clone runs this as it stands:
#   * the analysis script is taken from beside this wrapper;
#   * the interpreter is $PY when the caller exports one, the manet environment
#     when that environment is present, and python3 otherwise.
#
# Any argument is forwarded, so another bundle directory can be ranked with
#     bash fp_importance.sh /path/to/listener/colab_data
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/fp_importance.py"
[ -f "$SCRIPT" ] || {
  echo "ERROR: fp_importance.py not found beside this wrapper: $SCRIPT" >&2
  exit 1
}

PYBIN="${PY:-}"
if [ -z "$PYBIN" ]; then
  if [ -x "$HOME/miniconda3/envs/manet/bin/python" ]; then
    PYBIN="$HOME/miniconda3/envs/manet/bin/python"
  else
    PYBIN=python3
  fi
fi

exec "$PYBIN" -u "$SCRIPT" "$@"
