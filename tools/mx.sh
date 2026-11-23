#!/usr/bin/env bash
# Run Python in the live marimo kernel (marimo-pair skill). Needs:
#   MARIMO_URL         e.g. https://sb-xxxxxxxx.sb.molab.run/
#   MARIMO_TOKEN_FILE  file holding the marimo access token (never commit it)
#   MARIMO_PAIR_DIR    directory of the marimo-pair skill (contains scripts/execute-code.sh)
set -euo pipefail
MARIMO_TOKEN="$(cat "${MARIMO_TOKEN_FILE:?}")" exec bash "${MARIMO_PAIR_DIR:?}/scripts/execute-code.sh" --url "${MARIMO_URL:?}" "$@"
