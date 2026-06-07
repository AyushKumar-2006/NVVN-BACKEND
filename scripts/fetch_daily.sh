#!/usr/bin/env bash
#
# fetch_daily.sh — daily WRLDC PSP refresh for the CG (Chhattisgarh) demand series.
#
# Activates the project's Python environment and runs:
#     python manage.py fetch_wrldc_psp --years <CURRENT_YEAR>
# upserting the latest real Chhattisgarh demand points into StateLoad5Min
# (state='CG_WRLDC'). Safe to run repeatedly — fetch_wrldc_psp upserts and never
# deletes, so re-running only fills in / corrects rows.
#
# Intended to be driven by cron (see README / `python manage.py setup_cron`):
#     0 2 * * * bash ~/NVVN-backend/scripts/fetch_daily.sh
#
# Environment activation order (first that exists wins):
#   1. $NVVN_PYTHON           — explicit python interpreter, if exported
#   2. <project>/venv  or  <project>/.venv   — a local virtualenv
#   3. conda env named "$NVVN_CONDA_ENV" (if set) via `conda activate`
#   4. whatever `python3`/`python` is on PATH
#
set -euo pipefail

# ---- resolve the project root (parent of this scripts/ dir) ------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${PROJECT_DIR}"

# ---- logging -----------------------------------------------------------------
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/fetch_daily.log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"; }

YEAR="$(date '+%Y')"
log "=== fetch_daily start (project=${PROJECT_DIR}, year=${YEAR}) ==="

# ---- pick a python interpreter -----------------------------------------------
PY=""
if [[ -n "${NVVN_PYTHON:-}" && -x "${NVVN_PYTHON}" ]]; then
  PY="${NVVN_PYTHON}"
  log "using \$NVVN_PYTHON: ${PY}"
elif [[ -f "${PROJECT_DIR}/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_DIR}/venv/bin/activate"
  PY="python"
  log "activated venv: ${PROJECT_DIR}/venv"
elif [[ -f "${PROJECT_DIR}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_DIR}/.venv/bin/activate"
  PY="python"
  log "activated venv: ${PROJECT_DIR}/.venv"
elif [[ -n "${NVVN_CONDA_ENV:-}" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${NVVN_CONDA_ENV}"
  PY="python"
  log "activated conda env: ${NVVN_CONDA_ENV}"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
  log "using system python3: $(command -v python3)"
else
  PY="python"
  log "using system python: $(command -v python || echo 'NOT FOUND')"
fi

# ---- run the WRLDC fetch -----------------------------------------------------
rc=0
log "running: ${PY} manage.py fetch_wrldc_psp --years ${YEAR}"
if "${PY}" manage.py fetch_wrldc_psp --years "${YEAR}" >>"${LOG_FILE}" 2>&1; then
  log "fetch_wrldc_psp OK"
else
  rc=$?
  log "!!! fetch_wrldc_psp FAILED (exit ${rc}) — see ${LOG_FILE}"
fi

# ---- bring the live CG series up to today (5-min load + 4-district weather) ---
# Idempotent: fills only the gap from the last stored point to today, so the
# dashboard's intraday/forecast/temperature stay current. Runs even if the WRLDC
# fetch above failed (they are independent).
log "running: ${PY} manage.py backfill_cg_live"
if "${PY}" manage.py backfill_cg_live >>"${LOG_FILE}" 2>&1; then
  log "backfill_cg_live OK"
else
  brc=$?
  log "!!! backfill_cg_live FAILED (exit ${brc}) — see ${LOG_FILE}"
  [ "${rc}" -eq 0 ] && rc="${brc}"
fi

if [ "${rc}" -eq 0 ]; then
  log "=== fetch_daily OK ==="
else
  log "!!! fetch_daily completed with errors (exit ${rc}) — see ${LOG_FILE}"
fi
exit "${rc}"
