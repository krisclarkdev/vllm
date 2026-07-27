#!/usr/bin/env bash
# Bench one arm end-to-end against an ALREADY-RUNNING server (05's arms G/E,
# or arm O started by serve_arm_o.sh). Never starts or stops servers itself.
#   ARM=G BASE_URL=http://127.0.0.1:8004 MODEL=ornith \
#     MODEL_CONFIG=/models/Ornith-1.0-35B-MXFP4/config.json bash run_arm.sh
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM="${ARM:?G|E|O}"
BASE_URL="${BASE_URL:?e.g. http://127.0.0.1:8021}"
MODEL="${MODEL:?served model name}"
MODEL_CONFIG="${MODEL_CONFIG:-}"
RESULTS_DIR="${RESULTS_DIR:-${DIR}/results}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8}"
LENGTHS="${LENGTHS:-128:128 2048:256 8192:256}"
PEAK_BF16_TFLOPS="${PEAK_BF16_TFLOPS:-0}"
ACTIVE_PARAMS_B="${ACTIVE_PARAMS_B:-0}"
TELEMETRY="${TELEMETRY:-1}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
mkdir -p "${RESULTS_DIR}"

SWEEP_JSON="${RESULTS_DIR}/bench_${ARM}_${STAMP}.json"
FLOPS_JSON="${RESULTS_DIR}/flops_${ARM}_${STAMP}.json"
TELEM_CSV="${RESULTS_DIR}/telemetry_${ARM}_${STAMP}.csv"
TELEM_JSON="${RESULTS_DIR}/telemetry_${ARM}_${STAMP}.json"

curl -sf "${BASE_URL}/health" >/dev/null || {
  echo "FAIL: ${BASE_URL}/health not responding" >&2
  exit 1
}
curl -sf "${BASE_URL}/metrics" \
  >"${RESULTS_DIR}/metrics_${ARM}_${STAMP}_before.txt" || true

TELEM_PID=""
if [[ "${TELEMETRY}" == "1" ]]; then
  read -r TELEM_PID _ < <(bash "${DIR}/telemetry.sh" start "${TELEM_CSV}") || true
fi
cleanup() {
  if [[ -n "${TELEM_PID}" ]]; then
    bash "${DIR}/telemetry.sh" stop "${TELEM_PID}" "${TELEM_CSV}" \
      "${TELEM_JSON}" || true
    TELEM_PID=""
  fi
}
trap cleanup EXIT

"${PYTHON_BIN}" "${DIR}/bench_sweep.py" \
  --base-url "${BASE_URL}" --model "${MODEL}" --arm "${ARM}" \
  --concurrencies "${CONCURRENCIES}" --lengths "${LENGTHS}" \
  --out "${SWEEP_JSON}"

cleanup
trap - EXIT
curl -sf "${BASE_URL}/metrics" \
  >"${RESULTS_DIR}/metrics_${ARM}_${STAMP}_after.txt" || true

if [[ -n "${MODEL_CONFIG}" && -f "${MODEL_CONFIG}" ]]; then
  "${PYTHON_BIN}" "${DIR}/flops.py" \
    --config "${MODEL_CONFIG}" --bench-json "${SWEEP_JSON}" \
    --peak-tflops "${PEAK_BF16_TFLOPS}" \
    --active-params-b "${ACTIVE_PARAMS_B}" \
    --out "${FLOPS_JSON}"
else
  echo "MODEL_CONFIG unset/missing — skipping TFLOPS derivation" >&2
fi

echo "arm ${ARM} done: ${SWEEP_JSON}"
