#!/usr/bin/env bash
# Arm O: serve Ornith on the ORIGINAL pre-graphs image (577e1a932 rebuild,
# oneAPI 2025.3 / torch 2.12), eager, with the production-equivalent
# is_padding runtime patch applied inside the container at startup.
# Requires the GPU to be free (05's server down). On any failure the
# container is removed; on success it is LEFT RUNNING for run_arm.sh —
# tear down with: docker rm -f ornith-arm-o
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM_O_IMAGE="${ARM_O_IMAGE:?e.g. hal/vllm-xpu:kris-fork-577e1a932-rebuild}"
MODEL_HOST="${MODEL_HOST:?host path to Ornith model dir}"
NAME="${NAME:-ornith-arm-o}"
PORT="${PORT:-8021}"
SERVED="${SERVED:-ornith-arm-o}"
# Profile-consistency rule: set these to match whatever profile 05 serves
# arms G/E with.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
GPU_UTIL="${GPU_UTIL:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
DOCKER="${DOCKER:-docker}"

${DOCKER} rm -f "${NAME}" 2>/dev/null || true

${DOCKER} run -d --name "${NAME}" --privileged --device /dev/dri \
  -p "${PORT}:8000" \
  -e ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}" \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=0 \
  -v "${DIR}/patch_topk_is_padding.py:/opt/bench/patch_topk_is_padding.py:ro" \
  -v "${MODEL_HOST}:/model:ro" \
  --entrypoint bash "${ARM_O_IMAGE}" -c "
set -e
python3 /opt/bench/patch_topk_is_padding.py \
  \$(python3 -c 'import vllm._custom_ops as m; print(m.__file__)') \
  /workspace/vllm/vllm/_custom_ops.py 2>/dev/null || true
exec vllm serve /model \
  --host 0.0.0.0 --port 8000 \
  --served-model-name ${SERVED} \
  --max-model-len ${MAX_MODEL_LEN} \
  --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
  --max-num-seqs ${MAX_NUM_SEQS} \
  --gpu-memory-utilization ${GPU_UTIL} \
  --dtype bfloat16 \
  --kv-cache-dtype ${KV_CACHE_DTYPE} \
  --trust-remote-code \
  --enforce-eager \
  --limit-mm-per-prompt '{\"image\":0}' \
  --default-chat-template-kwargs '{\"enable_thinking\": false}'
"

fail() {
  echo "ARM O FAIL: $1" >&2
  ${DOCKER} logs "${NAME}" 2>&1 | tail -60 >&2 || true
  ${DOCKER} rm -f "${NAME}" >/dev/null 2>&1 || true
  exit 1
}

T0=$(date +%s)
for _ in $(seq 1 $((READY_TIMEOUT / 5))); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "READY after $(($(date +%s) - T0))s"
    break
  fi
  st=$(${DOCKER} inspect -f '{{.State.Status}}' "${NAME}" 2>/dev/null || echo gone)
  [[ "${st}" == "running" ]] || fail "container ${st}"
  sleep 5
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || fail "not ready"

# Sanity smoke (S2-style): greedy 2+2 must contain "4"; short decode must be
# non-empty and loop-free. Correctness for arms G/E is 04's + 05's job.
resp=$(curl -sf "http://127.0.0.1:${PORT}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${SERVED}\",\"prompt\":\"What is 2+2? Answer with just the number: \",\"max_tokens\":8,\"temperature\":0}")
echo "${resp}" | grep -q '"text"' || fail "empty S2 response"
echo "${resp}" | python3 -c '
import json, sys
text = json.load(sys.stdin)["choices"][0]["text"]
assert "4" in text, f"S2 wrong answer: {text!r}"
print(f"S2 OK: {text.strip()!r}")' || fail "S2 wrong answer"

resp=$(curl -sf "http://127.0.0.1:${PORT}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${SERVED}\",\"prompt\":\"Explain what a GPU does in two sentences.\",\"max_tokens\":64,\"temperature\":0}")
echo "${resp}" | python3 -c '
import json, sys
text = json.load(sys.stdin)["choices"][0]["text"]
assert len(text.strip()) > 20, "S3 too short"
assert "!!!!" not in text, "S3 garbage loop"
print(f"S3 OK: {text.strip()[:80]!r}")' || fail "S3 bad decode"

echo "arm O serving on :${PORT} — bench with run_arm.sh, then: ${DOCKER} rm -f ${NAME}"
