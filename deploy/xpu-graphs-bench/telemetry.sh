#!/usr/bin/env bash
# xpu-smi dump sidecar: start | stop <csv> <out_json>
# start prints "PID CSV_PATH"; stop kills the sampler and summarizes every
# numeric CSV column (mean/min/max) + duration; a column whose header mentions
# Power also yields approx_joules = mean_power * duration.
set -euo pipefail

CMD="${1:?start|stop}"
XPU_SMI_BIN="${XPU_SMI_BIN:-xpu-smi}"
XPU_SMI_DEVICE="${XPU_SMI_DEVICE:-0}"
# 0=util 1=power 2=freq 3=temp 5=mem-used 18=mem-bandwidth (see xpu-smi dump -h)
XPU_SMI_METRICS="${XPU_SMI_METRICS:-0,1,2,3,5,18}"

case "${CMD}" in
  start)
    CSV="${2:?csv path}"
    nohup "${XPU_SMI_BIN}" dump -d "${XPU_SMI_DEVICE}" \
      -m "${XPU_SMI_METRICS}" -i 1 >"${CSV}" 2>/dev/null &
    echo "$! ${CSV}"
    ;;
  stop)
    PID="${2:?sampler pid}"
    CSV="${3:?csv path}"
    OUT="${4:?out json}"
    kill "${PID}" 2>/dev/null || true
    python3 - "${CSV}" "${OUT}" <<'PY'
import csv, json, statistics, sys

csv_path, out_path = sys.argv[1], sys.argv[2]
rows = list(csv.reader(open(csv_path)))
if len(rows) < 2:
    json.dump({"error": "no samples"}, open(out_path, "w"))
    raise SystemExit
header = [h.strip() for h in rows[0]]
cols = {h: [] for h in header}
for row in rows[1:]:
    for h, v in zip(header, row):
        try:
            cols[h].append(float(v))
        except ValueError:
            pass
summary = {"n_samples": len(rows) - 1}
duration = len(rows) - 1  # 1s interval
summary["duration_s"] = duration
for h, vals in cols.items():
    if not vals or h.lower() in ("timestamp", "deviceid"):
        continue
    stat = {
        "mean": round(statistics.fmean(vals), 2),
        "min": round(min(vals), 2),
        "max": round(max(vals), 2),
    }
    summary[h] = stat
    if "power" in h.lower():
        summary["approx_joules"] = round(stat["mean"] * duration, 1)
json.dump(summary, open(out_path, "w"), indent=2)
print(f"wrote {out_path}")
PY
    ;;
  *) echo "usage: telemetry.sh start <csv> | stop <pid> <csv> <out_json>" >&2; exit 2 ;;
esac
