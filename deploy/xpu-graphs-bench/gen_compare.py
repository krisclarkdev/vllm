#!/usr/bin/env python3
"""Merge bench/flops/telemetry JSON for arms G/E/O into a 3-way report.

Usage: gen_compare.py <results_dir> <stamp>
Writes GRAPHS_FEATURE_COMPARE_<stamp>.{json,md} over whatever arms completed.
"""

import json
import os
import sys

results_dir, stamp = sys.argv[1], sys.argv[2]

ARM_NAMES = {
    "G": "G: feature image, graphs ON (FA-in-graph)",
    "E": "E: feature image, graphs OFF (eager)",
    "O": "O: original pre-graphs image (eager)",
}


def load(kind, arm):
    path = os.path.join(results_dir, f"{kind}_{arm}_{stamp}.json")
    return json.load(open(path)) if os.path.exists(path) else None


arms = [a for a in ("O", "E", "G") if load("bench", a)]
data = {
    a: {
        "bench": load("bench", a),
        "flops": load("flops", a),
        "telemetry": load("telemetry", a),
    }
    for a in arms
}


def combo_key(combo):
    return (
        combo["concurrency"],
        combo["target_in_tokens"],
        combo["target_out_tokens"],
    )


def indexed_combos(arm):
    return {combo_key(c): c for c in data[arm]["bench"]["combos"] if "error" not in c}


def flops_for(arm, key):
    fl = data[arm].get("flops")
    if not fl:
        return {}
    for entry in fl["per_combo"]:
        if combo_key(entry) == key:
            return entry
    return {}


all_keys = sorted({k for a in arms for k in indexed_combos(a)})
rows = []
for key in all_keys:
    conc, in_tok, out_tok = key
    row = {"concurrency": conc, "in_tokens": in_tok, "out_tokens": out_tok}
    for a in arms:
        combo = indexed_combos(a).get(key)
        if not combo:
            continue
        row[a] = {
            "ttft_ms_p50": combo["ttft_ms"]["p50"],
            "ttft_ms_p99": combo["ttft_ms"]["p99"],
            "tpot_ms_p50": (combo["tpot_ms"] or {}).get("p50"),
            "itl_ms_p50": (combo["itl_ms"] or {}).get("p50"),
            "output_tok_per_s": combo["output_tok_per_s"],
            "input_tok_per_s": combo["input_tok_per_s"],
            "req_per_s": combo["req_per_s"],
            **{
                k: v
                for k, v in flops_for(a, key).items()
                if "tflops" in k or "mfu" in k
            },
        }
    rows.append(row)


deltas = {}
for hi, lo in (("G", "E"), ("G", "O"), ("E", "O")):
    if hi not in arms or lo not in arms:
        continue
    per = []
    for row in rows:
        if hi in row and lo in row:
            base = row[lo]["output_tok_per_s"]
            per.append(
                {
                    "combo": f"C{row['concurrency']} "
                    f"{row['in_tokens']}in/{row['out_tokens']}out",
                    "decode_pct": round(
                        (row[hi]["output_tok_per_s"] - base) / base * 100, 1
                    ),
                    "ttft_p50_pct": round(
                        (row[hi]["ttft_ms_p50"] - row[lo]["ttft_ms_p50"])
                        / row[lo]["ttft_ms_p50"]
                        * 100,
                        1,
                    ),
                }
            )
    deltas[f"{hi}_vs_{lo}"] = per

summary = {
    "stamp": stamp,
    "arms_completed": arms,
    "per_combo": rows,
    "deltas": deltas,
    "telemetry": {a: data[a]["telemetry"] for a in arms if data[a]["telemetry"]},
    "flops_meta": {
        a: {k: v for k, v in (data[a]["flops"] or {}).items() if k != "per_combo"}
        for a in arms
        if data[a]["flops"]
    },
}
out_json = os.path.join(results_dir, f"GRAPHS_FEATURE_COMPARE_{stamp}.json")
with open(out_json, "w") as fh:
    json.dump(summary, fh, indent=2)

lines = [
    f"# XPU graphs feature 3-way compare — {stamp}",
    "",
    "Arms: " + "; ".join(ARM_NAMES[a] for a in arms),
    "",
]
for row in rows:
    lines.append(
        f"## C={row['concurrency']}  {row['in_tokens']}in / {row['out_tokens']}out"
    )
    lines.append(
        "| Arm | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | decode tok/s | "
        "prefill tok/s | dec TFLOPS | pre TFLOPS |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for a in arms:
        if a not in row:
            continue
        d = row[a]

        def fmt(val, nd=1):
            return f"{val:.{nd}f}" if isinstance(val, (int, float)) else "-"

        lines.append(
            f"| {a} | {fmt(d['ttft_ms_p50'])} | {fmt(d['ttft_ms_p99'])} | "
            f"{fmt(d['tpot_ms_p50'])} | {fmt(d['output_tok_per_s'])} | "
            f"{fmt(d['input_tok_per_s'])} | {fmt(d.get('decode_tflops'), 2)} | "
            f"{fmt(d.get('prefill_tflops'), 2)} |"
        )
    lines.append("")
for pair, per in deltas.items():
    if per:
        lines.append(f"**{pair.replace('_', ' ')}:**")
        for entry in per:
            lines.append(
                f"- {entry['combo']}: decode {entry['decode_pct']:+.1f}%, "
                f"TTFT p50 {entry['ttft_p50_pct']:+.1f}%"
            )
        lines.append("")

md = "\n".join(lines)
out_md = os.path.join(results_dir, f"GRAPHS_FEATURE_COMPARE_{stamp}.md")
with open(out_md, "w") as fh:
    fh.write(md + "\n")
print(md)
