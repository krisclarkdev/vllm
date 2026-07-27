#!/usr/bin/env python3
"""Full-sweep serving benchmark against an OpenAI-compatible vLLM endpoint.

Stdlib-only so it runs on the hal host (or in any container) with plain
python3. For each (concurrency, input-len:output-len) combo it issues greedy
streaming completions and records TTFT / TPOT / ITL / E2E percentiles plus
prefill ("encode") and decode token throughput.

Prompts are built to an exact token length via the server's /tokenize
endpoint and carry a unique per-request prefix so prefix caching (enabled on
some arms, not others) cannot skew the comparison. ``ignore_eos`` keeps
output token counts fixed; exact counts come from stream usage when the
server reports them.
"""

import argparse
import concurrent.futures
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request

API_KEY = os.environ.get("BENCH_API_KEY", "")


def _headers():
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


def post_json(url, payload, timeout=600):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=_headers()
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def count_tokens(base_url, model, text):
    out = post_json(f"{base_url}/tokenize", {"model": model, "prompt": text})
    return out["count"]


def build_prompt_template(base_url, model, target_tokens, filler="alpha beta "):
    """Return text whose token count is within 2% of target (excl. prefix).

    The caller prepends a short unique prefix per request; a fixed-width
    placeholder is included here so the measured length already accounts
    for it.
    """
    placeholder = "[req 000000-0000] "
    reps = max(1, target_tokens // 2)
    for _ in range(8):
        text = placeholder + filler * reps
        n = count_tokens(base_url, model, text)
        if abs(n - target_tokens) <= max(2, target_tokens // 50):
            return filler * reps, n
        reps = max(1, int(reps * target_tokens / max(n, 1)))
    return filler * reps, count_tokens(base_url, model, placeholder + filler * reps)


def one_request(base_url, model, prompt, max_tokens, timeout):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode(),
        headers=_headers(),
    )
    t0 = time.perf_counter()
    ttft = None
    last_tok_t = None
    itls = []
    chunks = 0
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices and choices[0].get("text"):
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                elif last_tok_t is not None:
                    itls.append(now - last_tok_t)
                last_tok_t = now
                chunks += 1
    e2e = time.perf_counter() - t0
    out_tokens = usage["completion_tokens"] if usage else chunks
    in_tokens = usage["prompt_tokens"] if usage else None
    tpot = (e2e - ttft) / (out_tokens - 1) if ttft and out_tokens > 1 else None
    return {
        "ttft_s": ttft,
        "e2e_s": e2e,
        "tpot_s": tpot,
        "itls_s": itls,
        "in_tokens": in_tokens,
        "out_tokens": out_tokens,
    }


def pct(values, q):
    if not values:
        return None
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, int(round(q / 100 * (len(vals) - 1)))))
    return vals[idx]


def summarize_ms(values_s):
    vals = [v for v in values_s if v is not None]
    if not vals:
        return None
    return {
        "mean": statistics.fmean(vals) * 1000,
        "p50": pct(vals, 50) * 1000,
        "p95": pct(vals, 95) * 1000,
        "p99": pct(vals, 99) * 1000,
    }


def run_combo(args, concurrency, in_tok, out_tok, rng):
    body, measured_in = build_prompt_template(args.base_url, args.model, in_tok)
    n_requests = args.requests or max(8, 2 * concurrency)

    def make_prompt():
        return f"[req {rng.randrange(10**6):06d}-{rng.randrange(10**4):04d}] {body}"

    for _ in range(args.warmup):
        one_request(args.base_url, args.model, make_prompt(), out_tok, args.timeout)

    results = []
    wall_t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [
            pool.submit(
                one_request,
                args.base_url,
                args.model,
                make_prompt(),
                out_tok,
                args.timeout,
            )
            for _ in range(n_requests)
        ]
        for fut in concurrent.futures.as_completed(futs):
            results.append(fut.result())
    wall = time.perf_counter() - wall_t0

    total_in = sum(r["in_tokens"] or measured_in for r in results)
    total_out = sum(r["out_tokens"] for r in results)
    all_itls = [itl for r in results for itl in r["itls_s"]]
    return {
        "concurrency": concurrency,
        "target_in_tokens": in_tok,
        "target_out_tokens": out_tok,
        "measured_in_tokens": measured_in,
        "n_requests": n_requests,
        "wall_s": wall,
        "req_per_s": n_requests / wall,
        "input_tok_per_s": total_in / wall,
        "output_tok_per_s": total_out / wall,
        "ttft_ms": summarize_ms([r["ttft_s"] for r in results]),
        "tpot_ms": summarize_ms([r["tpot_s"] for r in results]),
        "itl_ms": summarize_ms(all_itls),
        "e2e_ms": summarize_ms([r["e2e_s"] for r in results]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8021")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--arm", required=True, help="arm label (G/E/O)")
    ap.add_argument("--concurrencies", default="1 2 4 8")
    ap.add_argument("--lengths", default="128:128 2048:256 8192:256")
    ap.add_argument("--requests", type=int, default=0, help="0 = max(8, 2*C)")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    combos = []
    for length in args.lengths.split():
        in_tok, out_tok = (int(x) for x in length.split(":"))
        for conc in (int(c) for c in args.concurrencies.split()):
            print(
                f"=== arm {args.arm}: C={conc} in={in_tok} out={out_tok} ===",
                flush=True,
            )
            try:
                combos.append(run_combo(args, conc, in_tok, out_tok, rng))
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
                print(f"combo failed: {exc}", file=sys.stderr, flush=True)
                combos.append(
                    {
                        "concurrency": conc,
                        "target_in_tokens": in_tok,
                        "target_out_tokens": out_tok,
                        "error": str(exc),
                    }
                )

    out = {
        "arm": args.arm,
        "base_url": args.base_url,
        "model": args.model,
        "seed": args.seed,
        "combos": combos,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
