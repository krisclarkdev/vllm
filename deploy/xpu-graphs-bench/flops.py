#!/usr/bin/env python3
"""Derive achieved TFLOPS and MFU from a bench_sweep result + model config.

FLOPs model (decode, per token): ``2 * N_active`` where N_active counts only
weights a token actually touches — full-attention projections, top-k routed
experts + shared expert + router for MoE layers, state-space/GDN parameters
for linear-attention layers, and the LM head. Prefill adds the quadratic
attention term ``4 * hidden * ctx/2`` per full-attention layer per token
(average causal context = seq_len / 2).

This is the standard weight-FLOPs approximation (good to ~10% for MoE/hybrid
models); pass ``--active-params-b`` to override the derived count.
"""

import argparse
import json


def per_layer_attn_params(cfg):
    h = cfg["hidden_size"]
    heads = cfg.get("num_attention_heads", 0) or 1
    kv = cfg.get("num_key_value_heads", heads)
    head_dim = cfg.get("head_dim") or h // heads
    q = h * heads * head_dim
    k = v = h * kv * head_dim
    o = heads * head_dim * h
    return q + k + v + o


def per_layer_moe_params(cfg):
    h = cfg["hidden_size"]
    topk = cfg.get("num_experts_per_tok") or cfg.get("moe_top_k") or 0
    inter = cfg.get("moe_intermediate_size") or cfg.get("intermediate_size") or 0
    n_experts = (
        cfg.get("num_local_experts")
        or cfg.get("n_routed_experts")
        or cfg.get("num_experts")
        or 0
    )
    if not (topk and inter and n_experts):
        return None
    active = topk * 3 * h * inter
    shared = cfg.get("shared_expert_intermediate_size") or cfg.get(
        "moe_shared_expert_intermediate_size"
    )
    if shared:
        active += 3 * h * shared
    active += h * n_experts  # router
    return active


def per_layer_dense_mlp_params(cfg):
    return 3 * cfg["hidden_size"] * cfg["intermediate_size"]


def per_layer_linear_attn_params(cfg):
    """GDN/Mamba-style layer; best-effort from common config fields."""
    h = cfg["hidden_size"]
    d_state = cfg.get("linear_attn_state_size") or cfg.get("mamba_d_state") or 128
    expand = cfg.get("mamba_expand") or 2
    d_inner = cfg.get("mamba_d_inner") or expand * h
    # in/out projections dominate; conv + state params are comparatively small.
    return 2 * h * d_inner + d_inner * d_state


def layer_type_counts(cfg):
    """Return (n_full_attention, n_linear_attention) layers."""
    total = cfg["num_hidden_layers"]
    types = cfg.get("layer_types") or cfg.get("layers_block_type")
    if types:
        full = sum(1 for t in types if "full" in t or t == "attention")
        if len(types) != total:  # pattern list shorter than layer count
            full = round(full / len(types) * total)
        return full, total - full
    period = cfg.get("full_attention_interval") or cfg.get("attn_layer_period")
    if period:
        full = total // period
        return full, total - full
    return total, 0


def derive_active_params(cfg):
    n_full, n_linear = layer_type_counts(cfg)
    attn = per_layer_attn_params(cfg)
    moe = per_layer_moe_params(cfg)
    mlp = moe if moe is not None else per_layer_dense_mlp_params(cfg)
    linear = per_layer_linear_attn_params(cfg)
    body = n_full * (attn + mlp) + n_linear * (linear + mlp)
    lm_head = cfg["vocab_size"] * cfg["hidden_size"]
    return body + lm_head, n_full


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="model config.json")
    ap.add_argument("--bench-json", required=True, help="bench_sweep output")
    ap.add_argument("--peak-tflops", type=float, default=0.0)
    ap.add_argument("--active-params-b", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = json.load(fh)
    cfg = cfg.get("text_config", cfg)
    if args.active_params_b:
        n_active = args.active_params_b * 1e9
        n_full, _ = layer_type_counts(cfg)
    else:
        n_active, n_full = derive_active_params(cfg)
    h = cfg["hidden_size"]

    with open(args.bench_json) as fh:
        bench = json.load(fh)
    per_combo = []
    for combo in bench["combos"]:
        if "error" in combo:
            continue
        decode_flops_tok = 2 * n_active
        avg_ctx = combo["measured_in_tokens"] / 2
        prefill_flops_tok = decode_flops_tok + 4 * h * avg_ctx * n_full
        decode_tf = combo["output_tok_per_s"] * decode_flops_tok / 1e12
        prefill_tf = combo["input_tok_per_s"] * prefill_flops_tok / 1e12
        entry = {
            "concurrency": combo["concurrency"],
            "target_in_tokens": combo["target_in_tokens"],
            "target_out_tokens": combo["target_out_tokens"],
            "decode_tflops": round(decode_tf, 3),
            "prefill_tflops": round(prefill_tf, 3),
        }
        if args.peak_tflops:
            entry["decode_mfu_pct"] = round(decode_tf / args.peak_tflops * 100, 2)
            entry["prefill_mfu_pct"] = round(prefill_tf / args.peak_tflops * 100, 2)
        per_combo.append(entry)

    out = {
        "arm": bench["arm"],
        "active_params_b": round(n_active / 1e9, 3),
        "n_full_attn_layers": n_full,
        "flops_per_decode_token_g": round(2 * n_active / 1e9, 2),
        "peak_tflops": args.peak_tflops or None,
        "per_combo": per_combo,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
