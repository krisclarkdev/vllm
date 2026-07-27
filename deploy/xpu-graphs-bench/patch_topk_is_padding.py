#!/usr/bin/env python3
"""Replica of the production runtime patch for the pre-graphs image (arm O).

The 577e1a932 tree special-cases XPU to call ``_moe_C.topk_softmax`` /
``topk_softplus_sqrt`` WITHOUT ``is_padding``, but the fork kernels baked
into the image (@aa156578) require the argument, so Ornith cannot boot
without this patch (production worked around it with a ConfigMap patch;
the in-tree fix ships with the graphs feature). This script deletes the
stale ``if current_platform.is_xpu(): ... return`` blocks so the full-arg
call below them runs.

Usage: patch_topk_is_padding.py <path/to/_custom_ops.py> [...]
"""

import re
import sys

BLOCK_RE = re.compile(
    r"[ ]{4}if current_platform\.is_xpu\(\):\n"
    r"[ ]{8}# TODO: Remove after vllm-xpu-kernels supports is_padding\.\n"
    r"(?:[ ]{8}.*\n)+?"
    r"[ ]{8}return\n\n?",
)


def patch(path):
    with open(path) as fh:
        src = fh.read()
    patched, n = BLOCK_RE.subn("", src)
    if n == 0:
        print(f"{path}: no is_padding special-case found (already patched?)")
        return
    with open(path, "w") as fh:
        fh.write(patched)
    print(f"{path}: removed {n} stale XPU is_padding branch(es)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for target in sys.argv[1:]:
        patch(target)
