#!/usr/bin/env python3
"""clean_stale_claims.py — release claims held by DEAD workers of THIS host.

Restart hygiene for OPS §1 resume (launch_workers.sh runs it first). Scope
is strictly our own: claim tags embed <hostname>-<pid>-<role>; only claims
whose hostname matches ours AND whose pid no longer exists are removed.
Other agents' claims (any tag shape) and live workers are never touched —
breaking OTHER agents' stale claims stays an Opus-manager-only action.
"""
from __future__ import annotations

import os
import re
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

TAG = re.compile(r"__([a-zA-Z0-9.-]+)-(\d+)-(cpu|gpu\d+|smoke)$")


def main() -> None:
    host = socket.gethostname()
    removed = 0
    if not os.path.isdir(common.CLAIMS):
        return
    for name in os.listdir(common.CLAIMS):
        m = TAG.search(name)
        if not m or m.group(1) != host:
            continue
        pid = int(m.group(2))
        try:
            os.kill(pid, 0)
            continue                      # worker still alive
        except ProcessLookupError:
            pass
        except PermissionError:
            continue
        d = os.path.join(common.CLAIMS, name)
        hb = os.path.join(d, "hb")
        if os.path.exists(hb):
            os.unlink(hb)
        os.rmdir(d)
        removed += 1
        print(f"released stale claim: {name}")
    print(f"clean_stale_claims: {removed} released")


if __name__ == "__main__":
    main()
