# T9-STATUS — append-only coordination log for the 06:00Z swap
# Every agent appends timestamped lines. Format: [HH:MMZ] AGENT: message
[21:10Z] ORCH: T-0 = 2026-07-12 06:00Z. Sebastian authorized "make it happen" + API spend 21:00Z. BUILDER + ALIGN launching now. EVAL: finish remaining contenders WITHOUT FiQA, post here when GPU0 free. Swap gates: readback/smokes/resume/baselines/watchdogs/no-auto-updates — ANY red = no swap.
[21:32Z] CARDSPEC: WARNING — rig tailscale node key EXPIRED (~21:32Z); ssh/rsync PVE->rig now 'Network is unreachable'. EVAL bench was reachable ~30min ago, so expiry is fresh. Blocks ALIGN rsync + readback gate + swap-path access from PVE. Fix needs rig-side 'sudo tailscale up' (or key renewal in admin console) — flagging for Sebastian/ALIGN.
