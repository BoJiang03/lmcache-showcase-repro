"""Live vs cumulative KV working set from an aiperf profile_export (per conversation)."""
import json, sys, collections
R = sys.argv[1]; du = sys.argv[2] if len(sys.argv) > 2 else None
recs = []
for l in open(f"{R}/aiperf/profile_export.jsonl"):
    d = json.loads(l); m = d["metadata"]; me = d.get("metrics", {})
    isl = (me.get("input_sequence_length") or me.get("error_isl") or {}).get("value")
    osl = (me.get("output_sequence_length") or {}).get("value", 0) or 0
    if isl is None: continue
    recs.append((m["conversation_id"], m["request_start_ns"] / 1e9, m["request_end_ns"] / 1e9, isl + osl))
by = collections.defaultdict(list)
for c, s, e, n in recs: by[c].append((s, e, n))
t0 = min(r[1] for r in recs); t1 = max(r[2] for r in recs)
dus = [tuple(map(float, l.split())) for l in open(du)] if du else []
print(f"convs={len(by)} reqs={len(recs)} span={t1-t0:.0f}s")
print(" t(s)   live_conv  live_Mtok  cum_Mtok  live/cum   L2_GB  live_need_GB")
for k in range(0, 13):
    t = t0 + (t1 - t0) * k / 12
    live = cum = 0; nl = 0
    for c, rs in by.items():
        started = [r for r in rs if r[0] <= t]
        if not started: continue
        ctx = max(r[2] for r in started); cum += ctx
        if max(r[1] for r in rs) > t: live += ctx; nl += 1   # will send again / still running
    l2 = None
    if dus:
        near = min(dus, key=lambda x: abs(x[0] - t)); l2 = near[1] / 1e9
    print(f"{t-t0:5.0f}  {nl:9d}  {live/1e6:9.2f}  {cum/1e6:8.2f}  {live/cum if cum else 0:7.2f}  "
          + (f"{l2:6.0f}  {l2*live/cum:12.0f}" if l2 is not None and cum else ""))
