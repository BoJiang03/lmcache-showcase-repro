"""Token-weighted LRU hit ratio vs L2 size, replaying an aiperf profile_export.

Each conversation is one LRU object whose size is its latest context. A request
whose conversation was seen before reuses its previous context: a hit when the
object is still resident. Also reports reuse gaps (time since previous turn).
"""
import json, sys, collections
R = sys.argv[1]; kb_per_tok = float(sys.argv[2]) if len(sys.argv) > 2 else 29.0
ev = []
for l in open(f"{R}/aiperf/profile_export.jsonl"):
    d = json.loads(l); m = d["metadata"]; me = d.get("metrics", {})
    isl = (me.get("input_sequence_length") or me.get("error_isl") or {}).get("value")
    osl = (me.get("output_sequence_length") or {}).get("value", 0) or 0
    if isl is None: continue
    ev.append((m["request_start_ns"]/1e9, "req", m["conversation_id"], isl, isl+osl))
    ev.append((m["request_end_ns"]/1e9, "end", m["conversation_id"], isl, isl+osl))
ev.sort()
def sim(cap_tok):
    lru = collections.OrderedDict(); used = 0; hit = tot = 0
    for t, k, c, isl, ctx in ev:
        if k == "req":
            if c in lru:
                prev = lru[c]; tot += prev; lru.move_to_end(c); hit += prev
            elif any(True for _ in ()): pass
        else:
            used -= lru.pop(c, 0); lru[c] = ctx; used += ctx
            while used > cap_tok and lru:
                _, s = lru.popitem(last=False); used -= s
    return hit, tot
# reuse gaps
last = {}; gaps = []
for t, k, c, isl, ctx in ev:
    if k == "req" and c in last: gaps.append(t - last[c])
    if k == "end": last[c] = t
gaps.sort(); q = lambda p: gaps[int(p*(len(gaps)-1))]
print(f"reuse gaps s: p50={q(.5):.0f} p90={q(.9):.0f} p99={q(.99):.0f} max={gaps[-1]:.0f}  (n={len(gaps)})")
_, full = sim(10**12)
print(f"{'L2 GB':>7} {'hit/reusable':>13}")
for gb in (50, 75, 100, 125, 150, 175, 200, 250, 300, 400):
    h, tot = sim(gb*1e9/ (kb_per_tok*1e3))
    print(f"{gb:7d} {h/full:13.3f}")
