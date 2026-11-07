"""V_00: load all traces, report replay determinism stats and score a sample (development check)."""
import sys, os, json, collections, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import V_traces as L, V_core as V
t0=time.time()
recs, stats = L.load_all()
ex = stats.pop("mismatch_examples", [])
print(len(recs), round(time.time()-t0,1), json.dumps(stats, indent=0))
for e in ex[:8]: print(json.dumps(e)[:600])
print(collections.Counter((r['source'], r['mode']) for r in recs))
