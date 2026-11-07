import csv, collections, re, sys
csv.field_size_limit(10**9)
R='/Users/challagullayoshitha/Downloads/agentprop_repo/src/data/results/'
def load(f): return list(csv.DictReader(open(R+f, newline='')))
for f in ['interceptor_control_600.csv','interceptor_heldout_600.csv','interceptor_results.csv','ablation_results.csv']:
    rows=load(f); print('==',f,len(rows), list(rows[0].keys()))
    c=collections.Counter((r.get('model'),r.get('condition')) for r in rows); print(c)
    if 'domain' in rows[0]: print(collections.Counter((r['model'],r['domain']) for r in rows))
    if 'abstain_reason' in rows[0]:
        rc=collections.Counter((r['model'],re.sub(r'[:=].*','',r['abstain_reason'] or '')[:40]) for r in rows); print(rc)
        print(collections.Counter((r['model'],r['flag_count']) for r in rows))
    print(collections.Counter(r['task_id'].rsplit('_',1)[0] for r in rows))
h=load('interceptor_heldout_600.csv')
for r in h[:0]: pass
reasons=collections.Counter(r['abstain_reason'][:70] for r in h if r['abstained']=='True')
for k,v in reasons.most_common(25): print(v,k)
