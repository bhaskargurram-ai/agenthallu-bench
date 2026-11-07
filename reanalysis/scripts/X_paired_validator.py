import csv,sys,math,random,collections
from scipy.stats import binomtest
R=list(csv.DictReader(open(sys.argv[1])))
ok=lambda r:r["verifiable"].lower() in("true","1") and r["outcome"]!="infra_error"
B={(r["task_id"],r["model"]):r for r in R if r["source"]=="baseline" and r["condition"]=="baseline" and ok(r)}
P={}
for r in R:
    if r["condition"] in("p2_semantic_wrong","semantic_wrong") and ok(r): P.setdefault((r["task_id"],r["model"]),r)
keys=sorted(set(B)&set(P))
def stat(ks,lab):
    w=lambda r:str(r["CORRECT"]).lower() not in("true","1")
    b=sum(1 for k in ks if w(P[k]) and not w(B[k])); c=sum(1 for k in ks if not w(P[k]) and w(B[k]))
    n=len(ks); d=(b-c)/n
    tasks=sorted(set(k[0] for k in ks)); by=collections.defaultdict(list)
    for k in ks: by[k[0]].append(k)
    random.seed(1); bs=[]
    for _ in range(2000):
        s=[k for t in random.choices(tasks,k=len(tasks)) for k in by[t]]
        bb=sum(1 for k in s if w(P[k]) and not w(B[k])); cc=sum(1 for k in s if not w(P[k]) and w(B[k])); bs.append((bb-cc)/len(s))
    bs.sort(); p=binomtest(b,b+c,0.5).pvalue if b+c else 1
    print(f"{lab:28s} n={n:5d} err_base={sum(w(B[k]) for k in ks)/n*100:5.1f} err_inj={sum(w(P[k]) for k in ks)/n*100:5.1f} RD={d*100:+5.1f}pp CI[{bs[49]*100:+.1f},{bs[1949]*100:+.1f}] disc {b}/{c} p={p:.2g}")
stat(keys,"ALL")
ex=[k for k in keys if int(P[k]["n_exec"] or 0)>0 and int(P[k]["n_schema_err"] or 0)==0]
stat(ex,"injected call executed clean")
nt=[k for k in keys if int(P[k]["n_exec"] or 0)==0 and int(B[k]["n_exec"] or 0)==0]
stat(nt,"placebo: no tool either arm")
for m in sorted(set(k[1] for k in keys)):
    km=[k for k in keys if k[1]==m]
    if len(km)>=30: stat(km,"  "+m)
