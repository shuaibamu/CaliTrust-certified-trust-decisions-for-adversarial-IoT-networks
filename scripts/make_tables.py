"""Regenerate the revision tables (markdown) from the experiment outputs in results/ (out/).
Writes tables_rev.json, whose entries are injected into the manuscript as {{T:name}}."""
import json, os, sys
import numpy as np
OUT = os.environ.get("CALITRUST_OUT", "results")
T = {}
RB = {nm: json.load(open(os.path.join(OUT, f"rev_robust_{nm}.json"))) for nm in ["otc", "alpha"]}

RR={nm:json.load(open(f"{OUT}/rev_real_{nm}.json")) for nm in ["otc","alpha"]}
SA=json.load(open(f"{OUT}/rev_sim_ab.json"))
def p1(x): return f"{100*x:.1f}"
def f3(x): return f"{x:.3f}"
T={}
# ---- Table 11 semisyn regenerated
rows=["| | Bitcoin-OTC | | Bitcoin-Alpha | |","|:--|:-:|:-:|:-:|:-:|","| Point score | AUC | F1 | AUC | F1 |"]
for k,lab_ in [("Degree only","Degree only"),("BRS","BRS"),("EigenTrust","EigenTrust"),("SL","SL"),("MLP","MLP"),("GCN","GCN (Guardian-style)"),("RA-GNN","RA-GNN")]:
    o=RR["otc"]["point"][k]; a=RR["alpha"]["point"][k]
    rows.append(f"| {lab_} | {f3(o[0][0])}±{f3(o[1][0])} | {f3(o[0][1])} | {f3(a[0][0])}±{f3(a[1][0])} | {f3(a[0][1])} |")
T["semisyn_point"]="\n".join(rows)
rows=["| Decision procedure | Cov. M | Miss M | Isol. B | Verify | Cov. M | Miss M | Isol. B | Verify |","|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|"]
for k,lab_ in [("Empirical threshold","Empirical threshold (95% recall)"),("Split CP, marginal (alpha=0.10)","Split CP, marginal ($\\alpha=0.10$)"),("Split CP, marginal (alpha=0.05)","Split CP, marginal ($\\alpha=0.05$)"),
               ("SL + CaliTrust layer","SL + CaliTrust layer"),("GCN + CaliTrust layer","GCN + CaliTrust layer"),("CaliTrust","CaliTrust"),("CaliTrust-PAC","CaliTrust-PAC ($\\delta=0.1$)")]:
    o=RR["otc"]["sets"][k][0]; a=RR["alpha"]["sets"][k][0]
    rows.append(f"| {lab_} | "+" | ".join(f3(x) for x in o[:4])+" | "+" | ".join(f3(x) for x in a[:4])+" |")
T["semisyn_sets"]="\n".join(rows)
# ---- Table 12 low-cost rules
rows=["| | Simulation | | | Bitcoin-OTC | | | Bitcoin-Alpha | | |","|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|","| Decision rule | Miss M | Isol. B | Verify | Miss M | Isol. B | Verify | Miss M | Isol. B | Verify |"]
for k,lab_ in [("Attest all","Attest all"),("Activity + BRS rule","Activity threshold + BRS (tuned)"),("Split CP, marginal (alpha=0.05)","Split CP, marginal ($\\alpha=0.05$)"),("BRS + CaliTrust layer","BRS + CaliTrust layer"),("CaliTrust","CaliTrust")]:
    vals=[]
    for src in (SA,RR["otc"],RR["alpha"]):
        v=src["sets"][k][0]; vals+= [v[1],v[2],v[3]]
    rows.append(f"| {lab_} | "+" | ".join(f3(x) for x in vals)+" |")
T["cheap"]="\n".join(rows)
eff={nm:(src["sets"]["CaliTrust"][0][1],src["sets"]["CaliTrust"][0][4]) for nm,src in [("sim",SA),("otc",RR["otc"]),("alpha",RR["alpha"])]}
print({k:(round(m,3),round(a,3),[round(m+e*a,3) for e in (0.1,0.5,1.0)]) for k,(m,a) in eff.items()})
# ---- Table 8 anchor selection
def b(src,k,j): return src["bias"][k][0][j]
rows=["| Calibration pool | Sim. miss | Sim. verify | OTC miss | OTC verify | Alpha miss | Alpha verify |","|:--|:-:|:-:|:-:|:-:|:-:|:-:|"]
spec=[("Uniform anchors (Assumption 2)","uniform_enc",0,1),("Incident pool, ranked by BRS","incident",0,1),("Incident pool, ranked by the encoder","incident_enc",0,1),
      ("  + rolling uniform audits, windows 11–20","replayenc_rolling",1,2),("  + rolling audits and ACI, windows 11–20","replayenc_aci",1,2)]
for lab_,k,jm,jv in spec:
    rows.append(f"| {lab_} | "+" | ".join(f"{f3(b(s,k,jm))} | {f3(b(s,k,jv))}" for s in (SA,RR["otc"],RR["alpha"]))+" |")
T["bias"]="\n".join(rows)

g,fr=SA["generic"],SA["full_ref"]
rows=["| Encoder input (simulation, 10 networks) | AUC | Cov. M | Miss M | Verify |","|:--|:-:|:-:|:-:|:-:|",
      f"| All 22 features | {f3(fr[0][0])}±{f3(fr[1][0])} | {f3(fr[0][1])} | {f3(fr[0][2])} | {f3(fr[0][3])}±{f3(fr[1][3])} |",
      f"| 13 attack-agnostic features | {f3(g[0][0])}±{f3(g[1][0])} | {f3(g[0][1])} | {f3(g[0][2])} | {f3(g[0][3])}±{f3(g[1][3])} |"]
T["generic"]="\n".join(rows)
rows=["| Held-out family | AUC (held out) | AUC (all families) | Cov. M of family (held out) | Cov. M of family (all families) | Verify (held out) |","|:--|:-:|:-:|:-:|:-:|:-:|"]
for f in ["BS","BM","BST","OO","SB"]:
    l=SA["lofo"][f]; r_=SA["lofo_ref"][f]
    rows.append(f"| {f} | {f3(l[0])} | {f3(r_[0])} | {f3(l[1])} | {f3(r_[1])} | {f3(l[3])} |")
T["lofo"]="\n".join(rows); print(T["generic"]); print(T["lofo"]); print("ref verify",SA["lofo_ref"]["BS"][3])
# injection variants (Table 13)
def rv(nm,key,k,j): return RB[nm][key][str(k)][0][j]
rows=["| Injection (admission $m$) | Cost per target | OTC miss (refit / frozen) | OTC isol. B (refit / frozen) | Alpha miss (refit / frozen) | Alpha isol. B (refit / frozen) |","|:--|:-:|:-:|:-:|:-:|:-:|"]
def cell(nm,key,k,j): return f"{f3(rv(nm,key+'_refit',k,j))} / {f3(rv(nm,key+'_frozen',k,j))}"
rows.append(f"| None ($m=3$) | 0 | {cell('otc','m3_fresh',0,0)} | {cell('otc','m3_fresh',0,2)} | {cell('alpha','m3_fresh',0,0)} | {cell('alpha','m3_fresh',0,2)} |")
for kin in (1,3,10):
    rows.append(f"| {kin} colluding insider(s) ($m=3$) | {kin} | {cell('otc','m3_insider',kin,0)} | {cell('otc','m3_insider',kin,2)} | {cell('alpha','m3_insider',kin,0)} | {cell('alpha','m3_insider',kin,2)} |")
for m in (3,5,10):
    ext=max(m,3)+1
    for k in (5,20):
        rows.append(f"| {k} established sybils ($m={m}$) | {k*ext} | {cell('otc',f'm{m}_est',k,0)} | {cell('otc',f'm{m}_est',k,2)} | {cell('alpha',f'm{m}_est',k,0)} | {cell('alpha',f'm{m}_est',k,2)} |")
T["inject"]="\n".join(rows); print(T["inject"])
print("no-attack verify by m:", {nm:{m:round(RB[nm][f"m{m}_est_refit" if m in (5,10) else ("m0_fresh_refit" if m==0 else "m3_fresh_refit")]["0"][0][1],3) for m in (0,3,5,10)} for nm in RB})
# check: at k=20 established, does every (m,proto) violate at least one budget?
for nm in RB:
    for m in (3,5,10):
        for pr in ("refit","frozen"):
            mi=rv(nm,f"m{m}_est_{pr}",20,0); io=rv(nm,f"m{m}_est_{pr}",20,2); print(nm,m,pr,round(mi,3),round(io,3), "VIOL" if (mi>0.06 or io>0.12) else "ok")


rows=["| Injection after calibration | Network | Static miss | Rolling miss (w 11–20) | ACI miss (w 1–4) | ACI miss (w 11–20) | Static verify | ACI verify (w 11–20) |","|:--|:--|:-:|:-:|:-:|:-:|:-:|:-:|"]
spec=[("1 fresh sybil, no admission control","m0_fresh_1"),("10 fresh sybils, no admission control","m0_fresh_10"),("10 colluding insiders, $m=3$","m3_insider_10"),("20 established sybils, $m=3$","m3_est_20")]
stat_ver={}
for lab_,cfg in spec:
    for nm,nl in (("otc","OTC"),("alpha","Alpha")):
        a=RB[nm]["_aci"][cfg]
        rows.append(f"| {lab_} | {nl} | {f3(a['static']['miss_last10'])} | {f3(a['rolling']['miss_last10'])} | {f3(a['aci']['miss_first4'])} | {f3(a['aci']['miss_last10'])} | {f3(a['static']['ver_last10'])} | {f3(a['aci']['ver_last10'])} |")
T["aci"]="\n".join(rows); print(T["aci"])



# header emphasis
T["semisyn_point"]=T["semisyn_point"].replace("| Point score | AUC | F1 | AUC | F1 |","| **Point score** | **AUC** | **F1** | **AUC** | **F1** |")
T["cheap"]=T["cheap"].replace("| Decision rule | Miss M | Isol. B | Verify | Miss M | Isol. B | Verify | Miss M | Isol. B | Verify |","| **Decision rule** | **Miss M** | **Isol. B** | **Verify** | **Miss M** | **Isol. B** | **Verify** | **Miss M** | **Isol. B** | **Verify** |")
T["semisyn_sets"]="| | Bitcoin-OTC | | | | Bitcoin-Alpha | | | |\n|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|\n"+T["semisyn_sets"].replace("| Decision procedure | Cov. M | Miss M | Isol. B | Verify | Cov. M | Miss M | Isol. B | Verify |\n|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|\n","| **Decision procedure** | **Cov. M** | **Miss M** | **Isol. B** | **Verify** | **Cov. M** | **Miss M** | **Isol. B** | **Verify** |\n")

json.dump(T, open(sys.argv[1] if len(sys.argv) > 1 else "tables_rev.json", "w", encoding="utf8"), indent=1)
