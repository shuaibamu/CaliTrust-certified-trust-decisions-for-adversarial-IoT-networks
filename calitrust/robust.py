"""
Robustness of certified distrust to POST-CALIBRATION evidence manipulation on real networks.

Protocol: the encoder is trained and the conformal layer calibrated on the clean (injected-
attack) network; then malicious TEST users manipulate their evidence and the frozen encoder
is re-applied. Calibration scores are those of the clean network (the manipulation happens
after certification was set up).

Manipulations
  whitewash(keep): each malicious test user keeps only a fraction `keep` of its ratings
  sybil(k, fresh):  k new accounts each give the target a +1 rating (fresh: 1 rating each)
  sybil(k, admitted): as above, but every sybil also rates `m` random honest users so that
                    it passes an admission threshold of m+1 issued ratings (attack cost k*(m+1))

Defense: evidence admission control -- only ratings from raters with >= m_adm issued ratings
enter the features. Proposition: if every injected rater issues fewer than m_adm ratings, the
admitted evidence of every user is unchanged, so the certified guarantee holds exactly.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from .realdata import load_network, SignedNet
from .semisyn import inject
from .features import node_features, relation_graph
from . import model, conformal as cp

OUT = os.path.abspath(os.environ.get("CALITRUST_OUT", "results"))
os.makedirs(OUT, exist_ok=True)
DATA = os.environ.get("CALITRUST_DATA", "data")
AL = (0.10, 0.05)
KS = (0, 1, 3, 5, 10, 20)
M_ADM = 3


def admit(d, m):
    if m <= 0:
        return d
    cnt = d.s.value_counts()
    return d[d.s.map(cnt).fillna(0).values >= m]


def build(d, labels, m):
    net = SignedNet(admit(d, m), labels, drop_label_edges=False)
    X, _ = node_features(net, net.cfg.T); g = relation_graph(net, net.cfg.T)
    return net, X, g


def scores_frozen(prm, net, X, g):
    fwd = model.make_forward(g[0], g[1], g[2], net.cfg.N, g[3], "ragnn")
    return cp.scores(model.predict(prm, fwd, X)[0])


def add_sybils(d, targets, k, admitted, honest, rng, m=M_ADM):
    if k == 0:
        return d, []
    base = int(max(d.s.max(), d.t.max())) + 1
    tmax = float(d.time.max())
    rows, syb = [], []
    nid = base
    for a in targets:
        for _ in range(k):
            rows.append((nid, int(a), 1.0, tmax)); syb.append(nid)
            if admitted:
                for h in rng.choice(honest, m):
                    rows.append((nid, int(h), 0.8, float(rng.uniform(d.time.min(), tmax))))
            nid += 1
    add = pd.DataFrame(rows, columns=["s", "t", "w", "time"])
    return pd.concat([d, add], ignore_index=True), syb


def run(path, name, seeds=range(3)):
    t0 = time.time()
    d0 = load_network(path)
    res = {"sybil": {}, "whitewash": {}}
    for sd in seeds:
        d, lab, st = inject(d0, seed=sd)
        elig = st.pop("eligible")
        N0 = len(lab)
        y0 = np.array([lab[i] for i in range(N0)])
        p = np.random.default_rng(1000 + sd).permutation(elig); n = len(p)
        tr, va, cal, te = p[:int(.2 * n)], p[int(.2 * n):int(.25 * n)], p[int(.25 * n):int(.5 * n)], p[int(.5 * n):]
        tm = te[y0[te] == 1]
        honest = elig[y0[elig] == 0]
        for m in (0, M_ADM):
            net, X, g = build(d, lab, m)
            prm, fwd, _ = model.train(X, net.y, tr, va, g, seed=sd)
            S0 = scores_frozen(prm, net, X, g)
            refS, refy = S0[cal], net.y[cal]
            key = f"m{m}"
            for admitted in (False, True):
                for k in KS:
                    rng = np.random.default_rng(11 + sd + k)
                    da, syb = add_sybils(d, tm, k, admitted, honest, rng)
                    lab_a = dict(lab); lab_a.update({s_: 0 for s_ in syb})
                    neta, Xa, ga = build(da, lab_a, m)
                    Sa = scores_frozen(prm, neta, Xa, ga)[:N0]
                    C = cp.smoothed_sets_from_ref(Sa, refS, refy, np.array(AL), np.random.default_rng(7 + sd))
                    M = cp.metrics(C, y0, te)
                    res["sybil"].setdefault(f"{key}_{'adm' if admitted else 'fresh'}", {}).setdefault(str(k), []).append(
                        [M["miss_M"], M["verify"], M["isolate_B"]])
            if m == 0:
                for keep in (1.0, 0.5, 0.3, 0.1):
                    rng = np.random.default_rng(3 + sd)
                    mask = np.isin(d.s.values, tm) | np.isin(d.t.values, tm)
                    dw = d[~(mask & (rng.random(len(d)) > keep))]
                    netw, Xw, gw = build(dw, lab, 0)
                    Sw = scores_frozen(prm, netw, Xw, gw)[:N0]
                    C = cp.smoothed_sets_from_ref(Sw, refS, refy, np.array(AL), np.random.default_rng(7 + sd))
                    M = cp.metrics(C, y0, te)
                    res["whitewash"].setdefault(str(keep), []).append([M["miss_M"], M["verify"], M["isolate_B"]])
            print(name, "seed", sd, "m", m, f"({time.time()-t0:.0f}s)", flush=True)
    summ = {grp: {c: {k: [np.mean(v, 0).tolist(), np.std(v, 0).tolist()] for k, v in curves.items()}
                  for c, curves in res[grp].items()} if grp == "sybil" else
            {k: [np.mean(v, 0).tolist(), np.std(v, 0).tolist()] for k, v in res[grp].items()} for grp in res}
    summ["_meta"] = {"seeds": len(list(seeds)), "m_adm": M_ADM, "ks": list(KS)}
    os.makedirs(OUT, exist_ok=True)
    json.dump(summ, open(f"{OUT}/robust_{name}.json", "w"), indent=1)
    return summ


if __name__ == "__main__":
    nm = sys.argv[1]
    path = {"otc": os.path.join(DATA, "soc-sign-bitcoinotc.csv.gz"), "alpha": os.path.join(DATA, "soc-sign-bitcoinalpha.csv.gz")}[nm]
    run(path, nm)
