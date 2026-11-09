"""
Semi-synthetic validation on REAL trust networks (attack-injection protocol, as used on the
same Bitcoin networks by TrustGuard, IEEE TDSC 2024): the real topology and the real ratings
of honest users are kept; a random 10% of users with >=3 ratings become attackers whose
received and issued ratings are modified according to the five attack families. Ground truth
is exact by construction.

Caveat reported in the paper: the real networks contain some genuine fraudsters, who are
labelled honest here (label noise that works against every method equally).
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from .realdata import load_network, SignedNet
from .features import node_features, relation_graph
from . import model, conformal as cp, baselines as bl

OUT = os.path.abspath(os.environ.get("CALITRUST_OUT", "results"))
os.makedirs(OUT, exist_ok=True)
POOL = ("BS", "BM", "BST", "OO", "SB")


def inject(d, rho=0.10, seed=0, pool=POOL, ring=5):
    rng = np.random.default_rng(seed)
    ids = np.unique(np.concatenate([d.s.values, d.t.values])); m = {u: i for i, u in enumerate(ids)}
    s = d.s.map(m).values; t = d.t.map(m).values; w = d.w.values.astype(float).copy(); tm = d.time.values.astype(float)
    N = len(ids)
    deg = np.bincount(s, minlength=N) + np.bincount(t, minlength=N)
    elig = np.flatnonzero(deg >= 3)
    att = rng.choice(elig, int(round(rho * N)), replace=False)
    y = np.zeros(N, int); y[att] = 1
    typ = dict(zip(att, rng.choice(pool, len(att))))
    outdeg = np.bincount(s, minlength=N)
    infl = outdeg >= np.quantile(outdeg[outdeg > 0], 0.7)
    honest = np.flatnonzero((y == 0) & (deg >= 3))
    t0, t1 = tm.min(), tm.max()
    ns, nt, nw, ntm = [], [], [], []
    rings = {}
    bst = [a for a in att if typ[a] == "BST"]
    for i, a in enumerate(rng.permutation(bst)):
        rings.setdefault(i // ring, []).append(a)
    ring_of = {a: g for g, mem in rings.items() for a in mem}
    for a in att:
        k = typ[a]
        rec = np.flatnonzero((t == a) & (y[s] == 0))          # ratings received from honest users
        if k == "BS":
            f = rec[rng.random(len(rec)) < 0.6]; w[f] = -rng.uniform(0.5, 1.0, len(f))
        elif k == "OO":
            if len(rec):
                lo, hi = tm[rec].min(), tm[rec].max() + 1e-9
                phase = ((tm[rec] - lo) / ((hi - lo) / 4 + 1e-9)).astype(int) % 2 == 1
                f = rec[phase & (rng.random(len(rec)) < 0.8)]; w[f] = -rng.uniform(0.5, 1.0, len(f))
        elif k == "SB":
            f = rec[(~infl[s[rec]]) & (rng.random(len(rec)) < 0.7)]; w[f] = -rng.uniform(0.5, 1.0, len(f))
        elif k == "BM":
            f = rec[rng.random(len(rec)) < 0.2]; w[f] = -rng.uniform(0.5, 1.0, len(f))
            n = max(3, int(outdeg[a]))
            ns += [a] * n; nt += list(rng.choice(honest, n)); nw += list(-rng.uniform(0.7, 1.0, n)); ntm += list(rng.uniform(t0, t1, n))
        elif k == "BST":
            f = rec[rng.random(len(rec)) < 0.5]; w[f] = -rng.uniform(0.5, 1.0, len(f))
            for b in rings.get(ring_of[a], []):
                if b != a:
                    ns += [a] * 4; nt += [b] * 4; nw += list(rng.uniform(0.8, 1.0, 4)); ntm += list(rng.uniform(t0, t1, 4))
    out = pd.DataFrame({"s": np.concatenate([s, ns]).astype(int), "t": np.concatenate([t, nt]).astype(int),
                        "w": np.concatenate([w, nw]), "time": np.concatenate([tm, ntm])})
    stats = {"eligible": elig, "attackers": int(len(att)), "types": {k: int(sum(1 for a in att if typ[a] == k)) for k in pool},
             "injected_edges": int(len(ns)), "modified_ratings": int((w != d.w.values).sum()), "typ_map": typ}
    return out, {i: int(y[i]) for i in range(N)}, stats


def run(network, name, seeds=5, alphas=(0.10, 0.05)):
    d0 = load_network(network)
    point, sets, meta = {}, {}, []
    t0 = time.time()
    for sd in range(seeds):
        d, labels, st = inject(d0, seed=sd)
        elig = st.pop("eligible")               # users with >=3 real ratings (attacker-eligible)
        net = SignedNet(d, labels, drop_label_edges=False)
        y = net.y; N = net.cfg.N
        X, _ = node_features(net, net.cfg.T); g = relation_graph(net, net.cfg.T)
        # evaluate only on attacker-eligible users so that attacker selection cannot be
        # recovered from degree; ineligible users still take part in message passing
        p = np.random.default_rng(1000 + sd).permutation(elig); n = len(p)
        tr, va, cal, te = p[:int(.2 * n)], p[int(.2 * n):int(.25 * n)], p[int(.25 * n):int(.5 * n)], p[int(.5 * n):]
        probs = {}
        for kind, nm in (("mlp", "MLP"), ("gcn", "GCN (Guardian-style)"), ("ragnn", "RA-GNN (ours)")):
            prm, fwd, _ = model.train(X, y, tr, va, g, kind=kind, seed=sd)
            probs[nm] = model.predict(prm, fwd, X)[0]
        for nm, m in (("BRS", bl.brs(net, net.cfg.T)), ("SL", bl.subjective_logic(net, net.cfg.T)),
                      ("EigenTrust", bl.eigentrust(net, net.cfg.T, tr[y[tr] == 0]))):
            probs[nm] = np.stack([1 - m, m], 1)
        c_ = np.concatenate([l[0] for l in net.log]); p_ = np.concatenate([l[1] for l in net.log])
        dg = (np.bincount(c_, minlength=N) + np.bincount(p_, minlength=N)).astype(float)
        dg = dg / dg.max(); probs["Degree only"] = np.stack([1 - dg, dg], 1)
        for nm, pr in probs.items():
            ts = np.unique(np.quantile(pr[cal, 1], np.linspace(0.01, 0.99, 99)))
            thr = ts[int(np.argmax([f1_score(y[cal], pr[cal, 1] >= t_) for t_ in ts]))]
            pred = pr[:, 1] >= thr
            point.setdefault(nm, []).append([roc_auc_score(y[te], pr[te, 1]), f1_score(y[te], pred[te])])
        rng = np.random.default_rng(7 + sd)
        S = cp.scores(probs["RA-GNN (ours)"])
        pm = np.sort(probs["RA-GNN (ours)"][cal][y[cal] == 1, 1]); thr = pm[int(np.floor(0.05 * len(pm)))]
        E = np.zeros((N, 2), bool); E[:, 1] = probs["RA-GNN (ours)"][:, 1] >= thr; E[:, 0] = ~E[:, 1]
        cand = {"Empirical threshold": E,
                "Split CP, marginal": cp.sets_from_q(S, cp.marginal_q(S, y, cal, 0.10)),
                "SL + CaliTrust layer": cp.smoothed_sets(cp.scores(probs["SL"]), y, cal, alphas, rng),
                "GCN + CaliTrust layer": cp.smoothed_sets(cp.scores(probs["GCN (Guardian-style)"]), y, cal, alphas, rng),
                "CaliTrust": cp.smoothed_sets(S, y, cal, alphas, rng),
                "CaliTrust-PAC": cp.pac_sets(S, y, cal, alphas, 0.1)}
        for nm, C in cand.items():
            M = cp.metrics(C, y, te)
            sets.setdefault(nm, []).append([M["cov_M"], M["miss_M"], M["isolate_B"], M["verify"]])
        meta.append({**st, "nodes": int(N), "ratings": int(len(d)), "evaluated": int(len(elig))})
        print(f"{name} seed {sd} done ({time.time()-t0:.0f}s)", st, flush=True)
    res = {"info": {"name": name, "seeds": seeds, "nodes": meta[0]["nodes"],
                    "ratings_mean": float(np.mean([q["ratings"] for q in meta])),
                    "attackers": meta[0]["attackers"], "evaluated_users": meta[0]["evaluated"], "injected_edges_mean": float(np.mean([q["injected_edges"] for q in meta])),
                    "modified_ratings_mean": float(np.mean([q["modified_ratings"] for q in meta]))},
           "point": {k: [np.mean(v, 0).tolist(), np.std(v, 0).tolist()] for k, v in point.items()},
           "sets": {k: [np.mean(v, 0).tolist(), np.std(v, 0).tolist()] for k, v in sets.items()}}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/semisyn_{name}.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    run(a.network, a.name, a.seeds)
