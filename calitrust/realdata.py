"""
Real-data evaluation of CaliTrust on signed trust networks with fraud ground truth
(Bitcoin-OTC / Bitcoin-Alpha with the REV2 labels of Kumar et al., WSDM 2018).

Inputs (download them yourself; they are not redistributed here):
  network : SNAP soc-sign-bitcoin{otc,alpha}.csv(.gz)  -> source,target,rating,time
            or REV2 data/{otc,alpha}/{name}_network.csv (same columns, rating in [-1,1])
  gt      : REV2 data/{otc,alpha}/{name}_gt.csv        -> user_id,label  (label -1 = fraudulent)
            IDs must match the network file (SNAP and REV2 files share the original IDs).

Leakage control (default ON): REV2 labels are *defined* by strong ratings from benign users.
Every rating with |w| >= 0.5 from a benign-labelled user to any labelled user is removed
before features are computed. The rule depends only on the labels, never on the
train/calibration/test split, so exchangeability (and the conformal guarantee) is preserved.

Usage:
  python -m calitrust.realdata --network soc-sign-bitcoinotc.csv.gz --gt otc_gt.csv --name otc
Writes paper/data/realdata_<name>.json, which paper/render.py turns into a results table.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from .features import node_features, relation_graph
from . import model, conformal as cp, baselines as bl
from .sim import N_REL

OUT = os.path.abspath(os.environ.get("CALITRUST_OUT", "results"))
os.makedirs(OUT, exist_ok=True)


def load_network(path):
    d = pd.read_csv(path, header=None)
    d = d.iloc[:, :4] if d.shape[1] >= 4 else d
    d.columns = ["s", "t", "w", "time"][:d.shape[1]]
    if d.w.abs().max() > 1.0:          # SNAP scale -10..10 -> [-1,1]
        d["w"] = d.w / 10.0
    if "time" not in d:
        d["time"] = np.arange(len(d))
    return d


def load_gt(path):
    g = pd.read_csv(path, header=None, names=["u", "lab"])
    return {int(u): int(l == -1) for u, l in zip(g.u, g.lab)}   # 1 = fraudulent


class SignedNet:
    """Adapter exposing a signed trust network through the simulator's interface."""

    def __init__(self, d, labels, windows=20, drop_label_edges=True):
        ids = np.unique(np.concatenate([d.s.values, d.t.values, np.fromiter(labels.keys(), int)]))  # keep isolated labelled users
        self.idmap = {u: i for i, u in enumerate(ids)}
        N = len(ids)
        s = d.s.map(self.idmap).values; t = d.t.map(self.idmap).values; w = d.w.values
        y = -np.ones(N, int)
        for u, l in labels.items():
            if u in self.idmap:
                y[self.idmap[u]] = l
        self.dropped = 0
        if drop_label_edges:
            benign = (y == 0)
            keep = ~(benign[s] & (y[t] >= 0) & (np.abs(w) >= 0.5))
            self.dropped = int((~keep).sum())
            s, t, w, tm = s[keep], t[keep], w[keep], d.time.values[keep]
        else:
            tm = d.time.values
        order = np.argsort(tm, kind="stable")
        s, t, w = s[order], t[order], w[order]
        bounds = np.linspace(0, len(s), windows + 1).astype(int)
        self.log = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            self.log.append((s[a:b], t[a:b], np.full(b - a, N_REL), (w[a:b] + 1) / 2, np.zeros(b - a, np.int8)))

        class _C: pass
        self.cfg = _C(); self.cfg.N = N; self.cfg.T = windows
        self.edges = {k: np.zeros((0, 2), np.int64) for k in range(N_REL)}
        deg = np.bincount(s, minlength=N)
        self.influential = deg >= np.quantile(deg, 0.7)
        self.y = y


# Published sizes of the official REV2 label files (fraudulent, benign), used as a sanity check.
EXPECTED = {"otc": (136, 180), "alpha": (102, 138)}


def run(network, gt, name, splits=20, alphas=(0.10, 0.05), drop=True):
    d = load_network(network); labels = load_gt(gt)
    nf, nb = sum(labels.values()), len(labels) - sum(labels.values())
    if name in EXPECTED and (nf, nb) != EXPECTED[name]:
        print(f"WARNING: {gt} has {nf} fraudulent / {nb} benign users; the official REV2 file for "
              f"'{name}' has {EXPECTED[name][0]} / {EXPECTED[name][1]}. Check that you have the official file.")
    net = SignedNet(d, labels, drop_label_edges=drop)
    y_all = net.y; lab = np.flatnonzero(y_all >= 0); y = np.where(y_all < 0, 0, y_all)
    X, _ = node_features(net, net.cfg.T); g = relation_graph(net, net.cfg.T)
    info = {"name": name, "nodes": int(net.cfg.N), "edges_kept": int(sum(len(l[0]) for l in net.log)),
            "edges_dropped_label_defining": net.dropped, "labelled": int(len(lab)),
            "fraudulent": int((y_all == 1).sum()), "benign": int((y_all == 0).sum()), "splits": splits}
    print(info, flush=True)
    point, sets = {}, {}
    t0 = time.time()
    for r in range(splits):
        rng = np.random.default_rng(100 + r)
        p = rng.permutation(lab); n = len(p)
        tr, va, cal, te = p[:int(.40 * n)], p[int(.40 * n):int(.50 * n)], p[int(.50 * n):int(.75 * n)], p[int(.75 * n):]
        probs = {}
        for kind, nm in (("mlp", "MLP"), ("gcn", "GCN (Guardian-style)"), ("ragnn", "RA-GNN (ours)")):
            prm, fwd, _ = model.train(X, y, tr, va, g, kind=kind, seed=r)
            probs[nm] = model.predict(prm, fwd, X)[0]
        for nm, m in (("BRS", bl.brs(net, net.cfg.T)), ("SL", bl.subjective_logic(net, net.cfg.T)),
                      ("EigenTrust", bl.eigentrust(net, net.cfg.T, tr[y[tr] == 0]))):
            probs[nm] = np.stack([1 - m, m], 1)
        for nm, pr in probs.items():
            ts = np.unique(np.quantile(pr[cal, 1], np.linspace(0.01, 0.99, 99)))
            thr = ts[int(np.argmax([f1_score(y[cal], pr[cal, 1] >= t_) for t_ in ts]))]
            pred = pr[:, 1] >= thr
            point.setdefault(nm, []).append([roc_auc_score(y[te], pr[te, 1]), f1_score(y[te], pred[te])])
        S = cp.scores(probs["RA-GNN (ours)"])
        cand = {"Split CP, marginal": cp.sets_from_q(S, cp.marginal_q(S, y, cal, 0.10)),
                "SL + CaliTrust layer": cp.smoothed_sets(cp.scores(probs["SL"]), y, cal, alphas, rng),
                "GCN + CaliTrust layer": cp.smoothed_sets(cp.scores(probs["GCN (Guardian-style)"]), y, cal, alphas, rng),
                "CaliTrust": cp.smoothed_sets(S, y, cal, alphas, rng),
                "CaliTrust-PAC": cp.pac_sets(S, y, cal, alphas, 0.1)}
        for nm, C in cand.items():
            M = cp.metrics(C, y, te)
            sets.setdefault(nm, []).append([M["cov_M"], M["miss_M"], M["isolate_B"], M["verify"]])
        print(f"split {r} done ({time.time()-t0:.0f}s)", flush=True)
    res = {"info": info,
           "point": {k: [np.nanmean(v, 0).tolist(), np.nanstd(v, 0).tolist()] for k, v in point.items()},
           "sets": {k: [np.nanmean(v, 0).tolist(), np.nanstd(v, 0).tolist()] for k, v in sets.items()}}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/realdata_{name}.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--name", required=True); ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--keep-label-edges", action="store_true")
    a = ap.parse_args()
    run(a.network, a.gt, a.name, a.splits, drop=not a.keep_label_edges)
