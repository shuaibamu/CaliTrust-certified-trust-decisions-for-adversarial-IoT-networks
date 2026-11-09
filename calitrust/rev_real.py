"""Revision experiment on real networks (semi-synthetic protocol of semisyn.py):
regenerates Table 'semisyn' and adds consistent-target marginal CP, cheap comparators,
per-family BRS diagnosis, per-class attestation rates and anchor-selection bias."""
import json, os, sys, time
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
from .realdata import load_network, SignedNet
from .semisyn import inject
from .features import node_features, relation_graph
from . import model, conformal as cp, baselines as bl
from .revcommon import full_metrics, attest_all, activity_brs_rule, biased_cal, replay_online, decompose, AB, AM

OUT = os.path.abspath(os.environ.get("CALITRUST_OUT", "results"))
os.makedirs(OUT, exist_ok=True)
DATA = os.environ.get("CALITRUST_DATA", "data")


def run(path, name, seeds=5):
    d0 = load_network(path)
    R = {"point": {}, "sets": {}, "brs_family": {}, "rule": [], "bias": {}, "decomp": [], "info": []}
    t0 = time.time()
    for sd in range(seeds):
        d, labels, st = inject(d0, seed=sd)
        elig = st.pop("eligible"); typ = st.pop("typ_map")
        net = SignedNet(d, labels, drop_label_edges=False)
        y = net.y; N = net.cfg.N
        X, _ = node_features(net, net.cfg.T); g = relation_graph(net, net.cfg.T)
        p = np.random.default_rng(1000 + sd).permutation(elig); n = len(p)
        tr, va, cal, te = p[:int(.2 * n)], p[int(.2 * n):int(.25 * n)], p[int(.25 * n):int(.5 * n)], p[int(.5 * n):]
        probs = {}
        for kind, nm in (("mlp", "MLP"), ("gcn", "GCN"), ("ragnn", "RA-GNN")):
            prm, fwd, _ = model.train(X, y, tr, va, g, kind=kind, seed=sd)
            probs[nm] = model.predict(prm, fwd, X)[0]
        brs = bl.brs(net, net.cfg.T)
        for nm, m in (("BRS", brs), ("SL", bl.subjective_logic(net, net.cfg.T)),
                      ("EigenTrust", bl.eigentrust(net, net.cfg.T, tr[y[tr] == 0]))):
            probs[nm] = np.stack([1 - m, m], 1)
        c_ = np.concatenate([l[0] for l in net.log]); p_ = np.concatenate([l[1] for l in net.log])
        dg = (np.bincount(c_, minlength=N) + np.bincount(p_, minlength=N)).astype(float)
        dgn = dg / dg.max(); probs["Degree only"] = np.stack([1 - dgn, dgn], 1)
        recv = np.bincount(p_, minlength=N).astype(float)
        for nm, pr in probs.items():
            ts = np.unique(np.quantile(pr[cal, 1], np.linspace(0.01, 0.99, 99)))
            thr = ts[int(np.argmax([f1_score(y[cal], pr[cal, 1] >= t_) for t_ in ts]))]
            pred = pr[:, 1] >= thr
            R["point"].setdefault(nm, []).append([roc_auc_score(y[te], pr[te, 1]), f1_score(y[te], pred[te])])
        # ---- BRS diagnosis per attack family (family vs honest, test users)
        fam = np.array(["NONE"] * N, dtype=object)
        for a, k in typ.items():
            fam[a] = k
        hon = te[y[te] == 0]
        for k in ("BS", "BM", "BST", "OO", "SB"):
            mk = te[fam[te] == k]
            if len(mk):
                yy = np.r_[np.zeros(len(hon)), np.ones(len(mk))]
                R["brs_family"].setdefault(k, []).append([roc_auc_score(yy, np.r_[brs[hon], brs[mk]]),
                                                         float(np.mean(1 - brs[mk])), float(np.mean(1 - brs[hon]))])
        # ---- decision procedures
        rng = np.random.default_rng(7 + sd)
        S = cp.scores(probs["RA-GNN"])
        pm = np.sort(probs["RA-GNN"][cal][y[cal] == 1, 1]); thr = pm[int(np.floor(0.05 * len(pm)))]
        Emp = np.zeros((N, 2), bool); Emp[:, 1] = probs["RA-GNN"][:, 1] >= thr; Emp[:, 0] = ~Emp[:, 1]
        rule, rp = activity_brs_rule(brs, recv, y, cal)
        R["rule"].append(rp)
        cand = {"Attest all": attest_all(N),
                "Activity + BRS rule": rule,
                "Empirical threshold": Emp,
                "Split CP, marginal (alpha=0.10)": cp.sets_from_q(S, cp.marginal_q(S, y, cal, 0.10)),
                "Split CP, marginal (alpha=0.05)": cp.sets_from_q(S, cp.marginal_q(S, y, cal, 0.05)),
                "BRS + CaliTrust layer": cp.smoothed_sets(cp.scores(probs["BRS"]), y, cal, (AB, AM), rng),
                "SL + CaliTrust layer": cp.smoothed_sets(cp.scores(probs["SL"]), y, cal, (AB, AM), rng),
                "GCN + CaliTrust layer": cp.smoothed_sets(cp.scores(probs["GCN"]), y, cal, (AB, AM), rng),
                "CaliTrust": cp.smoothed_sets(S, y, cal, (AB, AM), rng),
                "CaliTrust-PAC": cp.pac_sets(S, y, cal, (AB, AM), 0.1)}
        for nm, C in cand.items():
            M = full_metrics(C, y, te)
            R["sets"].setdefault(nm, []).append([M["cov_M"], M["miss_M"], M["isolate_B"], M["verify"], M["attest_M"], M["attest_B"]])
        R["decomp"].append(decompose(cand["CaliTrust"], y, te))
        Cc = cand["CaliTrust"]; amb = Cc.sum(1) != 1
        for lo, hi in ((3, 5), (6, 10), (11, 20), (21, 50), (51, 10 ** 9)):
            sel = te[(dg[te] >= lo) & (dg[te] <= hi)]
            R.setdefault("deg_bins", {}).setdefault(f"{lo}-{hi}", []).append(
                [float(len(sel) / len(te)), float(amb[sel].mean()) if len(sel) else np.nan,
                 float((Cc[sel, 0] & ~Cc[sel, 1])[y[sel] == 1].mean()) if (y[sel] == 1).any() else np.nan])
        # ---- anchor-selection bias (incident-response pool ranked by incumbent BRS) + audit replay
        pool = np.concatenate([cal, te])
        bc = biased_cal(brs, y, cal, pool, frac=0.5, rng=np.random.default_rng(5 + sd))
        teb = np.setdiff1d(te, bc)
        Cu = cp.smoothed_sets(S, y, cal, (AB, AM), np.random.default_rng(7 + sd))
        Cb = cp.smoothed_sets(S, y, bc, (AB, AM), np.random.default_rng(7 + sd))
        Mu = full_metrics(Cu, y, teb); Mb = full_metrics(Cb, y, teb)
        R["bias"].setdefault("uniform", []).append([Mu["miss_M"], Mu["verify"]])
        R["bias"].setdefault("incident", []).append([Mb["miss_M"], Mb["verify"]])
        # stronger variant: incident pool = attackers the encoder itself flags most strongly
        bce = biased_cal(S[:, 0], y, cal, pool, frac=0.5, rng=np.random.default_rng(5 + sd))
        tee = np.setdiff1d(te, bce)
        Mue = full_metrics(cp.smoothed_sets(S, y, cal, (AB, AM), np.random.default_rng(7 + sd)), y, tee)
        Mbe = full_metrics(cp.smoothed_sets(S, y, bce, (AB, AM), np.random.default_rng(7 + sd)), y, tee)
        R["bias"].setdefault("uniform_enc", []).append([Mue["miss_M"], Mue["verify"]])
        R["bias"].setdefault("incident_enc", []).append([Mbe["miss_M"], Mbe["verify"]])
        for mode in ("static", "rolling", "aci"):
            rr = replay_online(S, y, bce, tee, windows=20, mode=mode, seed=sd)
            R["bias"].setdefault("replayenc_" + mode, []).append([float(np.mean(rr["miss"][:4])), float(np.mean(rr["miss"][10:])),
                                                               float(np.mean(rr["ver"][10:]))])
        for mode in ("static", "rolling", "aci"):
            rr = replay_online(S, y, bc, teb, windows=20, mode=mode, seed=sd)
            R["bias"].setdefault("replay_" + mode, []).append([float(np.mean(rr["miss"][:4])), float(np.mean(rr["miss"][10:])),
                                                               float(np.mean(rr["ver"][10:]))])
        R["info"].append({"nodes": int(N), "evaluated": int(len(elig)), "attackers": int(st["attackers"]),
                          "median_ratings_eval": float(np.median(dg[elig])), "frac_le10": float(np.mean(dg[elig] <= 10))})
        print(name, "seed", sd, f"{time.time()-t0:.0f}s", flush=True)
    summ = {"point": {k: [np.mean(v, 0).tolist(), np.std(v, 0).tolist()] for k, v in R["point"].items()},
            "sets": {k: [np.mean(v, 0).tolist(), np.std(v, 0).tolist()] for k, v in R["sets"].items()},
            "brs_family": {k: np.mean(v, 0).tolist() for k, v in R["brs_family"].items()},
            "bias": {k: [np.mean(v, 0).tolist(), np.std(v, 0).tolist()] for k, v in R["bias"].items()},
            "deg_bins": {k: np.nanmean(v, 0).tolist() for k, v in R["deg_bins"].items()},
            "rule": R["rule"], "decomp": R["decomp"], "info": R["info"],
            "raw_sets": R["sets"], "raw_point": R["point"],
            "cols_sets": ["cov_M", "miss_M", "isolate_B", "verify", "attest_M", "attest_B"]}
    os.makedirs(OUT, exist_ok=True)
    json.dump(summ, open(f"{OUT}/rev_real_{name}.json", "w"), indent=1)
    return summ


if __name__ == "__main__":
    import jax
    nm = sys.argv[1]
    path = {"otc": os.path.join(DATA, "soc-sign-bitcoinotc.csv.gz"), "alpha": os.path.join(DATA, "soc-sign-bitcoinalpha.csv.gz")}[nm]
    run(path, nm, int(sys.argv[2]) if len(sys.argv) > 2 else 5)
