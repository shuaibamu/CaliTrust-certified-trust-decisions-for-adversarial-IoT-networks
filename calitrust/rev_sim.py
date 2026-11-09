"""Revision experiments on the Social-IoT simulator.
  part A: consistent-target comparison, cheap comparators, per-class attestation,
          anchor-selection bias with audit replay, generic (attack-agnostic) features
  part B: leave-one-attack-family-out encoder training
  part C: sleeper drift with per-seed alpha_{M,t} trajectories (Theorem 2 diagnostics)"""
import json, os, sys, time
import numpy as np
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
from sklearn.metrics import roc_auc_score
from .sim import SIoT, Cfg
from .features import node_features, relation_graph
from . import model, conformal as cp, baselines as bl
from .experiments import build, split, empirical_threshold
from .revcommon import full_metrics, attest_all, activity_brs_rule, biased_cal, replay_online, AB, AM

OUT = os.path.abspath(os.environ.get("CALITRUST_OUT", "results"))
os.makedirs(OUT, exist_ok=True)
GENERIC = ["rec_mean_ewma", "rec_mean_last", "rec_count", "rec_rel0", "rec_rel1", "rec_rel2", "rec_rel3", "rec_rel4",
           "rec_disp", "deg_rel0", "deg_rel1", "deg_rel2", "deg_rel3"]
FAM = ("BS", "BM", "BST", "OO", "SB")


def part_ab(seeds=range(10), lofo_seeds=range(5)):
    R = {"sets": {}, "bias": {}, "generic": [], "full_ref": [], "lofo": {}, "lofo_ref": {}, "rule": []}
    t0 = time.time()
    for s in seeds:
        cfg = Cfg(seed=s)
        ctx = build(cfg, s)
        net, X, g, y, tr, va, cal, te = (ctx[k] for k in ("net", "X", "g", "y", "tr", "va", "cal", "te"))
        N = len(y)
        _, names = node_features(net, cfg.T)
        S = cp.scores(ctx["prob"])
        brs = bl.brs(net, cfg.T)
        p_ = np.concatenate([l[1] for l in net.log]); recv = np.bincount(p_, minlength=N).astype(float)
        rule, rp = activity_brs_rule(brs, recv, y, cal); R["rule"].append(rp)
        rng = np.random.default_rng(7 + s)
        cand = {"Attest all": attest_all(N), "Activity + BRS rule": rule,
                "Empirical threshold": empirical_threshold(ctx["prob"], y, cal, 1 - AM),
                "Split CP, marginal (alpha=0.10)": cp.sets_from_q(S, cp.marginal_q(S, y, cal, 0.10)),
                "Split CP, marginal (alpha=0.05)": cp.sets_from_q(S, cp.marginal_q(S, y, cal, 0.05)),
                "BRS + CaliTrust layer": cp.smoothed_sets(cp.scores(np.stack([1 - brs, brs], 1)), y, cal, (AB, AM), rng),
                "CaliTrust": cp.smoothed_sets(S, y, cal, (AB, AM), np.random.default_rng(7 + s)),
                "CaliTrust-PAC": cp.pac_sets(S, y, cal, (AB, AM), 0.1)}
        for nm, C in cand.items():
            M = full_metrics(C, y, te)
            R["sets"].setdefault(nm, []).append([M["cov_M"], M["miss_M"], M["isolate_B"], M["verify"], M["attest_M"], M["attest_B"]])
        # anchor-selection bias
        pool = np.concatenate([cal, te])
        bc = biased_cal(brs, y, cal, pool, frac=0.5, rng=np.random.default_rng(5 + s))
        teb = np.setdiff1d(te, bc)
        Mu = full_metrics(cp.smoothed_sets(S, y, cal, (AB, AM), np.random.default_rng(7 + s)), y, teb)
        Mb = full_metrics(cp.smoothed_sets(S, y, bc, (AB, AM), np.random.default_rng(7 + s)), y, teb)
        R["bias"].setdefault("uniform", []).append([Mu["miss_M"], Mu["verify"]])
        R["bias"].setdefault("incident", []).append([Mb["miss_M"], Mb["verify"]])
        # stronger variant: incident pool = attackers the encoder itself flags most strongly
        bce = biased_cal(S[:, 0], y, cal, pool, frac=0.5, rng=np.random.default_rng(5 + s))
        tee = np.setdiff1d(te, bce)
        Mue = full_metrics(cp.smoothed_sets(S, y, cal, (AB, AM), np.random.default_rng(7 + s)), y, tee)
        Mbe = full_metrics(cp.smoothed_sets(S, y, bce, (AB, AM), np.random.default_rng(7 + s)), y, tee)
        R["bias"].setdefault("uniform_enc", []).append([Mue["miss_M"], Mue["verify"]])
        R["bias"].setdefault("incident_enc", []).append([Mbe["miss_M"], Mbe["verify"]])
        for mode in ("static", "rolling", "aci"):
            rr = replay_online(S, y, bce, tee, windows=20, mode=mode, seed=s)
            R["bias"].setdefault("replayenc_" + mode, []).append([float(np.mean(rr["miss"][:4])), float(np.mean(rr["miss"][10:])),
                                                               float(np.mean(rr["ver"][10:]))])
        for mode in ("static", "rolling", "aci"):
            rr = replay_online(S, y, bc, teb, windows=20, mode=mode, seed=s)
            R["bias"].setdefault("replay_" + mode, []).append([float(np.mean(rr["miss"][:4])), float(np.mean(rr["miss"][10:])),
                                                               float(np.mean(rr["ver"][10:]))])
        # generic, attack-agnostic features
        cols = [names.index(c) for c in GENERIC]
        Xg = X[:, cols]
        pg, fg, _ = model.train(Xg, y, tr, va, g, seed=s)
        prg = model.predict(pg, fg, Xg)[0]
        Mg = full_metrics(cp.smoothed_sets(cp.scores(prg), y, cal, (AB, AM), np.random.default_rng(7 + s)), y, te)
        Mf = full_metrics(cand["CaliTrust"], y, te)
        R["generic"].append([roc_auc_score(y[te], prg[te, 1]), Mg["cov_M"], Mg["miss_M"], Mg["verify"]])
        R["full_ref"].append([roc_auc_score(y[te], ctx["prob"][te, 1]), Mf["cov_M"], Mf["miss_M"], Mf["verify"]])
        # leave-one-family-out
        if s in lofo_seeds:
            atk = net.atk
            hon = te[y[te] == 0]
            for f in FAM:
                keep = lambda idx: idx[~((y[idx] == 1) & (atk[idx] == f))]
                pf, ff, _ = model.train(X, y, keep(tr), keep(va), g, seed=s)
                prf = model.predict(pf, ff, X)[0]
                for tag, pr in (("lofo", prf), ("lofo_ref", ctx["prob"])):
                    C = cp.smoothed_sets(cp.scores(pr), y, cal, (AB, AM), np.random.default_rng(7 + s))
                    mf = te[(y[te] == 1) & (atk[te] == f)]
                    yy = np.r_[np.zeros(len(hon)), np.ones(len(mf))]
                    auc = roc_auc_score(yy, np.r_[pr[hon, 1], pr[mf, 1]])
                    covf = C[mf, 1].mean(); missf = (C[mf, 0] & ~C[mf, 1]).mean()
                    R[tag].setdefault(f, []).append([auc, covf, missf, (C[te].sum(1) != 1).mean()])
        print("A/B seed", s, f"{time.time()-t0:.0f}s", flush=True)
    summ = {"sets": {k: [np.mean(v, 0).tolist(), np.std(v, 0).tolist()] for k, v in R["sets"].items()},
            "bias": {k: [np.mean(v, 0).tolist(), np.std(v, 0).tolist()] for k, v in R["bias"].items()},
            "generic": [np.mean(R["generic"], 0).tolist(), np.std(R["generic"], 0).tolist()],
            "full_ref": [np.mean(R["full_ref"], 0).tolist(), np.std(R["full_ref"], 0).tolist()],
            "lofo": {f: np.mean(v, 0).tolist() for f, v in R["lofo"].items()},
            "lofo_ref": {f: np.mean(v, 0).tolist() for f, v in R["lofo_ref"].items()},
            "rule": R["rule"], "raw": R}
    json.dump(summ, open(f"{OUT}/rev_sim_ab.json", "w"), indent=1)


def part_c(seeds=range(6), T0=16, T=46, drift=22, audit=0.05, gamma=0.05):
    out = {m: [] for m in ("static", "rolling", "aci")}
    for s in seeds:
        cfg = Cfg(seed=s, T=T, pool=("BS", "BM", "BST"), drift_at=drift, drift_frac=0.8, drift_to=("SLEEP",))
        net = SIoT(cfg); net.run(T0)
        X, _ = node_features(net, T0); g = relation_graph(net, T0)
        y = net.y; tr, va, cal, te = split(len(y), s)
        p, fwd, _ = model.train(X, y, tr, va, g, seed=s)
        S0 = cp.scores(model.predict(p, fwd, X)[0])
        Sw = []
        for w in range(T0, T):
            net.step(w)
            Xw, _ = node_features(net, w + 1); gw = relation_graph(net, w + 1)
            fw = model.make_forward(*gw[:3], len(y), gw[3], "ragnn")
            Sw.append(cp.scores(model.predict(p, fw, Xw)[0]))
        for mode in ("static", "rolling", "aci"):
            rng = np.random.default_rng(99 + s)
            oc = cp.OnlineCaliTrust(S0[cal], y[cal], (AB, AM), mode=mode, gamma=gamma)
            miss, cmiss, ver, aM, errM = [], [], [], [], []
            for S in Sw:
                if mode == "static":
                    refS, refy = oc.init
                else:
                    refS = np.vstack([b[0] for b in oc.buf[-oc.K:]]); refy = np.concatenate([b[1] for b in oc.buf[-oc.K:]])
                a = oc.at if mode == "aci" else oc.alphas
                aM.append(float(oc.at[1]))
                C = cp.smoothed_sets_from_ref(S, refS, refy, np.clip(a, -1, 1.0), rng)
                M = cp.metrics(C, y, te)
                miss.append(M["miss_M"]); cmiss.append(1 - M["cov_M"]); ver.append(M["verify"])
                aud = rng.choice(te, int(audit * len(te)), replace=False)
                ma = aud[y[aud] == 1]
                errM.append(float(1 - C[ma, 1].mean()) if len(ma) else np.nan)
                oc.update(S[aud], y[aud], C[aud])
            out[mode].append(dict(miss=miss, cmiss=cmiss, ver=ver, alphaM=aM, errM=errM, n_aud_M=None))
        print("C seed", s, flush=True)
    json.dump(out, open(f"{OUT}/rev_sim_c.json", "w"), indent=1)


if __name__ == "__main__":
    import jax
    {"ab": part_ab, "c": part_c}[sys.argv[1]]()
