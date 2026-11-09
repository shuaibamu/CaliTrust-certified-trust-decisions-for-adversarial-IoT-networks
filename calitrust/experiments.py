"""All experiments for the CaliTrust paper. Every number in the manuscript is
produced here; data files (.dat) are consumed directly by PGFPlots."""
import json, os, sys, time
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from .sim import SIoT, Cfg
from .features import node_features, relation_graph
from . import model, conformal as cp, baselines as bl

OUT = os.path.abspath(os.environ.get("CALITRUST_OUT", "results"))
os.makedirs(OUT, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
AB, AM = 0.10, 0.05               # default alpha_Benign, alpha_Malicious


def save(name, header, rows):
    with open(f"{OUT}/{name}.dat", "w") as f:
        f.write(" ".join(header) + "\n")
        for r in rows:
            f.write(" ".join(f"{v:.5g}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + "\n")


def split(N, seed, ftr=0.20, fva=0.05, fcal=0.25):
    perm = np.random.default_rng(1000 + seed).permutation(N)
    a, b, c = int(ftr * N), int((ftr + fva) * N), int((ftr + fva + fcal) * N)
    return perm[:a], perm[a:b], perm[b:c], perm[c:]


def build(cfg, seed, kind="ragnn", groups=("P", "R", "T", "S"), signed=True, fcal=0.25, upto=None):
    net = SIoT(cfg).run()
    upto = upto or cfg.T
    X, names = node_features(net, upto, groups)
    g = relation_graph(net, upto, signed)
    y = net.y
    tr, va, cal, te = split(len(y), seed, fcal=fcal)
    p, fwd, tt = model.train(X, y, tr, va, g, kind=kind, seed=seed)
    prob, H, att = model.predict(p, fwd, X)
    return dict(net=net, X=X, g=g, y=y, tr=tr, va=va, cal=cal, te=te, prob=prob, H=H, att=att,
                params=p, fwd=fwd, tt=tt)


def classical_probs(net, upto, tr, y):
    out = {"BRS": bl.brs(net, upto),
           "EigenTrust": bl.eigentrust(net, upto, tr[y[tr] == 0]),
           "SL": bl.subjective_logic(net, upto)}
    return {k: np.stack([1 - v, v], 1) for k, v in out.items()}


def cset(prob, y, cal, alphas, seed, method="calitrust", g=None, H=None):
    rng = np.random.default_rng(7 + seed)
    S = cp.scores(prob)
    if method == "calitrust":
        return cp.smoothed_sets(S, y, cal, alphas, rng)
    if method == "mondrian_det":
        return cp.sets_from_q(S, cp.mondrian_q(S, y, cal, alphas))
    if method == "marginal":
        return cp.sets_from_q(S, cp.marginal_q(S, y, cal, 0.10))
    if method == "daps":        # uniform-diffusion scores, marginal quantile (original DAPS)
        SS = cp.diffuse(S, g[0], g[1], len(y), 0.4)
        return cp.sets_from_q(SS, cp.marginal_q(SS, y, cal, 0.10))
    if method == "daps_mond":
        SS = cp.diffuse(S, g[0], g[1], len(y), 0.4)
        return cp.smoothed_sets(SS, y, cal, alphas, rng)
    if method == "egd_mond":
        SS = cp.diffuse(S, g[0], g[1], len(y), 0.4, H, 16)
        return cp.smoothed_sets(SS, y, cal, alphas, rng)
    raise ValueError(method)


def empirical_threshold(prob, y, cal, target):
    """Non-conformal baseline: pick threshold on calibration so that the
    empirical malicious recall >= target; trust iff p_M < thr (no abstention)."""
    pm = np.sort(prob[cal][y[cal] == 1, 1])
    k = int(np.floor((1 - target) * len(pm)))
    thr = pm[k] if len(pm) else 0.5
    C = np.zeros((len(y), 2), bool)
    C[:, 1] = prob[:, 1] >= thr
    C[:, 0] = ~C[:, 1]
    return C


def f1_best_thr(score, y, cal):
    ts = np.unique(np.quantile(score[cal], np.linspace(0.01, 0.99, 99)))
    f = [f1_score(y[cal], score[cal] >= t) for t in ts]
    return ts[int(np.argmax(f))]


# =========================================================================
def exp_main(seeds=range(10)):
    """Table: SOTA comparison (point + set-valued), MIX attack."""
    point, sets = {}, {}
    for s in seeds:
        cfg = Cfg(seed=s)
        ctx = build(cfg, s)
        y, cal, te, tr = ctx["y"], ctx["cal"], ctx["te"], ctx["tr"]
        probs = classical_probs(ctx["net"], cfg.T, tr, y)
        for kind in ("mlp", "gcn"):
            c2 = model.train(ctx["X"], y, tr, ctx["va"], ctx["g"], kind=kind, seed=s)
            probs[{"mlp": "MLP", "gcn": "GCN (Guardian-style)"}[kind]] = model.predict(c2[0], c2[1], ctx["X"])[0]
        probs["RA-GNN (ours)"] = ctx["prob"]
        for k, pr in probs.items():
            thr = f1_best_thr(pr[:, 1], y, cal)
            pred = pr[:, 1] >= thr
            point.setdefault(k, []).append([roc_auc_score(y[te], pr[te, 1]), f1_score(y[te], pred[te]),
                                            pred[te][y[te] == 1].mean(), (pred[te][y[te] == 0]).mean()])
        # set-valued decision procedures (all target malicious coverage 0.95)
        cands = {
            "Empirical threshold (RA-GNN)": empirical_threshold(ctx["prob"], y, cal, 1 - AM),
            "Split CP, marginal (RA-GNN)": cp.sets_from_q(cp.scores(ctx["prob"]), cp.marginal_q(cp.scores(ctx["prob"]), y, cal, 0.10)),
            "DAPS (RA-GNN)": cset(ctx["prob"], y, cal, (AB, AM), s, "daps", ctx["g"]),
            "BRS + CaliTrust layer": cset(probs["BRS"], y, cal, (AB, AM), s),
            "SL + CaliTrust layer": cset(probs["SL"], y, cal, (AB, AM), s),
            "GCN + CaliTrust layer": cset(probs["GCN (Guardian-style)"], y, cal, (AB, AM), s),
            "CaliTrust (full)": cset(ctx["prob"], y, cal, (AB, AM), s),
            "CaliTrust-PAC (delta=0.1)": cp.pac_sets(cp.scores(ctx["prob"]), y, cal, (AB, AM), 0.1),
        }
        for k, C in cands.items():
            M = cp.metrics(C, y, te)
            sets.setdefault(k, []).append([M["cov_M"], M["cov_B"], M["verify"], M["miss_M"], M["isolate_B"], M["single_acc"]])
        print("main seed", s, flush=True)
    res = {"point": {k: [np.mean(v, 0).tolist(), np.std(v, 0).tolist()] for k, v in point.items()},
           "sets": {k: [np.nanmean(v, 0).tolist(), np.nanstd(v, 0).tolist()] for k, v in sets.items()},
           "raw_point_auc": {k: [r[0] for r in v] for k, v in point.items()},
           "raw_point_f1": {k: [r[1] for r in v] for k, v in point.items()},
           "raw_sets_verify": {k: [r[2] for r in v] for k, v in sets.items()},
           "raw_sets_covM": {k: [r[0] for r in v] for k, v in sets.items()}}
    json.dump(res, open(f"{OUT}/main.json", "w"), indent=1)
    return res


def exp_validity_efficiency(seeds=range(10)):
    """R1: malicious-class coverage vs target; R2: verification rate vs alpha_M
    for several encoders; R6 ROC; R7 score histograms (seed 0)."""
    alphas = [0.02, 0.05, 0.10, 0.15, 0.20]
    cov = {m: {a: [] for a in alphas} for m in ("marginal", "daps", "calitrust", "empirical")}
    ver = {e: {a: [] for a in alphas} for e in ("RA-GNN", "GCN", "MLP", "SL", "BRS")}
    roc_store = {}
    for s in seeds:
        cfg = Cfg(seed=s)
        ctx = build(cfg, s)
        y, cal, te, tr = ctx["y"], ctx["cal"], ctx["te"], ctx["tr"]
        S = cp.scores(ctx["prob"])
        encs = {"RA-GNN": ctx["prob"]}
        for kind, nm in (("gcn", "GCN"), ("mlp", "MLP")):
            c2 = model.train(ctx["X"], y, tr, ctx["va"], ctx["g"], kind=kind, seed=s)
            encs[nm] = model.predict(c2[0], c2[1], ctx["X"])[0]
        cl = classical_probs(ctx["net"], cfg.T, tr, y)
        encs["SL"], encs["BRS"] = cl["SL"], cl["BRS"]
        for a in alphas:
            cov["marginal"][a].append(cp.metrics(cp.sets_from_q(S, cp.marginal_q(S, y, cal, a)), y, te)["cov_M"])
            SS = cp.diffuse(S, ctx["g"][0], ctx["g"][1], len(y), 0.4)
            cov["daps"][a].append(cp.metrics(cp.sets_from_q(SS, cp.marginal_q(SS, y, cal, a)), y, te)["cov_M"])
            cov["calitrust"][a].append(cp.metrics(cset(ctx["prob"], y, cal, (a, a), s), y, te)["cov_M"])
            cov["empirical"][a].append(cp.metrics(empirical_threshold(ctx["prob"], y, cal, 1 - a), y, te)["cov_M"])
            for e, pr in encs.items():
                ver[e][a].append(cp.metrics(cset(pr, y, cal, (AB, a), s), y, te)["verify"])
        if s == 0:
            for e, pr in {**encs, "EigenTrust": cl["EigenTrust"]}.items():
                fpr, tpr, _ = roc_curve(y[te], pr[te, 1])
                grid = np.linspace(0, 1, 101)
                roc_store[e] = np.interp(grid, fpr, tpr)
            # score histogram for R7
            sM = S[te, 1]; yt = y[te]
            bins = np.linspace(0, 1, 26)
            hb, _ = np.histogram(sM[yt == 0], bins); hm, _ = np.histogram(sM[yt == 1], bins)
            save("hist_scores", ["left", "benign", "malicious"],
                 [[bins[i], hb[i] / max(hb.sum(), 1), hm[i] / max(hm.sum(), 1)] for i in range(25)] + [[1.0, 0, 0]])
            qB = cp.cq(S[cal][y[cal] == 0, 0], AB); qM = cp.cq(S[cal][y[cal] == 1, 1], AM)
            json.dump({"qB": float(qB), "qM": float(qM), "thrB_on_sM": float(1 - qB)}, open(f"{OUT}/hist_q.json", "w"))
        print("valid seed", s, flush=True)
    rows = []
    for a in alphas:
        rows.append([1 - a] + sum([[np.mean(cov[m][a]), np.std(cov[m][a])] for m in ("marginal", "daps", "empirical", "calitrust")], []))
    save("validity", ["target", "marg", "margsd", "daps", "dapssd", "emp", "empsd", "cali", "calisd"], rows)
    rows = [[a] + sum([[np.mean(ver[e][a]), np.std(ver[e][a])] for e in ver], []) for a in alphas]
    save("efficiency", ["alpha", "ragnn", "ragnnsd", "gcn", "gcnsd", "mlp", "mlpsd", "sl", "slsd", "brs", "brssd"], rows)
    grid = np.linspace(0, 1, 101)
    save("roc", ["fpr"] + [k.replace(" ", "").replace("-", "") for k in roc_store],
         [[grid[i]] + [roc_store[k][i] for k in roc_store] for i in range(101)])


def exp_attacks(seeds=range(8)):
    """R3: per-attack malicious coverage and verification rate."""
    rows = []
    for atk in ["BS", "BM", "BST", "OO", "SB", "MIX"]:
        acc = {"mc": [], "cc": [], "mv": [], "cv": []}
        for s in seeds:
            ctx = build(Cfg(seed=s, attack=atk), s)
            y, cal, te = ctx["y"], ctx["cal"], ctx["te"]
            S = cp.scores(ctx["prob"])
            Mm = cp.metrics(cp.sets_from_q(S, cp.marginal_q(S, y, cal, 0.10)), y, te)
            Mc = cp.metrics(cset(ctx["prob"], y, cal, (AB, AM), s), y, te)
            acc["mc"].append(Mm["cov_M"]); acc["cc"].append(Mc["cov_M"])
            acc["mv"].append(Mm["verify"]); acc["cv"].append(Mc["verify"])
        rows.append([atk] + [f(acc[k]) for k in ("mc", "cc", "mv", "cv") for f in (np.mean, np.std)])
        print("attack", atk, rows[-1], flush=True)
    save("attacks", ["atk", "margcov", "margcovsd", "calicov", "calicovsd", "margver", "margversd", "caliver", "caliversd"], rows)


def exp_temporal(seeds=range(6), gammas=(0.0, 0.01, 0.02, 0.05, 0.1, 0.2), T0=16, T=46, drift=22, audit=0.05,
                 drift_to=("CAM",), drift_frac=0.8, tag=""):
    """R4 + R5: adaptive adversary switches strategy after deployment (window
    `drift`); the encoder is frozen. Compares static / rolling / ACI."""
    curves = {m: [] for m in ("static", "rolling", "aci")}
    gam = {g: [] for g in gammas}
    for s in seeds:
        cfg = Cfg(seed=s, T=T, pool=("BS", "BM", "BST"), drift_at=drift, drift_frac=drift_frac, drift_to=drift_to)
        net = SIoT(cfg); net.run(T0)
        X, _ = node_features(net, T0); g = relation_graph(net, T0)
        y = net.y; tr, va, cal, te = split(len(y), s)
        p, fwd, _ = model.train(X, y, tr, va, g, seed=s)
        prob0 = model.predict(p, fwd, X)[0]
        S0 = cp.scores(prob0)
        # frozen encoder applied to evolving evidence
        Sw = []
        for w in range(T0, T):
            net.step(w)
            Xw, _ = node_features(net, w + 1); gw = relation_graph(net, w + 1)
            fw = model.make_forward(*gw[:3], len(y), gw[3], "ragnn")
            Sw.append(cp.scores(model.predict(p, fw, Xw)[0]))
        modes = [("static", 0.0), ("rolling", 0.0)] + [("aci", gg) for gg in gammas]
        for mode, gg in modes:
            rng = np.random.default_rng(99 + s)
            oc = cp.OnlineCaliTrust(S0[cal], y[cal], (AB, AM), mode=mode if mode != "aci" else "aci", gamma=gg)
            miss, ver = [], []
            for t, S in enumerate(Sw):
                if mode == "static":
                    refS, refy = oc.init
                else:
                    refS = np.vstack([b[0] for b in oc.buf[-oc.K:]]); refy = np.concatenate([b[1] for b in oc.buf[-oc.K:]])
                a = oc.at if mode == "aci" else oc.alphas
                C = cp.smoothed_sets_from_ref(S, refS, refy, np.clip(a, -1, 1.0), rng)
                M = cp.metrics(C, y, te)
                miss.append(1 - M["cov_M"]); ver.append(M["verify"])
                aud = rng.choice(te, int(audit * len(te)), replace=False)
                oc.update(S[aud], y[aud], C[aud])
            if mode == "aci" and gg == 0.05:
                curves["aci"].append((miss, ver, [h[1] for h in oc.hist]))
            elif mode in ("static", "rolling"):
                curves[mode].append((miss, ver, None))
            if mode == "aci":
                post = slice(drift - T0, None)
                gam[gg].append([np.mean(miss[post]), np.mean(ver[post]), np.mean(np.abs(np.array(miss[post]) - AM))])
        print("temporal seed", s, flush=True)
    rows = []
    for t in range(T - T0):
        r = [T0 + t]
        for m in ("static", "rolling", "aci"):
            r += [np.mean([c[0][t] for c in curves[m]]), np.mean([c[1][t] for c in curves[m]])]
        r += [np.mean([c[2][t] for c in curves["aci"]])]
        rows.append(r)
    save("temporal" + tag, ["w", "stmiss", "stver", "romiss", "rover", "acimiss", "aciver", "alphaM"], rows)
    save("gamma" + tag, ["gamma", "miss", "misssd", "ver", "versd", "dev"],
         [[gg, np.mean([v[0] for v in gam[gg]]), np.std([v[0] for v in gam[gg]]),
           np.mean([v[1] for v in gam[gg]]), np.std([v[1] for v in gam[gg]]), np.mean([v[2] for v in gam[gg]])] for gg in gammas])
    summ = {m: {"post_miss": float(np.mean([np.mean(c[0][drift - T0:]) for c in curves[m]])),
                "post_ver": float(np.mean([np.mean(c[1][drift - T0:]) for c in curves[m]])),
                "pre_miss": float(np.mean([np.mean(c[0][:drift - T0]) for c in curves[m]]))} for m in curves}
    json.dump(summ, open(f"{OUT}/temporal{tag}.json", "w"), indent=1)


def exp_rho(seeds=range(5), rhos=(0.10, 0.20, 0.30, 0.40, 0.45)):
    """R8 malicious-fraction sweep; R13 downstream service success."""
    rows, util = [], []
    for rho in rhos:
        a = {"cc": [], "mc": [], "cv": [], "u": {k: [] for k in ("none", "BRS", "SL", "GCN", "RAGNN", "Cali")}}
        for s in seeds:
            cfg = Cfg(seed=s, rho=rho)
            ctx = build(cfg, s)
            y, cal, te, tr = ctx["y"], ctx["cal"], ctx["te"], ctx["tr"]
            S = cp.scores(ctx["prob"])
            Mc = cp.metrics(cset(ctx["prob"], y, cal, (AB, AM), s), y, te)
            Mm = cp.metrics(cp.sets_from_q(S, cp.marginal_q(S, y, cal, 0.10)), y, te)
            a["cc"].append(Mc["cov_M"]); a["mc"].append(Mm["cov_M"]); a["cv"].append(Mc["verify"])
            # ---- downstream utility: provider selection on a fresh window
            cl = classical_probs(ctx["net"], cfg.T, tr, y)
            c2 = model.train(ctx["X"], y, tr, ctx["va"], ctx["g"], kind="gcn", seed=s)
            pg = model.predict(c2[0], c2[1], ctx["X"])[0]
            known = np.zeros(len(y), bool); known[np.concatenate([tr, ctx["va"], cal])] = True
            trusted = {"none": np.ones(len(y), bool)}
            for nm, pr in (("BRS", cl["BRS"]), ("SL", cl["SL"]), ("GCN", pg), ("RAGNN", ctx["prob"])):
                thr = f1_best_thr(pr[:, 1], y, cal)
                trusted[nm] = pr[:, 1] < thr
            C = cset(ctx["prob"], y, cal, (AB, AM), s)
            amb = C.sum(1) != 1
            trusted["Cali"] = (C[:, 0] & ~C[:, 1]) | (amb & (y == 0))   # ambiguous resolved by attestation
            for k in trusted:
                tk = np.where(known, y == 0, trusted[k]) if k != "none" else trusted[k]
                a["u"][k].append(service_success(ctx["net"], tk, s))
        rows.append([rho] + [f(a[k]) for k in ("cc", "mc", "cv") for f in (np.mean, np.std)])
        util.append([rho] + [np.mean(a["u"][k]) for k in ("none", "BRS", "SL", "GCN", "RAGNN", "Cali")])
        print("rho", rho, rows[-1], util[-1], flush=True)
    save("rho", ["rho", "calicov", "calicovsd", "margcov", "margcovsd", "caliver", "caliversd"], rows)
    save("utility", ["rho", "none", "brs", "sl", "gcn", "ragnn", "cali"], util)


def service_success(net, trusted, seed, req=20):
    """Honest consumers select providers among trusted neighbours (fallback: any
    neighbour); returns mean realised service success rate."""
    rng = np.random.default_rng(555 + seed)
    ok, tot = 0, 0
    w = len(net.log)
    for i in np.flatnonzero(~net.mal):
        nb = net.nbr[i]
        if len(nb) == 0:
            continue
        cand = nb[trusted[nb]]
        if len(cand) == 0:
            cand = nb
        for j in rng.choice(cand, req):
            ok += net._quality(i, int(j), w); tot += 1
    return ok / tot


def exp_budget(seeds=range(6), fcals=(0.05, 0.10, 0.15, 0.25, 0.35)):
    """R9 calibration-budget sweep (train/val fixed)."""
    rows = []
    for fc in fcals:
        a = {"cov": [], "ver": [], "ver_det": []}
        for s in seeds:
            ctx = build(Cfg(seed=s), s, fcal=fc)
            y, cal, te = ctx["y"], ctx["cal"], ctx["te"]
            a["cov"].append(cp.metrics(cset(ctx["prob"], y, cal, (AB, AM), s), y, te)["cov_M"])
            a["ver"].append(cp.metrics(cset(ctx["prob"], y, cal, (AB, AM), s), y, te)["verify"])
            a["ver_det"].append(cp.metrics(cset(ctx["prob"], y, cal, (AB, AM), s, "mondrian_det"), y, te)["verify"])
        nM = int(round(fc * 1000 * 0.2))
        rows.append([fc, nM] + [f(a[k]) for k in ("cov", "ver", "ver_det") for f in (np.mean, np.std)])
        print("budget", fc, rows[-1], flush=True)
    save("budget", ["fcal", "nM", "cov", "covsd", "ver", "versd", "vdet", "vdetsd"], rows)


def exp_splits(n_rep=500):
    """R10 finite-sample validity across random calibration/test re-splits."""
    ctx = build(Cfg(seed=0), 0)
    y = ctx["y"]; pool = np.concatenate([ctx["cal"], ctx["te"]])
    rng = np.random.default_rng(3)
    covs = {"emp": [], "marg": [], "cali": [], "pac": []}
    vers = {"cali": [], "pac": []}
    S = cp.scores(ctx["prob"])
    for r in range(n_rep):
        pp = rng.permutation(pool); cal, te = pp[:250], pp[250:]
        covs["emp"].append(cp.metrics(empirical_threshold(ctx["prob"], y, cal, 0.95), y, te)["cov_M"])
        covs["marg"].append(cp.metrics(cp.sets_from_q(S, cp.marginal_q(S, y, cal, 0.05)), y, te)["cov_M"])
        Mc = cp.metrics(cp.smoothed_sets(S, y, cal, (AB, AM), np.random.default_rng(r)), y, te)
        Mp = cp.metrics(cp.pac_sets(S, y, cal, (AB, AM), 0.1), y, te)
        covs["cali"].append(Mc["cov_M"]); covs["pac"].append(Mp["cov_M"])
        vers["cali"].append(Mc["verify"]); vers["pac"].append(Mp["verify"])
    bins = np.linspace(0.80, 1.0, 21)
    rows = []
    hs = {k: np.histogram(np.clip(v, 0.8, 1.0), bins)[0] / n_rep for k, v in covs.items()}
    for i in range(20):
        rows.append([bins[i], hs["emp"][i], hs["marg"][i], hs["cali"][i], hs["pac"][i]])
    rows.append([1.0, 0, 0, 0, 0])
    save("splits", ["left", "emp", "marg", "cali", "pac"], rows)
    stats = {k: {"mean": float(np.mean(v)), "viol": float(np.mean(np.array(v) < 0.95)), "sd": float(np.std(v))} for k, v in covs.items()}
    for k in vers: stats[k]["verify"] = float(np.mean(vers[k]))
    stats["ymax"] = float(max(max(h) for h in hs.values()))
    json.dump(stats, open(f"{OUT}/splits.json", "w"), indent=1)
    grid = np.linspace(0.80, 1.0, 161)
    save("splits_ecdf", ["x", "emp", "marg", "cali", "pac"],
         [[x] + [float(np.mean(np.array(covs[k]) <= x + 1e-12)) for k in ("emp", "marg", "cali", "pac")] for x in grid])
    print(stats)


def exp_scalability(Ns=(500, 1000, 2000, 5000, 10000)):
    rows = []
    for N in Ns:
        cfg = Cfg(N=N, seed=0)
        t0 = time.time(); net = SIoT(cfg).run(); tsim = time.time() - t0
        t0 = time.time(); X, _ = node_features(net, cfg.T); g = relation_graph(net, cfg.T); tfeat = time.time() - t0
        y = net.y; tr, va, cal, te = split(N, 0)
        p, fwd, ttrain = model.train(X, y, tr, va, g, epochs=100, patience=1000, seed=0)
        model.predict(p, fwd, X)   # compile
        t0 = time.time(); prob = model.predict(p, fwd, X)[0]; tinf = time.time() - t0
        t0 = time.time(); cset(prob, y, cal, (AB, AM), 0); tconf = time.time() - t0
        t0 = time.time(); bl.eigentrust(net, cfg.T, tr[y[tr] == 0]); tet = time.time() - t0
        t0 = time.time(); bl.subjective_logic(net, cfg.T); tsl = time.time() - t0
        rows.append([N, len(g[0]), tfeat, ttrain / 100 * 1000, tinf * 1000, tconf * 1000, tet * 1000, tsl * 1000])
        print("scal", rows[-1], flush=True)
    save("scal", ["N", "E", "tfeat", "epoch_ms", "inf_ms", "conf_ms", "et_ms", "sl_ms"], rows)


def exp_ablation(seeds=range(10)):
    variants = {
        "full": dict(),
        "no_relatt": dict(kind="gcn"),
        "no_signed": dict(signed=False),
        "no_rater": dict(groups=("P", "T", "S")),
        "no_temporal": dict(groups=("P", "R", "S")),
    }
    res = {}
    for s in seeds:
        for nm, kw in variants.items():
            ctx = build(Cfg(seed=s), s, **kw)
            y, cal, te = ctx["y"], ctx["cal"], ctx["te"]
            M = cp.metrics(cset(ctx["prob"], y, cal, (AB, AM), s), y, te)
            res.setdefault(nm, []).append([roc_auc_score(y[te], ctx["prob"][te, 1]), M["cov_M"], M["verify"], M["miss_M"]])
            if nm == "full":
                S = cp.scores(ctx["prob"])
                for nm2, C in (("no_mondrian", cp.sets_from_q(S, cp.marginal_q(S, y, cal, 0.10))),
                               ("no_smoothing", cset(ctx["prob"], y, cal, (AB, AM), s, "mondrian_det")),
                               ("plus_uniform_diff", cset(ctx["prob"], y, cal, (AB, AM), s, "daps_mond", ctx["g"])),
                               ("plus_gated_diff", cset(ctx["prob"], y, cal, (AB, AM), s, "egd_mond", ctx["g"], ctx["H"]))):
                    M = cp.metrics(C, y, te)
                    res.setdefault(nm2, []).append([roc_auc_score(y[te], ctx["prob"][te, 1]), M["cov_M"], M["verify"], M["miss_M"]])
        print("ablation seed", s, flush=True)
    out = {k: [np.mean(v, 0).tolist(), np.std(v, 0).tolist()] for k, v in res.items()}
    out["raw_verify"] = {k: [r[2] for r in v] for k, v in res.items()}
    out["raw_auc"] = {k: [r[0] for r in v] for k, v in res.items()}
    json.dump(out, open(f"{OUT}/ablation.json", "w"), indent=1)


def exp_attention(seeds=range(5)):
    """Mean learned relation attention per attack (interpretability, used in text)."""
    rows = {}
    for atk in ["BS", "BM", "BST", "OO", "SB"]:
        acc = []
        for s in seeds:
            ctx = build(Cfg(seed=s, attack=atk), s)
            a = ctx["att"][-1]                       # (R,N) last layer
            acc.append(a[:, ctx["y"] == 1].mean(1))
        rows[atk] = np.mean(acc, 0).tolist()
    json.dump(rows, open(f"{OUT}/attention.json", "w"), indent=1)


if __name__ == "__main__":
    which = sys.argv[1:]
    fns = {"main": exp_main, "valid": exp_validity_efficiency, "attacks": exp_attacks, "temporal": exp_temporal,
           "rho": exp_rho, "budget": exp_budget, "splits": exp_splits, "scal": exp_scalability,
           "ablation": exp_ablation, "attention": exp_attention,
           "temporal_sleep": lambda: exp_temporal(drift_to=("SLEEP",), drift_frac=0.8, tag=""),
           "temporal_cam": lambda: exp_temporal(seeds=range(4), drift_to=("CAM",), drift_frac=0.8, tag="_cam"),
           "temporal_oosb": lambda: exp_temporal(drift_to=("OO", "SB"), drift_frac=0.6, tag="_oosb")}
    for w in which:
        t = time.time(); fns[w](); print(f"== {w} done in {time.time()-t:.0f}s", flush=True)
