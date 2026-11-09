"""Revision experiment: post-calibration evidence injection on real networks.
Extends robust.py with (i) frozen feature standardisation (statistics fixed at
calibration time) next to the original refit protocol, (ii) outcome decomposition of
malicious test users, (iii) insider collusion by existing malicious accounts,
(iv) admission thresholds m in {3,5,10}, (v) audit-driven recalibration (rolling + ACI)
engaged after the injection."""
import json, os, sys, time
import numpy as np
import pandas as pd
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
from .realdata import load_network, SignedNet
from .semisyn import inject
from .features import node_features, relation_graph
from . import model, conformal as cp
from .revcommon import decompose, replay_online, AB, AM

OUT = os.path.abspath(os.environ.get("CALITRUST_OUT", "results"))
os.makedirs(OUT, exist_ok=True)
DATA = os.environ.get("CALITRUST_DATA", "data")
AL = np.array((AB, AM))
KS = (0, 1, 3, 5, 10, 20)
KIN = (1, 3, 5, 10)
MS = (0, 3, 5, 10)


def admit(d, m):
    if m <= 0:
        return d
    cnt = d.s.value_counts()
    return d[d.s.map(cnt).fillna(0).values >= m]


def build_raw(d, labels, m):
    net = SignedNet(admit(d, m), labels, drop_label_edges=False)
    Xr, _ = node_features(net, net.cfg.T, raw=True)
    g = relation_graph(net, net.cfg.T)
    return net, Xr, g


def std(X, stats):
    return ((X - stats[0]) / stats[1]).astype(np.float32)


def scores(prm, net, X, g):
    fwd = model.make_forward(g[0], g[1], g[2], net.cfg.N, g[3], "ragnn")
    return cp.scores(model.predict(prm, fwd, X)[0])


def add_rows(d, rows):
    return pd.concat([d, pd.DataFrame(rows, columns=["s", "t", "w", "time"])], ignore_index=True)


def sybils(d, targets, k, extra, honest, rng):
    if k == 0:
        return d, []
    nid = int(max(d.s.max(), d.t.max())) + 1
    tmax = float(d.time.max()); tmin = float(d.time.min())
    rows, syb = [], []
    for a in targets:
        for _ in range(k):
            rows.append((nid, int(a), 1.0, tmax)); syb.append(nid)
            for h in (rng.choice(honest, extra) if extra > 0 else []):
                rows.append((nid, int(h), 0.8, float(rng.uniform(tmin, tmax))))
            nid += 1
    return add_rows(d, rows), syb


def insiders(d, targets, k, pool, rng):
    tmax = float(d.time.max())
    rows = [(int(b), int(a), 1.0, tmax) for a in targets for b in rng.choice(pool, k, replace=False)]
    return add_rows(d, rows)


def evaluate(prm, net, Xr, g, N0, stats_c, refS, refy, y0, te, sd):
    out = {}
    stats_r = (Xr.mean(0), Xr.std(0) + 1e-6)
    for proto, st in (("refit", stats_r), ("frozen", stats_c)):
        S = scores(prm, net, std(Xr, st), g)[:N0]
        C = cp.smoothed_sets_from_ref(S, refS, refy, AL, np.random.default_rng(7 + sd))
        M = cp.metrics(C, y0, te)
        out[proto] = dict(miss=float(M["miss_M"]), verify=float(M["verify"]), iso=float(M["isolate_B"]),
                          dec=decompose(C, y0, te)["M"], S=S)
    return out


def run(path, name, seeds=range(3)):
    t0 = time.time()
    d0 = load_network(path)
    res = {}
    aci = {}

    def put(key, k, val):
        res.setdefault(key, {}).setdefault(str(k), []).append(val)

    for sd in seeds:
        d, lab, st = inject(d0, seed=sd)
        elig = st.pop("eligible"); st.pop("typ_map")
        N0 = len(lab); y0 = np.array([lab[i] for i in range(N0)])
        p = np.random.default_rng(1000 + sd).permutation(elig); n = len(p)
        tr, va, cal, te = p[:int(.2 * n)], p[int(.2 * n):int(.25 * n)], p[int(.25 * n):int(.5 * n)], p[int(.5 * n):]
        tm = te[y0[te] == 1]
        honest = elig[y0[elig] == 0]
        ins_pool = np.concatenate([tr, va, cal]); ins_pool = ins_pool[y0[ins_pool] == 1]
        for m in MS:
            net, Xr, g = build_raw(d, lab, m)
            stats_c = (Xr.mean(0), Xr.std(0) + 1e-6)
            X = std(Xr, stats_c)
            prm, fwd, _ = model.train(X, net.y, tr, va, g, seed=sd)
            S0 = scores(prm, net, X, g)[:N0]
            refS, refy = S0[cal], y0[cal]
            extra = max(m, 3)
            configs = []
            if m in (0, 3):
                configs += [("fresh", k) for k in KS] + [("insider", k) for k in KIN]
            configs += [("est", k) for k in KS if k > 0 or m not in (0, 3)]
            for kind, k in configs:
                rng = np.random.default_rng(11 + sd + k)
                if kind == "fresh":
                    da, syb = sybils(d, tm, k, 0, honest, rng)
                elif kind == "est":
                    da, syb = sybils(d, tm, k, extra, honest, rng)
                else:
                    da, syb = insiders(d, tm, k, ins_pool, rng), []
                lab_a = dict(lab); lab_a.update({s_: 0 for s_ in syb})
                neta, Xa, ga = build_raw(da, lab_a, m)
                ev = evaluate(prm, neta, Xa, ga, N0, stats_c, refS, refy, y0, te, sd)
                for proto in ("refit", "frozen"):
                    e = ev[proto]
                    put(f"m{m}_{kind}_{proto}", k, [e["miss"], e["verify"], e["iso"], e["dec"]["trust"],
                                                    e["dec"]["attest_both"], e["dec"]["attest_empty"], e["dec"]["isolate"]])
                # audit-driven recalibration after the injection (frozen protocol)
                if (m, kind, k) in ((0, "fresh", 1), (0, "fresh", 10), (3, "est", 5), (3, "est", 20), (3, "insider", 3), (3, "insider", 10)):
                    S = ev["frozen"]["S"]
                    for mode in ("static", "rolling", "aci"):
                        rr = replay_online(S, y0, None, te, windows=20, mode=mode, seed=sd, ref=(refS, refy))
                        aci.setdefault(f"m{m}_{kind}_{k}", {}).setdefault(mode, []).append(rr)
            if m == 0:
                for keep in (1.0, 0.5, 0.3, 0.1):
                    rng = np.random.default_rng(3 + sd)
                    mask = np.isin(d.s.values, tm) | np.isin(d.t.values, tm)
                    dw = d[~(mask & (rng.random(len(d)) > keep))]
                    netw, Xw, gw = build_raw(dw, lab, 0)
                    ev = evaluate(prm, netw, Xw, gw, N0, stats_c, refS, refy, y0, te, sd)
                    for proto in ("refit", "frozen"):
                        e = ev[proto]
                        put(f"whitewash_{proto}", keep, [e["miss"], e["verify"], e["iso"], e["dec"]["trust"],
                                                         e["dec"]["attest_both"], e["dec"]["attest_empty"], e["dec"]["isolate"]])
            print(name, "seed", sd, "m", m, f"({time.time()-t0:.0f}s)", flush=True)
    summ = {key: {k: [np.mean(v, 0).tolist(), np.std(v, 0).tolist()] for k, v in curves.items()} for key, curves in res.items()}
    summ["_cols"] = ["miss", "verify", "iso", "M_trust", "M_attest_both", "M_attest_empty", "M_isolate"]
    summ["_aci"] = {cfg: {mode: {"miss_w": np.mean([r["miss"] for r in rs], 0).tolist(),
                                 "ver_w": np.mean([r["ver"] for r in rs], 0).tolist(),
                                 "alphaM_w": np.mean([r["alphaM"] for r in rs], 0).tolist(),
                                 "miss_first4": float(np.mean([np.mean(r["miss"][:4]) for r in rs])),
                                 "miss_last10": float(np.mean([np.mean(r["miss"][10:]) for r in rs])),
                                 "ver_last10": float(np.mean([np.mean(r["ver"][10:]) for r in rs]))}
                          for mode, rs in modes.items()} for cfg, modes in aci.items()}
    summ["_raw"] = res
    os.makedirs(OUT, exist_ok=True)
    json.dump(summ, open(f"{OUT}/rev_robust_{name}.json", "w"), indent=1)
    return summ


if __name__ == "__main__":
    import jax
    nm = sys.argv[1]
    path = {"otc": os.path.join(DATA, "soc-sign-bitcoinotc.csv.gz"), "alpha": os.path.join(DATA, "soc-sign-bitcoinalpha.csv.gz")}[nm]
    run(path, nm)
