"""
SeqCert: sequential certified distrust with anytime-valid alpha spending.

Instead of attesting every ambiguous device at once, the manager re-examines undecided devices
at K looks as evidence accumulates. At look k the class-c budget is alpha_{c,k} with
sum_k alpha_{c,k} = alpha_c, and p-values are computed against calibration anchors observed at
the SAME look (same evidence age). A device is trusted at the first look where its set is {B},
isolated at the first look where it is {M}, and attested if still undecided after look K.

Validity (Theorem 4): the event "malicious device trusted" implies that at some look k the set
excluded M, so P[trusted | M] <= sum_k P[M not in C_k | M] = sum_k alpha_{M,k} = alpha_M,
for any stopping behaviour. Naive re-testing at full alpha each look has no such bound.

The spending schedule w_k ∝ r^k is tuned once on an independent pilot network (never on the
calibration or test devices), so it cannot affect validity.
"""
import json, os, sys, time
import numpy as np
from .sim import SIoT, Cfg
from .features import node_features, relation_graph
from . import model, conformal as cp

OUT = os.path.abspath(os.environ.get("CALITRUST_OUT", "results"))
os.makedirs(OUT, exist_ok=True)
DATA = os.environ.get("CALITRUST_DATA", "data")
AB, AM = 0.10, 0.05
RS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)          # candidate geometric spending ratios


def pvals(S_k, y, cal, rng):
    """Smoothed class-conditional conformal p-values at one look. Returns (N,2)."""
    N = S_k.shape[0]; P = np.zeros((N, 2))
    for c in (0, 1):
        sc = np.sort(S_k[cal][y[cal] == c, c]); n = len(sc); s = S_k[:, c]
        gt = n - np.searchsorted(sc, s, side="right")
        eq = np.searchsorted(sc, s, side="right") - np.searchsorted(sc, s, side="left")
        P[:, c] = (gt + rng.random(N) * (eq + 1)) / (n + 1)
    return P


def schedule(K, kind, r=1.0, alphas=(AB, AM)):
    """(K,2) per-look budgets."""
    a = np.array(alphas, float)
    if kind == "naive":
        return np.tile(a, (K, 1))
    if kind == "first":
        W = np.zeros(K); W[0] = 1
    elif kind == "last":
        W = np.zeros(K); W[-1] = 1
    elif kind == "uniform":
        W = np.ones(K)
    else:  # geometric
        W = r ** np.arange(K)
    W = W / W.sum()
    return W[:, None] * a[None, :]


def decide(Ps, y, te, A, attest_first=False):
    """Run the sequential rule. Ps: list of (N,2) p-values per look; A: (K,2) budgets.
    kind 'first' is the one-shot baseline: decide at look 0, attest the rest."""
    K = len(Ps)
    state = np.zeros(len(te), int)       # 0 undecided, 1 trusted, 2 isolated
    when = np.full(len(te), -1)
    for k in range(K):
        if A[k].sum() == 0:
            continue
        P = Ps[k][te]
        inB = P[:, 0] > A[k, 0]; inM = P[:, 1] > A[k, 1]
        und = state == 0
        tr = und & inB & ~inM; iso = und & inM & ~inB
        state[tr] = 1; state[iso] = 2; when[tr | iso] = k
    yy = y[te]
    dec = state > 0
    return {"miss_M": float(((state == 1) & (yy == 1)).sum() / max((yy == 1).sum(), 1)),
            "iso_B": float(((state == 2) & (yy == 0)).sum() / max((yy == 0).sum(), 1)),
            "attest": float((state == 0).mean()),
            "delay": float(when[dec].mean()) if dec.any() else np.nan,
            "delay_M": float(when[dec & (yy == 1)].mean()) if (dec & (yy == 1)).any() else np.nan,
            "decided_first": float((when == 0).mean())}


def looks_sim(seed, T0=12, gap=3, K=6, N=1000):
    cfg = Cfg(seed=seed, N=N, T=T0 + gap * (K - 1))
    net = SIoT(cfg).run()
    y = net.y
    perm = np.random.default_rng(1000 + seed).permutation(N)
    tr, va, cal, te = perm[:int(.2 * N)], perm[int(.2 * N):int(.25 * N)], perm[int(.25 * N):int(.5 * N)], perm[int(.5 * N):]
    S = []
    for k in range(K):
        w = T0 + gap * k
        X, _ = node_features(net, w); g = relation_graph(net, w)
        prm, fwd, _ = model.train(X, y, tr, va, g, seed=seed)
        S.append(cp.scores(model.predict(prm, fwd, X)[0]))
    return S, y, cal, te


def evaluate(S, y, cal, te, seed, r_star):
    rng = np.random.default_rng(77 + seed)
    Ps = [pvals(Sk, y, cal, rng) for Sk in S]
    K = len(S)
    out = {}
    for nm, A in (("One-shot (look 1)", schedule(K, "first")), ("One-shot (last look)", schedule(K, "last")),
                  ("Naive re-testing", schedule(K, "naive")), ("SeqCert (uniform)", schedule(K, "uniform")),
                  ("SeqCert (tuned)", schedule(K, "geo", r_star))):
        out[nm] = decide(Ps, y, te, A)
    # per-look diagnostics for the figure: one-shot attest rate if one stopped at look k
    out["_per_look_attest"] = [decide([Ps[k]], y, te, schedule(1, "first"))["attest"] for k in range(K)]
    return out


def tune_r(S, y, cal, te, seed, target_miss=None):
    """Pick r minimising attestation on a pilot network (independent of evaluation networks)."""
    rng = np.random.default_rng(5 + seed)
    Ps = [pvals(Sk, y, cal, rng) for Sk in S]
    res = {r: decide(Ps, y, te, schedule(len(S), "geo", r))["attest"] for r in RS}
    return min(res, key=res.get), res


def run_sim(seeds=range(10), pilot_seed=500):
    t0 = time.time()
    Sp, yp, calp, tep = looks_sim(pilot_seed)
    r_star, curve = tune_r(Sp, yp, calp, tep, pilot_seed)
    print("pilot r*", r_star, curve, f"({time.time()-t0:.0f}s)", flush=True)
    allres = []
    for s in seeds:
        S, y, cal, te = looks_sim(s)
        allres.append(evaluate(S, y, cal, te, s, r_star))
        print("seq sim seed", s, {k: round(v["attest"], 3) for k, v in allres[-1].items() if not k.startswith("_")}, f"({time.time()-t0:.0f}s)", flush=True)
    return summarize(allres, {"r_star": r_star, "pilot_curve": {str(k): v for k, v in curve.items()}}, "sim")


def summarize(allres, extra, tag):
    methods = [k for k in allres[0] if not k.startswith("_")]
    summ = {m: {q: [float(np.nanmean([a[m][q] for a in allres])), float(np.nanstd([a[m][q] for a in allres]))]
                for q in allres[0][m]} for m in methods}
    summ["_per_look_attest"] = np.mean([a["_per_look_attest"] for a in allres], 0).tolist()
    summ["_raw_attest"] = {m: [a[m]["attest"] for a in allres] for m in methods}
    summ["_raw_miss"] = {m: [a[m]["miss_M"] for a in allres] for m in methods}
    summ["_meta"] = extra
    os.makedirs(OUT, exist_ok=True)
    json.dump(summ, open(f"{OUT}/seqcert_{tag}.json", "w"), indent=1)
    return summ


# ---------------------------------------------------------------- real networks
def looks_real(path, seed, K=6, W=20, first=10):
    from .realdata import load_network, SignedNet
    from .semisyn import inject
    d0 = load_network(path)
    d, labels, st = inject(d0, seed=seed)
    elig = st.pop("eligible")
    net = SignedNet(d, labels, windows=W, drop_label_edges=False)
    y = net.y
    p = np.random.default_rng(1000 + seed).permutation(elig); n = len(p)
    tr, va, cal, te = p[:int(.2 * n)], p[int(.2 * n):int(.25 * n)], p[int(.25 * n):int(.5 * n)], p[int(.5 * n):]
    gap = (W - first) // (K - 1)
    S = []
    for k in range(K):
        w = first + gap * k
        X, _ = node_features(net, w); g = relation_graph(net, w)
        prm, fwd, _ = model.train(X, y, tr, va, g, seed=seed)
        S.append(cp.scores(model.predict(prm, fwd, X)[0]))
    return S, y, cal, te


def run_real(path, name, pilot_path, seeds=range(5)):
    """Schedule tuned on the OTHER network (pilot), evaluated on this one."""
    t0 = time.time()
    Sp, yp, calp, tep = looks_real(pilot_path, 900)
    r_star, curve = tune_r(Sp, yp, calp, tep, 900)
    print(name, "pilot r*", r_star, curve, f"({time.time()-t0:.0f}s)", flush=True)
    allres = []
    for s in seeds:
        S, y, cal, te = looks_real(path, s)
        allres.append(evaluate(S, y, cal, te, s, r_star))
        print("seq", name, "seed", s, {k: round(v["attest"], 3) for k, v in allres[-1].items() if not k.startswith("_")}, f"({time.time()-t0:.0f}s)", flush=True)
    return summarize(allres, {"r_star": r_star, "pilot_curve": {str(k): v for k, v in curve.items()}}, name)


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "sim":
        run_sim()
    elif which == "otc":
        run_real(os.path.join(DATA, "soc-sign-bitcoinotc.csv.gz"), "otc", os.path.join(DATA, "soc-sign-bitcoinalpha.csv.gz"))
    elif which == "alpha":
        run_real(os.path.join(DATA, "soc-sign-bitcoinalpha.csv.gz"), "alpha", os.path.join(DATA, "soc-sign-bitcoinotc.csv.gz"))
