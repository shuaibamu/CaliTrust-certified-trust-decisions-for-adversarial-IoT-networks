"""
Label-free node evidence features and relation-typed message-passing graph.
Feature groups (used by the ablation study):
  P : provider evidence (received ratings, per-relation, dispersion, selectivity)
  R : rater-behaviour evidence (consensus deviation, concentration, reciprocity)
  T : temporal evidence (volatility, long/short gap)
  S : structural evidence (per-relation degree)
"""
import numpy as np
from .sim import N_REL

EWMA = 0.8


def _bincount_mean(idx, val, N, default):
    s = np.bincount(idx, weights=val, minlength=N)
    n = np.bincount(idx, minlength=N)
    out = np.full(N, default, float)
    m = n > 0
    out[m] = s[m] / n[m]
    return out, n


def node_features(net, upto, groups=("P", "R", "T", "S"), stats=None, raw=False):
    """Features at the end of window `upto` (exclusive), using logs [0, upto)."""
    N = net.cfg.N
    logs = net.log[:upto]
    W = len(logs)
    # EWMA weights over windows
    wts = np.array([EWMA ** (W - 1 - t) for t in range(W)])
    cons = np.concatenate([l[0] for l in logs]); prov = np.concatenate([l[1] for l in logs])
    rel = np.concatenate([l[2] for l in logs]); rat = np.concatenate([l[3] for l in logs]).astype(float)
    ww = np.concatenate([np.full(len(l[0]), wts[t]) for t, l in enumerate(logs)])
    gmean = rat.mean()
    feats, names = [], []

    def add(v, n):
        feats.append(v); names.append(n)

    # ---- provider evidence
    s = np.bincount(prov, weights=ww * rat, minlength=N); n = np.bincount(prov, weights=ww, minlength=N)
    rec_mean = np.where(n > 0, s / np.maximum(n, 1e-9), gmean)
    per_win = np.full((W, N), np.nan)
    for t, l in enumerate(logs):
        m, c = _bincount_mean(l[1], l[3].astype(float), N, np.nan)
        per_win[t] = m
    last = per_win[-1]; last = np.where(np.isnan(last), rec_mean, last)
    if "P" in groups:
        add(rec_mean, "rec_mean_ewma"); add(last, "rec_mean_last")
        add(np.log1p(np.bincount(prov, minlength=N) / W), "rec_count")
        for k in range(N_REL + 1):
            m = rel == k
            v, c = _bincount_mean(prov[m], rat[m], N, np.nan)
            add(np.where(np.isnan(v), rec_mean, v), f"rec_rel{k}")
        # dispersion across raters
        sq, _ = _bincount_mean(prov, rat ** 2, N, 0.0)
        add(np.sqrt(np.maximum(sq - rec_mean ** 2, 0)), "rec_disp")
        # selectivity: rating from influential vs non-influential consumers
        inf = net.influential[cons]
        a, _ = _bincount_mean(prov[inf], rat[inf], N, np.nan)
        b, _ = _bincount_mean(prov[~inf], rat[~inf], N, np.nan)
        add(np.nan_to_num(a - b, nan=0.0), "selectivity")
    # ---- rater behaviour
    if "R" in groups:
        cons_prov_mean, _ = _bincount_mean(prov, rat, N, gmean)
        dev = np.abs(rat - cons_prov_mean[prov])
        d, _ = _bincount_mean(cons, dev, N, 0.0)
        add(d, "rater_dev")
        g, _ = _bincount_mean(cons, rat, N, gmean)
        add(g, "given_mean")
        # rating to malicious-looking (low consensus) vs high-consensus providers
        low = cons_prov_mean[prov] < np.median(cons_prov_mean)
        hi_to_low, _ = _bincount_mean(cons[low], rat[low], N, 0.0)
        add(hi_to_low, "pos_to_lowrep")
        # concentration (Herfindahl) of rated partners
        key = cons.astype(np.int64) * N + prov
        uk, cnt = np.unique(key, return_counts=True)
        ci = uk // N
        tot = np.bincount(ci, weights=cnt, minlength=N)
        hh = np.bincount(ci, weights=cnt.astype(float) ** 2, minlength=N) / np.maximum(tot, 1) ** 2
        add(hh, "concentration")
        # reciprocity of positive ratings
        pos = rat > 0.5
        pk = set(zip(cons[pos].tolist(), prov[pos].tolist()))
        recip = np.array([(b, a) in pk for a, b in zip(cons[pos], prov[pos])], float)
        r_, _ = _bincount_mean(cons[pos], recip, N, 0.0)
        add(r_, "reciprocity")
    # ---- temporal evidence
    if "T" in groups:
        K = min(6, W)
        pw = per_win[-K:]
        fill = np.where(np.isnan(pw), rec_mean[None, :], pw)
        add(fill.std(0), "volatility")
        short = fill[-3:].mean(0); longm = np.nanmean(np.where(np.isnan(per_win), rec_mean[None, :], per_win), 0)
        add(short - longm, "short_long_gap")
        add(np.abs(np.diff(fill, axis=0)).mean(0) if K > 1 else np.zeros(N), "mean_abs_change")
    # ---- structural
    if "S" in groups:
        for k in range(N_REL):
            E = net.edges[k]
            add(np.log1p(np.bincount(E.ravel(), minlength=N)), f"deg_rel{k}")
    X = np.stack(feats, 1).astype(np.float32)
    if raw:
        return X, names
    if stats is None:
        stats = (X.mean(0), X.std(0) + 1e-6)
    X = (X - stats[0]) / stats[1]
    return X, names


def relation_graph(net, upto, signed=True):
    """Typed edges for message passing: 4 SIoT relations + INT+ / INT- (signed
    interaction evidence, last 6 windows). Returns src, dst, etype, weight, R."""
    N = net.cfg.N
    src, dst, et = [], [], []
    for k in range(N_REL):
        E = net.edges[k]
        src += [E[:, 0], E[:, 1]]; dst += [E[:, 1], E[:, 0]]; et += [np.full(2 * len(E), k)]
    R = N_REL
    logs = net.log[max(0, upto - 6):upto]
    cons = np.concatenate([l[0] for l in logs]); prov = np.concatenate([l[1] for l in logs])
    rat = np.concatenate([l[3] for l in logs]).astype(float)
    key = cons.astype(np.int64) * N + prov
    uk, inv = np.unique(key, return_inverse=True)
    mr = np.bincount(inv, weights=rat) / np.bincount(inv)
    a, b = uk // N, uk % N
    if signed:
        for k, m in ((R, mr >= 0.5), (R + 1, mr < 0.5)):
            # message flows consumer->provider and provider->consumer
            src += [a[m], b[m]]; dst += [b[m], a[m]]; et += [np.full(2 * m.sum(), k)]
        R += 2
    else:
        src += [a, b]; dst += [b, a]; et += [np.full(2 * len(a), R)]
        R += 1
    src = np.concatenate(src); dst = np.concatenate(dst); et = np.concatenate(et)
    return src.astype(np.int32), dst.astype(np.int32), et.astype(np.int32), R
