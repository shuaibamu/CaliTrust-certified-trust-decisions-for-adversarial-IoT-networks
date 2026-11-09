"""
Classical trust baselines, re-implemented from their original formulations.
Each returns a maliciousness score m_i in [0,1] (higher = less trustworthy).
  BRS        Beta reputation (Josang & Ismail, 2002) with forgetting factor
  EigenTrust Kamvar et al. (2003), pre-trusted = labelled benign anchors
  SL         Subjective-logic opinions with credibility-discounted fusion
"""
import numpy as np

FORGET = 0.8


def _stack(net, upto):
    logs = net.log[:upto]; W = len(logs)
    cons = np.concatenate([l[0] for l in logs]); prov = np.concatenate([l[1] for l in logs])
    rat = np.concatenate([l[3] for l in logs]).astype(float)
    ww = np.concatenate([np.full(len(l[0]), FORGET ** (W - 1 - t)) for t, l in enumerate(logs)])
    return cons, prov, rat, ww


def brs(net, upto):
    N = net.cfg.N
    c, p, r, w = _stack(net, upto)
    a = np.bincount(p, weights=w * r, minlength=N); b = np.bincount(p, weights=w * (1 - r), minlength=N)
    return 1 - (a + 1) / (a + b + 2)


def eigentrust(net, upto, pretrusted, a=0.15, iters=50):
    N = net.cfg.N
    c, p, r, w = _stack(net, upto)
    import scipy.sparse as sp
    sat = sp.coo_matrix((w * (2 * r - 1), (c, p)), shape=(N, N)).tocsr()
    sat.data = np.maximum(sat.data, 0); sat.eliminate_zeros()
    rs = np.asarray(sat.sum(1)).ravel()
    pre = np.zeros(N); pre[pretrusted] = 1; pre /= pre.sum()
    Cn = sp.diags(1 / np.maximum(rs, 1e-12)) @ sat
    dangling = rs <= 0
    t = pre.copy()
    for _ in range(iters):
        t = (1 - a) * (Cn.T @ t + t[dangling].sum() * pre) + a * pre
    rank = t.argsort().argsort() / (N - 1)
    return 1 - rank


def subjective_logic(net, upto, iters=5):
    N = net.cfg.N
    c, p, r, w = _stack(net, upto)
    cred = np.ones(N)
    for _ in range(iters):
        wc = w * cred[c]
        rr = np.bincount(p, weights=wc * r, minlength=N); ss = np.bincount(p, weights=wc * (1 - r), minlength=N)
        bel = rr / (rr + ss + 2); dis = ss / (rr + ss + 2); unc = 2 / (rr + ss + 2)
        exp = bel + 0.5 * unc                              # expected probability
        # rater credibility = 1 - mean deviation from current consensus
        dev = np.abs(r - exp[p])
        d = np.bincount(c, weights=dev, minlength=N) / np.maximum(np.bincount(c, minlength=N), 1)
        cred = np.clip(1 - 2 * d, 0.05, 1.0) * exp.clip(0.05, 1)
    return 1 - exp
