"""
Conformal trust layer.
  score s(x, y) = 1 - p_y(x)            (least-ambiguous set score, Sadinle et al.)
  Mondrian (class-conditional) quantiles  -> P(y in C | Y=y) >= 1 - alpha_y
  Marginal split CP                     -> P(Y in C) >= 1 - alpha
  Diffusion:  uniform (DAPS-style) or embedding-gated (proposed, EGD)
  Online:     rolling audit calibration + per-class adaptive conformal inference
Decision map: {B} -> trust, {M} -> isolate, {B,M} or {} -> verify (attest)
"""
import numpy as np


def scores(prob):
    return 1.0 - prob  # (N,2): column y = score if label were y


def cq(s, alpha):
    """Finite-sample conformal quantile of calibration scores s."""
    n = len(s)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return np.sort(s)[k - 1]


def mondrian_q(S, y, cal, alphas):
    return np.array([cq(S[cal][y[cal] == c, c], alphas[c]) for c in (0, 1)])


def marginal_q(S, y, cal, alpha):
    q = cq(S[cal, y[cal]], alpha)
    return np.array([q, q])


def sets_from_q(S, q):
    return S <= q[None, :]           # (N,2) boolean membership


def diffuse(S, src, dst, N, lam=0.3, H=None, tau=4.0):
    """One-step score diffusion. H=None -> uniform weights (DAPS-style);
    otherwise edge weight = max(cos(h_i,h_j),0)^tau (embedding gate).
    Label-free and permutation-equivariant => exchangeability is preserved."""
    if H is None:
        w = np.ones(len(src))
    else:
        Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
        w = np.maximum((Hn[src] * Hn[dst]).sum(1), 0.0) ** tau
    num = np.zeros_like(S)
    np.add.at(num, dst, w[:, None] * S[src])
    den = np.bincount(dst, weights=w, minlength=N)[:, None]
    nb = np.where(den > 1e-9, num / np.maximum(den, 1e-9), S)
    return (1 - lam) * S + lam * nb


def metrics(C, y, idx):
    C = C[idx]; yy = y[idx]
    cov = C[np.arange(len(yy)), yy]
    size = C.sum(1)
    single = size == 1
    out = {
        "cov": cov.mean(),
        "cov_M": cov[yy == 1].mean() if (yy == 1).any() else np.nan,
        "cov_B": cov[yy == 0].mean() if (yy == 0).any() else np.nan,
        "size": size.mean(),
        "verify": (size != 1).mean(),                  # ambiguous or empty -> attest
        "singleton": single.mean(),
        "single_acc": (cov[single]).mean() if single.any() else np.nan,
        # certified-trust miss: malicious node whose set is exactly {B}
        "miss_M": ((C[:, 0] & ~C[:, 1])[yy == 1]).mean() if (yy == 1).any() else np.nan,
        # benign nodes wrongly isolated: set exactly {M}
        "isolate_B": ((~C[:, 0] & C[:, 1])[yy == 0]).mean() if (yy == 0).any() else np.nan,
    }
    return out


class OnlineCaliTrust:
    """Rolling audit-driven Mondrian calibration + per-class ACI.
    mode: 'static' (initial cal only), 'rolling' (recent audits, fixed alpha),
          'aci' (rolling + adaptive alpha_t)  <- proposed"""

    def __init__(self, S0, y0, alphas, mode="aci", gamma=0.02, K=8):
        self.alphas = np.array(alphas, float)
        self.at = np.array(alphas, float)
        self.mode, self.gamma, self.K = mode, gamma, K
        self.buf = [(S0, y0)]           # list of (scores, labels) per window
        self.init = (S0, y0)
        self.hist = []

    def quantiles(self):
        if self.mode == "static":
            S, y = self.init
        else:
            S = np.vstack([b[0] for b in self.buf[-self.K:]]); y = np.concatenate([b[1] for b in self.buf[-self.K:]])
        a = self.at if self.mode == "aci" else self.alphas
        q = []
        for c in (0, 1):
            ac = np.clip(a[c], 1e-4, 1.0)
            q.append(np.inf if a[c] <= 0 else cq(S[y == c, c], ac))
        return np.array(q)

    def update(self, S_aud, y_aud, C_aud):
        """Feedback from randomly audited nodes (unbiased)."""
        for c in (0, 1):
            m = y_aud == c
            if m.any():
                err = 1.0 - C_aud[m, c].mean()
                if self.mode == "aci":
                    self.at[c] = self.at[c] + self.gamma * (self.alphas[c] - err)
        self.hist.append(self.at.copy())
        if self.mode != "static":
            self.buf.append((S_aud, y_aud))


def smoothed_sets(S, y, cal, alphas, rng, idx=None):
    """Smoothed (randomised) Mondrian conformal sets: include class c iff
    p_c(x) = (#{s_i > s} + U (#{s_i = s} + 1)) / (n_c + 1) > alpha_c.
    Exact class-conditional coverage 1 - alpha_c under exchangeability."""
    N = S.shape[0]
    C = np.zeros((N, 2), bool)
    for c in (0, 1):
        sc = np.sort(S[cal][y[cal] == c, c]); n = len(sc)
        s = S[:, c]
        gt = n - np.searchsorted(sc, s, side="right")
        eq = np.searchsorted(sc, s, side="right") - np.searchsorted(sc, s, side="left")
        U = rng.random(N)
        pv = (gt + U * (eq + 1)) / (n + 1)
        C[:, c] = pv > alphas[c]
    return C


def smoothed_sets_from_ref(S, refS, refy, alphas, rng):
    """Same as smoothed_sets but with an explicit reference (calibration) pool."""
    N = S.shape[0]
    C = np.zeros((N, 2), bool)
    for c in (0, 1):
        sc = np.sort(refS[refy == c, c]); n = len(sc)
        if alphas[c] <= 0:
            C[:, c] = True; continue
        s = S[:, c]
        gt = n - np.searchsorted(sc, s, side="right")
        eq = np.searchsorted(sc, s, side="right") - np.searchsorted(sc, s, side="left")
        pv = (gt + rng.random(N) * (eq + 1)) / (n + 1)
        C[:, c] = pv > alphas[c]
    return C


def pac_quantile(s, alpha, delta):
    """Training-conditional (PAC) conformal quantile (Vovk, 2012): with
    k-th order statistic, coverage ~ Beta(k, n+1-k). Choose the smallest k with
    Pr[coverage < 1-alpha] <= delta, so that with prob >= 1-delta over the
    calibration draw the conditional miss rate is <= alpha."""
    from scipy.stats import beta as B
    n = len(s)
    ss = np.sort(s)
    ks = np.arange(1, n + 1)
    ok = B.cdf(1 - alpha, ks, n + 1 - ks) <= delta
    if not ok.any():
        return np.inf
    return ss[ks[ok][0] - 1]


def pac_sets(S, y, cal, alphas, delta=0.1):
    q = np.array([pac_quantile(S[cal][y[cal] == c, c], alphas[c], delta) for c in (0, 1)])
    return sets_from_q(S, q)
