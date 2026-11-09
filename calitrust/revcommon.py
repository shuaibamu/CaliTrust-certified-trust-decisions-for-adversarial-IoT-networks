"""Helpers for the revision experiments (consistent targets, cheap comparators,
anchor-selection bias, audit-driven recalibration replay, outcome decomposition)."""
import numpy as np
from . import conformal as cp

AB, AM = 0.10, 0.05


def decompose(C, y, idx):
    """Outcome shares of malicious and benign devices in idx."""
    C = C[idx]; yy = y[idx]
    trust = C[:, 0] & ~C[:, 1]; iso = ~C[:, 0] & C[:, 1]
    both = C[:, 0] & C[:, 1]; empty = ~C[:, 0] & ~C[:, 1]
    out = {}
    for c, nm in ((1, "M"), (0, "B")):
        m = yy == c
        out[nm] = dict(trust=float(trust[m].mean()), isolate=float(iso[m].mean()),
                       attest_both=float(both[m].mean()), attest_empty=float(empty[m].mean()))
    return out


def full_metrics(C, y, idx):
    M = cp.metrics(C, y, idx)
    D = decompose(C, y, idx)
    M["attest_M"] = D["M"]["attest_both"] + D["M"]["attest_empty"]
    M["attest_B"] = D["B"]["attest_both"] + D["B"]["attest_empty"]
    return M


def attest_all(N):
    return np.ones((N, 2), bool)


def activity_brs_rule(brs_m, activity, y, cal, aM=AM, aB=AB, grid=(0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50)):
    """Cheap tuned comparator: devices with activity < a are attested; the rest are
    trusted if BRS maliciousness < tT, isolated if > tI, attested otherwise.
    (a, tT, tI) are tuned on calibration anchors so that the empirical miss rate of
    malicious anchors is <= aM and the empirical isolation rate of benign anchors is
    <= aB, choosing a to minimise the calibration attestation rate. No finite-sample
    guarantee."""
    yc = y[cal]; mc = brs_m[cal]; ac = activity[cal]
    nM = max((yc == 1).sum(), 1); nB = max((yc == 0).sum(), 1)
    best = None
    for a in grid:
        act = ac >= a
        # trust threshold: largest tT with #(malicious, active, m < tT) <= aM * nM
        sm = np.sort(mc[act & (yc == 1)])
        kM = int(np.floor(aM * nM))
        tT = sm[kM] if kM < len(sm) else np.inf
        sb = np.sort(mc[act & (yc == 0)])[::-1]
        kB = int(np.floor(aB * nB))
        tI = sb[kB] if kB < len(sb) else -np.inf
        tI = max(tI, tT)
        trust = act & (mc < tT); iso = act & (mc > tI)
        ver = 1 - (trust | iso).mean()
        if best is None or ver < best[0]:
            best = (ver, a, tT, tI)
    _, a, tT, tI = best
    N = len(y)
    act = activity >= a
    C = np.ones((N, 2), bool)
    tr = act & (brs_m < tT); iso = act & (brs_m > tI)
    C[tr, 1] = False; C[iso, 0] = False
    return C, dict(a=int(a), tT=float(tT), tI=float(tI))


def biased_cal(S_det, y, cal, pool, frac=0.5, rng=None):
    """Incident-response calibration pool: benign anchors as in `cal`; malicious anchors
    (same number as in `cal`) drawn only from the most *detectable* fraction `frac` of
    malicious devices in `pool`, ranked by an incumbent detector score S_det (higher =
    more suspicious)."""
    rng = rng or np.random.default_rng(0)
    nM = int((y[cal] == 1).sum())
    malp = pool[y[pool] == 1]
    order = malp[np.argsort(-S_det[malp])]
    top = order[:max(int(np.ceil(frac * len(order))), nM)]
    pick = rng.choice(top, nM, replace=False)
    return np.concatenate([cal[y[cal] == 0], pick])


def replay_online(S, y, ref_idx, judge_idx, windows=20, audit=0.05, mode="aci", gamma=0.05, K=8,
                  alphas=(AB, AM), seed=0, ref=None):
    """Audit-driven recalibration on fixed (possibly manipulated) evidence. At each
    window, sets are built from the reference pool (initial anchors + rolling uniform
    audits) with adaptive levels; a uniform audit sample of judge_idx is verified and
    appended. Returns per-window miss (Pr[C={B}|M]), coverage-miss and verify,
    and alpha_M trajectory."""
    rng = np.random.default_rng(99 + seed)
    S0, y0 = ref if ref is not None else (S[ref_idx], y[ref_idx])
    oc = cp.OnlineCaliTrust(S0, y0, alphas, mode=mode, gamma=gamma, K=K)
    miss, cmiss, ver, aM = [], [], [], []
    for t in range(windows):
        if mode == "static":
            refS, refy = oc.init
        else:
            refS = np.vstack([b[0] for b in oc.buf[-oc.K:]]); refy = np.concatenate([b[1] for b in oc.buf[-oc.K:]])
        a = oc.at if mode == "aci" else oc.alphas
        C = cp.smoothed_sets_from_ref(S, refS, refy, np.clip(a, -1, 1.0), rng)
        M = cp.metrics(C, y, judge_idx)
        miss.append(M["miss_M"]); cmiss.append(1 - M["cov_M"]); ver.append(M["verify"]); aM.append(float(oc.at[1]))
        aud = rng.choice(judge_idx, max(int(audit * len(judge_idx)), 1), replace=False)
        oc.update(S[aud], y[aud], C[aud])
    return dict(miss=miss, cmiss=cmiss, ver=ver, alphaM=aM)
