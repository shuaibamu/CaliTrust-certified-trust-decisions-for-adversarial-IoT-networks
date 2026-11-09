"""
Relation-aware trust encoder (RA-GNN) in JAX with node-level semantic attention
over relation types, and two baselines sharing the same training loop:
  kind="ragnn" : proposed encoder
  kind="gcn"   : homogeneous GCN (relations collapsed; Guardian-style)
  kind="mlp"   : no message passing
"""
import time
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")


def glorot(key, shape):
    lim = np.sqrt(6.0 / (shape[-2] + shape[-1]))
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim)


def init(key, F, H, R, L=2, kind="ragnn"):
    ks = iter(jax.random.split(key, 64))
    p = {"Win": glorot(next(ks), (F, H)), "bin": jnp.zeros(H),
         "Wout": glorot(next(ks), (H, 2)), "bout": jnp.zeros(2), "layers": []}
    for _ in range(L):
        lay = {"Ws": glorot(next(ks), (H, H)), "b": jnp.zeros(H)}
        if kind == "ragnn":
            lay["Wr"] = jnp.stack([glorot(next(ks), (H, H)) for _ in range(R)])
            lay["Wa"] = glorot(next(ks), (H, 16)); lay["q"] = glorot(next(ks), (16, 1))[:, 0]
        elif kind == "gcn":
            lay["Wn"] = glorot(next(ks), (H, H))
        p["layers"].append(lay)
    return p


def make_forward(src, dst, et, N, R, kind):
    src = jnp.asarray(src); dst = jnp.asarray(dst); et = jnp.asarray(et)
    # per-relation mean normalisation
    cnt = np.zeros((R, N)); np.add.at(cnt, (np.asarray(et), np.asarray(dst)), 1)
    inv = jnp.asarray(1.0 / np.maximum(cnt, 1))                       # (R,N)
    has = jnp.asarray((cnt > 0).astype(np.float32))                   # (R,N)
    deg = np.bincount(np.asarray(dst), minlength=N) + 1.0
    gnorm = jnp.asarray(1.0 / np.sqrt(deg))

    def forward(p, x, key=None, drop=0.0):
        h = jax.nn.relu(x @ p["Win"] + p["bin"])
        att_all = []
        for lay in p["layers"]:
            if key is not None and drop > 0:
                key, k2 = jax.random.split(key)
                h = h * jax.random.bernoulli(k2, 1 - drop, h.shape) / (1 - drop)
            if kind == "ragnn":
                # m_r = mean_{j in N_r(i)} h_j  for every relation r
                seg = et * N + dst
                M = jax.ops.segment_sum(h[src], seg, num_segments=R * N).reshape(R, N, -1)
                M = M * inv[:, :, None]
                Z = jnp.einsum("rnh,rhk->rnk", M, lay["Wr"])          # (R,N,H)
                s = jnp.tanh(Z @ lay["Wa"]) @ lay["q"]                  # (R,N)
                s = jnp.where(has > 0, s, -1e9)
                a = jax.nn.softmax(s, axis=0) * (has.sum(0, keepdims=True) > 0)
                agg = (a[:, :, None] * Z).sum(0)
                att_all.append(a)
                h = jax.nn.relu(h @ lay["Ws"] + agg + lay["b"]) + h
            elif kind == "gcn":
                msg = jax.ops.segment_sum(h[src] * gnorm[src][:, None], dst, num_segments=N)
                agg = (msg + h * gnorm[:, None]) * gnorm[:, None]
                h = jax.nn.relu(h @ lay["Ws"] + agg @ lay["Wn"] + lay["b"])
            else:
                h = jax.nn.relu(h @ lay["Ws"] + lay["b"]) + h
        return h @ p["Wout"] + p["bout"], h, att_all

    return forward


def train(X, y, tr, va, graph, kind="ragnn", H=32, epochs=300, lr=0.01, wd=5e-4,
          drop=0.2, seed=0, patience=60, conftr=0.0, alphas=(0.10, 0.05), temp=0.1):
    """conftr > 0 adds attestation-aware conformal training: each step splits the
    training anchors into pseudo-calibration / pseudo-test halves, computes class-
    conditional conformal thresholds on the pseudo-calibration half, and penalises the
    soft verification rate P(|C| != 1) on the pseudo-test half. Real calibration is
    untouched, so validity is unaffected."""
    src, dst, et, R = graph
    N, F = X.shape
    fwd = make_forward(src, dst, et, N, R, kind)
    key = jax.random.PRNGKey(seed)
    p = init(key, F, H, R, kind=kind)
    x = jnp.asarray(X); yy = jnp.asarray(y)
    trm = jnp.asarray(tr)
    w1 = float((y[tr] == 0).sum()) / max((y[tr] == 1).sum(), 1)
    cw = jnp.array([1.0, w1])

    aB, aM = alphas

    def loss(p, k):
        k1, k2 = jax.random.split(k)
        lo, _, _ = fwd(p, x, k1, drop)
        lp = jax.nn.log_softmax(lo[trm])
        yt = yy[trm]
        ce = -(cw[yt] * lp[jnp.arange(len(tr)), yt]).sum() / cw[yt].sum()
        l2 = sum(jnp.sum(v ** 2) for v in jax.tree_util.tree_leaves(p))
        if conftr <= 0:
            return ce + wd * l2
        pr = jax.nn.softmax(lo[trm], axis=1)
        s = 1.0 - pr                                         # scores (n,2)
        half = jax.random.bernoulli(k2, 0.5, (len(tr),))     # pseudo-calibration mask
        sB = jnp.where(half & (yt == 0), s[:, 0], jnp.nan)
        sM = jnp.where(half & (yt == 1), s[:, 1], jnp.nan)
        qB = jnp.nanquantile(sB, 1 - aB); qM = jnp.nanquantile(sM, 1 - aM)
        inB = jax.nn.sigmoid((qB - s[:, 0]) / temp); inM = jax.nn.sigmoid((qM - s[:, 1]) / temp)
        amb = inB * inM + (1 - inB) * (1 - inM)              # soft |C| != 1
        test = (~half).astype(jnp.float32)
        verif = (amb * test).sum() / jnp.maximum(test.sum(), 1.0)
        return ce + wd * l2 + conftr * verif

    g = jax.jit(jax.value_and_grad(loss))
    pred = jax.jit(lambda p: fwd(p, x)[0])
    m = jax.tree_util.tree_map(jnp.zeros_like, p); v = jax.tree_util.tree_map(jnp.zeros_like, p)
    b1, b2 = 0.9, 0.999
    best, bestp, bad = -1, p, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        key, k = jax.random.split(key)
        l, gr = g(p, k)
        m = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m, gr)
        v = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, v, gr)
        p = jax.tree_util.tree_map(lambda w, a, b: w - lr * (a / (1 - b1 ** ep)) / (jnp.sqrt(b / (1 - b2 ** ep)) + 1e-8), p, m, v)
        if ep % 5 == 0:
            lo = np.asarray(pred(p))
            pr = lo.argmax(1)
            # balanced accuracy on validation
            yv = y[va]
            ba = 0.5 * ((pr[va][yv == 1] == 1).mean() + (pr[va][yv == 0] == 0).mean())
            if ba > best:
                best, bestp, bad = ba, p, 0
            else:
                bad += 5
                if bad >= patience:
                    break
    train_time = time.time() - t0
    return bestp, fwd, train_time


def predict(p, fwd, X):
    lo, h, att = jax.jit(lambda p: fwd(p, jnp.asarray(X)))(p)
    prob = np.asarray(jax.nn.softmax(lo, axis=1))
    return prob, np.asarray(h), [np.asarray(a) for a in att]
