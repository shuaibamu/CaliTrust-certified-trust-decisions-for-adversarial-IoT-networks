"""
Social-IoT (SIoT) network + service-interaction simulator with trust attacks.

Relations follow the SIoT model of Atzori et al. (2012):
  0 OOR  ownership object relationship (same owner)
  1 CLOR co-location object relationship
  2 CWOR co-work object relationship
  3 SOR  social object relationship (owners are friends)
Interactions: consumer i requests a service from provider j, observes a binary
quality outcome, and reports a rating (possibly dishonest).

Attacks on malicious nodes:
  BS  bad service               BM  bad-mouthing (dishonest negative ratings)
  BST ballot-stuffing/collusion OO  on-off (periodic camouflage)
  SB  selective behaviour       MIX each malicious node draws one of the above
"""
import numpy as np
from dataclasses import dataclass, field

N_REL = 4
ATTACKS = ["BS", "BM", "BST", "OO", "SB"]


@dataclass
class Cfg:
    N: int = 1000
    rho: float = 0.20          # malicious fraction
    attack: str = "MIX"
    T: int = 30                # interaction windows
    req: int = 6               # requests per node per window
    noise: float = 0.05        # honest rating flip prob
    weak_frac: float = 0.15    # honest-but-weak devices (hard negatives)
    discovery: float = 0.15    # prob of request outside social graph
    ring: int = 8              # collusion ring size
    seed: int = 0
    # temporal drift: from window drift_at, a fraction of attackers switch
    drift_at: int = -1
    drift_frac: float = 0.6
    drift_to: tuple = ("OO", "SB")
    pool: tuple = ("BS", "BM", "BST", "OO", "SB")   # attacks drawn under MIX


class SIoT:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._build_nodes()
        self._build_graph()
        self._assign_attacks()
        self.log = []   # list of per-window arrays (cons, prov, rel, rating)

    # ------------------------------------------------------------------ nodes
    def _build_nodes(self):
        c, r = self.cfg, self.rng
        N = c.N
        n_own = max(N // 4, 2)
        self.owner = r.integers(0, n_own, N)
        home = r.integers(0, max(N // 25, 2), n_own)
        roam = r.random(N) < 0.35
        self.loc = np.where(roam, r.integers(0, max(N // 25, 2), N), home[self.owner])
        self.work = r.integers(0, max(N // 40, 2), N)
        # malicious: 60% whole compromised owners, 40% random compromised devices
        n_mal = int(round(c.rho * N))
        mal = np.zeros(N, bool)
        owners = r.permutation(n_own)
        target_owner = int(0.6 * n_mal)
        for o in owners:
            if mal.sum() >= target_owner:
                break
            mal[self.owner == o] = True
        rest = np.flatnonzero(~mal)
        need = max(n_mal - mal.sum(), 0)
        mal[r.choice(rest, need, replace=False)] = True
        if mal.sum() > n_mal:  # trim overshoot
            idx = np.flatnonzero(mal)
            mal[r.choice(idx, mal.sum() - n_mal, replace=False)] = False
        self.mal = mal
        self.y = mal.astype(np.int32)
        cap = r.beta(8, 2, N)
        weak = (~mal) & (r.random(N) < c.weak_frac)
        cap[weak] = r.beta(3, 3, weak.sum())
        self.cap = cap
        self.weak = weak

    # ------------------------------------------------------------------ graph
    def _build_graph(self):
        c, r = self.cfg, self.rng
        N = c.N
        edges = {k: set() for k in range(N_REL)}

        def add(k, a, b):
            if a != b:
                edges[k].add((min(a, b), max(a, b)))

        for o in np.unique(self.owner):
            m = np.flatnonzero(self.owner == o)
            for i in range(len(m)):
                for j in range(i + 1, len(m)):
                    add(0, m[i], m[j])
        for key, rel, k in ((self.loc, 1, 4), (self.work, 2, 3)):
            for g in np.unique(key):
                m = np.flatnonzero(key == g)
                if len(m) < 2:
                    continue
                for a in m:
                    for b in r.choice(m, min(k, len(m) - 1), replace=False):
                        add(rel, a, b)
        # owner friendship: Watts-Strogatz ring among owners
        owners = np.unique(self.owner)
        no = len(owners)
        kk = 3
        for i in range(no):
            for d in range(1, kk + 1):
                j = (i + d) % no
                if r.random() < 0.1:
                    j = r.integers(0, no)
                di = np.flatnonzero(self.owner == owners[i])
                dj = np.flatnonzero(self.owner == owners[j])
                if len(di) and len(dj) and r.random() < 0.6:
                    add(3, r.choice(di), r.choice(dj))
        # collusion rings among malicious nodes (fake social ties)
        mal_idx = r.permutation(np.flatnonzero(self.mal))
        self.ring_id = -np.ones(N, int)
        for g, s in enumerate(range(0, len(mal_idx), c.ring)):
            ring = mal_idx[s:s + c.ring]
            self.ring_id[ring] = g
        self.edges = {k: np.array(sorted(v), dtype=np.int64).reshape(-1, 2) for k, v in edges.items()}
        # adjacency lists + relation lookup
        self.nbr = [[] for _ in range(N)]
        self.nbr_rel = [[] for _ in range(N)]
        for k, E in self.edges.items():
            for a, b in E:
                self.nbr[a].append(b); self.nbr_rel[a].append(k)
                self.nbr[b].append(a); self.nbr_rel[b].append(k)
        self.nbr = [np.array(x, int) for x in self.nbr]
        self.nbr_rel = [np.array(x, int) for x in self.nbr_rel]
        deg = np.array([len(x) for x in self.nbr])
        self.deg = deg
        self.influential = deg >= np.quantile(deg, 0.7)

    def _add_ring_edges(self):
        """BST colluders declare fake SOR ties inside their ring."""
        extra = []
        for g in np.unique(self.ring_id[self.ring_id >= 0]):
            m = np.flatnonzero((self.ring_id == g) & np.isin(self.atk, ["BST"]))
            for i in range(len(m)):
                for j in range(i + 1, len(m)):
                    extra.append((m[i], m[j]))
        if extra:
            E = np.vstack([self.edges[3], np.array(extra)])
            self.edges[3] = np.unique(np.sort(E, 1), axis=0)

    # ---------------------------------------------------------------- attacks
    def _assign_attacks(self):
        c, r = self.cfg, self.rng
        atk = np.array(["NONE"] * c.N, dtype=object)
        m = np.flatnonzero(self.mal)
        if c.attack == "MIX":
            atk[m] = r.choice(list(c.pool), len(m))
        else:
            atk[m] = c.attack
        self.atk = atk
        self._add_ring_edges()
        self.on_len = r.integers(3, 7, c.N)
        self.off_len = r.integers(2, 5, c.N)
        self.phase = r.integers(0, 10, c.N)

    def _apply_drift(self, w):
        c = self.cfg
        if c.drift_at >= 0 and w == c.drift_at:
            m = np.flatnonzero(self.mal)
            sw = self.rng.choice(m, int(c.drift_frac * len(m)), replace=False)
            self.atk[sw] = self.rng.choice(list(c.drift_to), len(sw))

    def _is_on(self, i, w):
        L = self.on_len[i] + self.off_len[i]
        return ((w + self.phase[i]) % L) < self.on_len[i]

    # ------------------------------------------------------------ simulation
    def step(self, w):
        c, r = self.cfg, self.rng
        self._apply_drift(w)
        N = c.N
        cons, prov, rel, rat, qual = [], [], [], [], []
        for i in range(N):
            nb = self.nbr[i]
            for _ in range(c.req):
                if len(nb) == 0 or r.random() < c.discovery:
                    j = int(r.integers(0, N))
                    if j == i:
                        continue
                    k = N_REL  # discovery (no social relation)
                else:
                    t = r.integers(0, len(nb))
                    j, k = int(nb[t]), int(self.nbr_rel[i][t])
                q = self._quality(i, j, w)
                cons.append(i); prov.append(j); rel.append(k); qual.append(q)
                rat.append(self._rating(i, j, q))
        # ballot-stuffing fake interactions inside rings
        for i in np.flatnonzero(self.atk == "BST"):
            mates = np.flatnonzero((self.ring_id == self.ring_id[i]) & (self.atk == "BST"))
            mates = mates[mates != i]
            for j in r.choice(mates, min(4, len(mates)), replace=False) if len(mates) else []:
                cons.append(i); prov.append(int(j)); rel.append(3); qual.append(0); rat.append(1)
        arr = (np.array(cons), np.array(prov), np.array(rel), np.array(rat, np.int8), np.array(qual, np.int8))
        self.log.append(arr)
        return arr

    def _quality(self, i, j, w):
        r = self.rng
        a = self.atk[j]
        if a == "NONE":
            p = self.cap[j]
        elif a == "BS":
            p = 0.25
        elif a == "BM":
            p = 0.60
        elif a == "BST":
            p = 0.30
        elif a == "OO":
            p = 0.85 if self._is_on(j, len(self.log)) else 0.15
        elif a == "SB":
            p = 0.90 if self.influential[i] else 0.20
        elif a == "CAM":      # camouflage: mimic weak-but-honest devices
            p = 0.50
        elif a == "SLEEP":    # sleeper: behave like a capable honest device
            p = 0.85
        else:
            p = 0.5
        return int(r.random() < p)

    def _rating(self, i, j, q):
        a = self.atk[i]
        if a == "BM":
            return 0 if not self.mal[j] else 1
        if a == "BST" and self.mal[j] and self.ring_id[j] == self.ring_id[i]:
            return 1
        if a in ("NONE",) or True:
            return int(q ^ (self.rng.random() < self.cfg.noise))

    def run(self, T=None):
        T = T or self.cfg.T
        for w in range(len(self.log), T):
            self.step(w)
        return self
