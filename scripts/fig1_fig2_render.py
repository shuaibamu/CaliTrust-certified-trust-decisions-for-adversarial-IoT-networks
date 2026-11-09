"""Raster renderings (600 dpi) of Fig. 1 and Fig. 2 that reproduce the TikZ sources
fig1_siot_structure.tex and fig2_architecture.tex coordinate for coordinate (1 data unit = 1 cm).
Use the .tex files for vector output; use these PNGs where a raster image is needed (Word)."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon, Ellipse

mpl.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "savefig.dpi": 600})
C = dict(OOR="#6A51A3", CLOR="#E08214", CWOR="#4393C3", SOR="#878787", POS="#01665E", NEG="#C51B7D",
         MAL="#262626", B="#01665E", M="#C51B7D", Att="#E6AB02",
         gEvi="#EAF1FB", gEnc="#FDF1E4", gCer="#E8F4EE", gOnl="#FBF6E2")
DASH = {"oor": "-", "clor": (0, (2.2, 1.4)), "cwor": (0, (0.8, 1.0)), "sor": (0, (2.5, 1, 0.6, 1)), "neg": (0, (1.8, 1.1)), "pos": "-"}
LC = {"oor": C["OOR"], "clor": C["CLOR"], "cwor": C["CWOR"], "sor": C["SOR"], "pos": C["POS"], "neg": C["NEG"]}
LW = {"oor": 1.0, "clor": 0.8, "cwor": 1.0, "sor": 0.8, "pos": 0.7, "neg": 0.7}


def canvas(x0, x1, y0, y1):
    fig = plt.figure(figsize=((x1 - x0) / 2.54, (y1 - y0) / 2.54))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def tint(hexc, a):
    c = np.array(mpl.colors.to_rgb(hexc)); return tuple(a * c + (1 - a))


def rbox(ax, x0, y0, x1, y1, fc="white", ec="black", lw=0.5, ls="-", r=0.08, z=1):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def T(ax, x, y, s, size=6.5, ha="center", va="center", **k):
    return ax.text(x, y, s, fontsize=size, ha=ha, va=va, **k)


def arrow(ax, p, q, col="#404040", lw=0.8, ls="-", z=5, hl=3.4, hw=1.7, shrinkA=0, shrinkB=0, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=f"-|>,head_length={hl},head_width={hw}", color=col, lw=lw, linestyle=ls,
                                 shrinkA=shrinkA, shrinkB=shrinkB, connectionstyle=f"arc3,rad={rad}", zorder=z))


def route(ax, pts, col="#595959", lw=0.7, ls=(0, (3, 2)), z=5):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=col, lw=lw, ls=ls, zorder=z, solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1], col=col, lw=lw, ls=ls, z=z)


def cyl(ax, x, y, w, h, fc, ec, lab, size=5.5, z=4):
    e = h * 0.22
    ax.add_patch(Rectangle((x - w / 2, y - h / 2 + e / 2), w, h - e, fc=fc, ec="none", zorder=z))
    for xx in (x - w / 2, x + w / 2):
        ax.plot([xx, xx], [y - h / 2 + e / 2, y + h / 2 - e / 2], color=ec, lw=0.5, zorder=z + 0.1)
    ax.add_patch(Ellipse((x, y - h / 2 + e / 2), w, e, fc=fc, ec=ec, lw=0.5, zorder=z - 0.05))
    ax.add_patch(Ellipse((x, y + h / 2 - e / 2), w, e, fc=fc, ec=ec, lw=0.5, zorder=z + 0.2))
    T(ax, x, y - e * 0.2, lab, size, zorder=z + 0.3, linespacing=1.0)


def check_text(fig, ax):
    r = fig.canvas.get_renderer(); texts = [(t, t.get_window_extent(r)) for t in ax.texts]
    ov = [(a.get_text()[:25], b.get_text()[:25]) for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:] if ba.overlaps(bb)]
    out = [t.get_text()[:25] for t, b in texts if b.x1 > fig.bbox.x1 + 1 or b.x0 < -1 or b.y0 < -1 or b.y1 > fig.bbox.y1 + 1]
    return ov, out


# =========================================================================== Fig. 1
def fig1(path="fig1_siot_structure.png"):
    fig, ax = canvas(-0.08, 10.62, -0.08, 4.68)
    grey = dict(fc=tint("#000000", 0.04), ec=tint("#000000", 0.35), r=0.09, z=0)
    rbox(ax, 0, 2.45, 3.10, 4.60, **grey); rbox(ax, 3.30, 2.45, 6.40, 4.60, **grey); rbox(ax, 0, 0, 3.10, 2.25, **grey)
    rbox(ax, 3.30, 0, 6.40, 2.25, fc=tint(C["NEG"], 0.06), ec=tint(C["NEG"], 0.6), ls=(0, (3, 2)), r=0.09, z=0)
    for x, y, s, col in [(0.06, 4.56, "Owner A", "#595959"), (3.36, 4.56, "Owner B", "#595959"),
                         (0.06, 2.21, "Owner D", "#595959"), (3.36, 2.21, "Owner C (compromised)", "#a3176a")]:
        T(ax, x, y, s, 6, ha="left", va="top", style="italic", color=col)
    dev = {"a1": (0.55, 3.45, "tr", "B"), "a2": (1.50, 4.00, "jd", "?b"), "a3": (2.50, 3.05, "ca", "B"),
           "b1": (3.95, 3.65, "jd", "?b"), "b2": (5.60, 3.20, "jd", "?m"),
           "d1": (0.55, 0.60, "jd", "?b"), "d2": (1.55, 1.45, "ca", "B"), "d3": (2.55, 0.55, "jd", "?b"),
           "c1": (4.00, 1.50, "ca", "M"), "c2": (5.70, 1.35, "jd", "?m"), "c3": (4.85, 0.55, "tr", "M")}
    SHR = {"jd": 5.4, "tr": 5.6, "ca": 6.6}

    def node(x, y, role, lab, z=4):
        mal = lab in ("M", "?m"); fc = C["MAL"] if mal else "white"; tc = "white" if mal else "black"
        if role == "jd":
            ax.add_patch(Circle((x, y), 0.18, fc=fc, ec="black", lw=0.55, zorder=z))
        elif role == "tr":
            ax.add_patch(Rectangle((x - 0.165, y - 0.165), 0.33, 0.33, fc=fc, ec="black", lw=0.55, zorder=z))
        else:
            ax.add_patch(Polygon([(x, y + 0.22), (x + 0.22, y), (x, y - 0.22), (x - 0.22, y)], closed=True, fc=fc, ec="black", lw=0.55, zorder=z))
        if lab:
            T(ax, x, y - 0.005, lab[0], 5.5, color=tc, weight="bold", zorder=z + 1)

    for k, (x, y, r, l) in dev.items():
        node(x, y, r, l)

    def link(a, b, sty, rad=0.0, arr=False):
        xa, ya, ra, _ = dev[a]; xb, yb, rb, _ = dev[b]
        ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-|>,head_length=3.2,head_width=1.6" if arr else "-",
                                     color=LC[sty], lw=LW[sty], linestyle=DASH[sty], connectionstyle=f"arc3,rad={rad}",
                                     shrinkA=SHR[ra] if arr else 0, shrinkB=SHR[rb] if arr else 0, zorder=3 if arr else 2))
    for a, b in [("a1", "a2"), ("a2", "a3"), ("a1", "a3"), ("b1", "b2"), ("d1", "d2"), ("d2", "d3"), ("d1", "d3"),
                 ("c1", "c2"), ("c2", "c3"), ("c1", "c3")]:
        link(a, b, "oor")
    link("a3", "b1", "clor"); link("b2", "c2", "clor"); link("a3", "d2", "cwor"); link("d3", "c3", "cwor")
    link("a2", "b1", "sor", rad=-0.16)
    link("d2", "a1", "pos", arr=True); link("c3", "c1", "pos", rad=0.32, arr=True)
    link("b2", "a3", "neg", rad=0.11, arr=True); link("c2", "d3", "neg", arr=True)
    lx0 = 6.75; tx = lx0 + 0.62
    for y, sty, txt in [(4.40, "oor", "Ownership (OOR)"), (3.96, "clor", "Co-location (C-LOR)"),
                        (3.52, "cwor", "Co-work (C-WOR)"), (3.08, "sor", "Social (SOR)")]:
        ax.plot([lx0, lx0 + 0.5], [y, y], color=LC[sty], lw=LW[sty], linestyle=DASH[sty]); T(ax, tx, y, txt, ha="left")
    for y, sty, txt in [(2.64, "pos", "Positive rating (INT$^{+}$)"), (2.20, "neg", "Negative rating (INT$^{-}$)")]:
        arrow(ax, (lx0, y), (lx0 + 0.5, y), col=LC[sty], lw=0.7, ls=DASH[sty], hl=3.2, hw=1.6); T(ax, tx, y, txt, ha="left")
    node(lx0 + 0.25, 1.76, "tr", ""); T(ax, tx, 1.76, "Training anchor (label used)", ha="left")
    node(lx0 + 0.25, 1.32, "ca", ""); T(ax, tx, 1.32, "Calibration anchor (label used)", ha="left")
    node(lx0 + 0.25, 0.88, "jd", "?b"); T(ax, tx, 0.88, "Judged device (label hidden)", ha="left")
    ax.add_patch(Circle((lx0 + 0.08, 0.44), 0.15, fc="white", ec="black", lw=0.55))
    ax.add_patch(Circle((lx0 + 0.42, 0.44), 0.15, fc=C["MAL"], ec="black", lw=0.55))
    T(ax, tx, 0.44, "Benign / malicious (true class)", ha="left")
    fig.savefig(path); ov = check_text(fig, ax); plt.close(fig); return ov


# =========================================================================== Fig. 2
def fig2(path="fig2_architecture.png"):
    fig, ax = canvas(-0.06, 16.80, -3.40, 1.18)
    G = "#666666"
    rbox(ax, 0.00, -0.90, 8.30, 1.13, fc=C["gEvi"], ec=G, ls=(0, (3, 2)), r=0.14, z=0)
    rbox(ax, 8.40, -0.90, 12.50, 1.13, fc=C["gEnc"], ec=G, ls=(0, (3, 2)), r=0.14, z=0)
    rbox(ax, 12.60, -0.90, 16.72, 1.13, fc=C["gCer"], ec=tint(C["B"], 0.7), ls=(0, (3, 2)), r=0.14, z=0)
    for x, s in [(4.15, "Evidence pipeline"), (10.45, "Score model (any encoder)"), (14.66, "Certification layer")]:
        T(ax, x, 1.10, s, 6.5, va="top", weight="bold", style="italic", color="#4d4d4d")
    cx = {"s1": 1.0, "s2": 3.1, "s3": 5.2, "s4": 7.3, "s5": 9.4, "s6": 11.5, "s7": 13.6, "s8": 15.7}
    for x in cx.values():
        rbox(ax, x - 0.925, -0.80, x + 0.925, 0.80, fc="white", ec="#8c8c8c", lw=0.45, r=0.07, z=1)

    def cap(x, s):
        T(ax, x, -0.42, s, 6.3, linespacing=1.05, zorder=3)
    # S1 network and log
    x = 1.0; P = [(x + a, b) for a, b in [(-0.62, 0.50), (-0.28, 0.66), (0.02, 0.38), (-0.45, 0.17)]]
    ax.plot(*zip(P[0], P[1]), color=C["OOR"], lw=0.8, zorder=2)
    ax.plot(*zip(P[1], P[2]), color=C["CLOR"], lw=0.7, ls=(0, (1.5, 1)), zorder=2)
    ax.plot(*zip(P[0], P[3]), color=C["CWOR"], lw=0.8, ls=(0, (0.6, 0.8)), zorder=2)
    arrow(ax, P[3], P[2], col=C["POS"], lw=0.6, hl=2, hw=1, shrinkA=2, shrinkB=2, z=2)
    for i, p in enumerate(P):
        ax.add_patch(Circle(p, 0.065, fc=C["MAL"] if i == 3 else "white", ec="black", lw=0.4, zorder=3))
    ax.add_patch(Polygon([(x + 0.30, 0.12), (x + 0.30, 0.68), (x + 0.62, 0.68), (x + 0.72, 0.58), (x + 0.72, 0.12)], closed=True,
                         fc="white", ec="#999999", lw=0.4, zorder=2))
    for yy in (0.52, 0.42, 0.32, 0.22):
        ax.plot([x + 0.37, x + 0.65], [yy, yy], color="#b3b3b3", lw=0.35, zorder=3)
    cap(x, "SIoT network and\nrating log")
    # S2 funnel
    x = 3.1
    ax.add_patch(Polygon([(x - 0.5, 0.68), (x + 0.5, 0.68), (x + 0.11, 0.30), (x + 0.11, 0.10), (x - 0.11, 0.10), (x - 0.11, 0.30)],
                         closed=True, fc=tint(C["OOR"], 0.15), ec=tint(C["OOR"], 0.7), lw=0.5, zorder=2))
    T(ax, x, 0.53, r"$n_j\geq m$", 6, zorder=3)
    ax.add_patch(Circle((x + 0.70, 0.52), 0.07, fc="white", ec="#8c8c8c", lw=0.4, ls=(0, (1.5, 1)), zorder=2))
    ax.plot([x + 0.63, x + 0.77], [0.45, 0.59], color=C["NEG"], lw=0.7, zorder=3)
    ax.plot([x + 0.63, x + 0.77], [0.59, 0.45], color=C["NEG"], lw=0.7, zorder=3)
    cap(x, "Evidence admission\ncontrol")
    # S3 feature grid
    x = 5.2; cw = 0.13
    for i in range(22):
        r_, c_ = divmod(i, 11)
        col = tint(C["CWOR"], 0.45) if i < 10 else tint(C["CLOR"], 0.55) if i < 15 else tint(C["POS"], 0.40) if i < 18 else tint(C["OOR"], 0.40)
        ax.add_patch(Rectangle((x - 0.715 + c_ * cw, 0.56 - r_ * cw - cw), cw, cw, fc=col, ec="white", lw=0.3, zorder=2))
    T(ax, x, 0.17, "P   R   T   S", 5.5, color="#4d4d4d", zorder=3)
    cap(x, "Features\n" + r"$\mathbf{x}_i\in\mathbb{R}^{22}$")
    # S4 typed graph
    x = 7.3; g = [(x + a, b) for a, b in [(-0.55, 0.58), (0.0, 0.66), (0.55, 0.55), (-0.35, 0.18), (0.32, 0.16)]]
    for (i, j), col, ls, lw in [((0, 1), C["OOR"], "-", 0.8), ((1, 2), C["CLOR"], (0, (1.5, 1)), 0.7), ((0, 3), C["CWOR"], (0, (0.6, 0.8)), 0.8),
                                ((3, 4), C["SOR"], "-", 0.7), ((1, 4), C["POS"], "-", 0.7), ((2, 4), C["NEG"], (0, (1.2, 0.8)), 0.7)]:
        ax.plot(*zip(g[i], g[j]), color=col, ls=ls, lw=lw, zorder=2)
    for p in g:
        ax.add_patch(Circle(p, 0.06, fc="white", ec="black", lw=0.4, zorder=3))
    cap(x, "Typed graph,\n" + r"$|\mathcal{R}|=6$")
    # S5 encoder
    x = 9.4
    for dy, lab in [(0.0, "layer 1"), (0.20, "layer 2")]:
        ax.add_patch(Rectangle((x - 0.51, 0.06 + dy), 0.85, 0.17, fc="#bfbfbf", ec="none", alpha=0.35, zorder=2))
        ax.add_patch(Rectangle((x - 0.55, 0.10 + dy), 0.85, 0.17, fc=tint(C["CLOR"], 0.20), ec="#a8600e", lw=0.45, zorder=2.5))
        T(ax, x - 0.125, 0.185 + dy, lab, 5.4, zorder=3)
    T(ax, x + 0.58, 0.36, r"$\beta_{i,r}$", 6.5, zorder=3)
    cap(x, "Encoder (RA-GNN,\n" + r"$L=2$)")
    # S6 scores
    x = 11.5; ax.plot([x - 0.62, x + 0.10], [0.10, 0.10], color="#808080", lw=0.4, zorder=2)
    ax.add_patch(Rectangle((x - 0.50, 0.10), 0.20, 0.52, fc=tint(C["B"], 0.55), ec="none", zorder=2))
    ax.add_patch(Rectangle((x - 0.15, 0.10), 0.20, 0.14, fc=tint(C["M"], 0.55), ec="none", zorder=2))
    T(ax, x - 0.40, 0.70, r"$\hat p_{\mathsf{B}}$", 5.4, zorder=3); T(ax, x - 0.05, 0.32, r"$\hat p_{\mathsf{M}}$", 5.4, zorder=3)
    T(ax, x + 0.13, 0.36, r"$1-\hat p_y$", 6, ha="left", zorder=3)
    cap(x, "Scores\n" + r"$s(\mathbf{x},y)$")
    # S7 class-wise calibration pools
    x = 13.6
    cyl(ax, x - 0.38, 0.40, 0.62, 0.52, tint(C["B"], 0.18), "#015049", r"$\mathcal{D}^{\mathsf{B}}_{\mathrm{cal}}$")
    cyl(ax, x + 0.38, 0.40, 0.62, 0.52, tint(C["M"], 0.15), "#9e1662", r"$\mathcal{D}^{\mathsf{M}}_{\mathrm{cal}}$")
    cap(x, "Class-wise calibration\n" + r"$\hat q_{\mathsf{B}},\,\hat q_{\mathsf{M}}$")
    # S8 decision
    x = 15.7
    for y, fc, s in [(0.60, tint(C["B"], 0.22), r"trust: $\mathcal{C}=\{\mathsf{B}\}$"),
                     (0.38, tint(C["Att"], 0.30), r"attest: $\{\mathsf{B},\mathsf{M}\}$ or $\varnothing$"),
                     (0.16, tint(C["M"], 0.18), r"isolate: $\mathcal{C}=\{\mathsf{M}\}$")]:
        rbox(ax, x - 0.81, y - 0.085, x + 0.81, y + 0.085, fc=fc, ec="none", r=0.03, z=2); T(ax, x, y, s, 5.4, zorder=3)
    cap(x, "Decision $a_i$")
    ks = list(cx)
    for a, b in zip(ks[:-1], ks[1:]):
        arrow(ax, (cx[a] + 0.925, 0), (cx[b] - 0.925, 0))
    # Zoom A: admission control
    rbox(ax, 0.00, -3.35, 5.00, -1.35, fc=tint(C["OOR"], 0.05), ec="#8c8c8c", ls=(0, (3, 2)), r=0.14, z=0)
    for p, q in [((cx["s2"] - 0.925, -0.80), (0.00, -1.35)), ((cx["s2"] + 0.925, -0.80), (5.00, -1.35))]:
        ax.plot(*zip(p, q), color="#a6a6a6", lw=0.4, ls=(0, (3, 2)), zorder=1)
    T(ax, 0.08, -1.38, "Evidence admission control (threshold $m$)", 6.5, ha="left", va="top", weight="bold", style="italic", color="#4d4d4d")
    for y, fc, ls, txt, sym, stat, col in [(-1.90, "white", "-", r"honest rater, $n_j=14$", "\u2713", "admitted", "#015049"),
                                           (-2.28, "#262626", "-", r"colluding insider, $n_k=9$", "\u2713", "admitted", "#a87b00"),
                                           (-2.66, "white", (0, (1.5, 1)), r"fresh sybil, $n_s=1<m$", "\u2717", "dropped", "#9e1662")]:
        ax.add_patch(Circle((0.30, y), 0.07, fc=fc, ec="black" if ls == "-" else "#8c8c8c", lw=0.4, ls=ls, zorder=2))
        T(ax, 0.46, y, txt, 6.5, ha="left")
        T(ax, 3.05, y, sym, 6.0, ha="left", color=col, fontfamily="DejaVu Sans")
        T(ax, 3.25, y, stat, 6.5, ha="left", color=col)
    ax.plot([0.15, 4.85], [-2.90, -2.90], color="#b3b3b3", lw=0.35)
    T(ax, 0.15, -3.12, r"statistics $(\boldsymbol{\mu},\boldsymbol{\sigma})$ frozen at calibration", 6, ha="left")
    # Online recalibration loop
    rbox(ax, 5.30, -3.35, 11.00, -1.35, fc=C["gOnl"], ec="#a87b00", ls=(0, (3, 2)), r=0.14, z=0)
    cyl(ax, 6.05, -2.15, 1.25, 0.85, "white", "#8c8c8c", "rolling\npool $\\mathcal{P}$", 6)
    rbox(ax, 8.025 - 0.875, -2.575, 8.025 + 0.875, -1.725, fc="white", ec="#8c8c8c", lw=0.45, r=0.05, z=2)
    T(ax, 8.025, -2.15, "$\\alpha_{c,t+1}=\\alpha_{c,t}$\n$+\\gamma(\\alpha_c-\\mathrm{err}_{c,t})$", 6, zorder=3, linespacing=1.15)
    rbox(ax, 10.20 - 0.65, -2.575, 10.20 + 0.65, -1.725, fc="white", ec="#8c8c8c", lw=0.45, r=0.05, z=2)
    T(ax, 10.20, -2.15, "verification\noracle", 6, zorder=3, linespacing=1.05)
    arrow(ax, (9.55, -2.15), (8.90, -2.15)); T(ax, 9.225, -2.06, r"$\mathrm{err}_{c,t}$", 5.6, va="bottom")
    arrow(ax, (7.15, -2.15), (6.675, -2.15))
    route(ax, [(10.00, -2.575), (10.00, -2.80), (6.05, -2.80), (6.05, -2.575)])
    T(ax, 8.15, -3.31, "Online recalibration from uniform audits $\\mathcal{A}_t$", 6.5, va="bottom", weight="bold", style="italic", color="#4d4d4d")
    # Zoom B: certification internals
    rbox(ax, 11.30, -3.35, 16.72, -1.35, fc=C["gCer"], ec="#8c8c8c", ls=(0, (3, 2)), r=0.14, z=0)
    for p, q in [((cx["s7"] - 0.925, -0.80), (11.30, -1.35)), ((cx["s8"] + 0.925, -0.80), (16.72, -1.35))]:
        ax.plot(*zip(p, q), color="#a6a6a6", lw=0.4, ls=(0, (3, 2)), zorder=1)
    T(ax, 11.38, -1.38, "Class-conditional smoothed calibration", 6.5, ha="left", va="top", weight="bold", style="italic", color="#4d4d4d")
    for x0, x1, fc, lab in [(11.50, 12.35, tint(C["M"], 0.30), "isolate"), (12.35, 13.20, tint(C["Att"], 0.35), "attest"),
                            (13.20, 14.05, tint(C["B"], 0.30), "trust")]:
        ax.add_patch(Rectangle((x0, -2.56), x1 - x0, 0.20, fc=fc, ec="none", zorder=2)); T(ax, (x0 + x1) / 2, -2.24, lab, 5.6)
    arrow(ax, (11.50, -2.56), (14.20, -2.56), col="#4d4d4d", lw=0.45, hl=2.6, hw=1.3)
    for xx, lab in [(12.35, r"$1-\hat q_{\mathsf{B}}$"), (13.20, r"$\hat q_{\mathsf{M}}$")]:
        ax.plot([xx, xx], [-2.53, -2.62], color="#4d4d4d", lw=0.45, zorder=3); T(ax, xx, -2.76, lab, 5.6)
    T(ax, 14.20, -2.86, r"$s(\mathbf{x},\mathsf{M})$", 5.6, ha="right", va="top")
    T(ax, 14.32, -2.02, r"$\pi_c(\mathbf{x})=\dfrac{G_c+U(E_c+1)}{n_c+1}$", 6.2, ha="left")
    T(ax, 14.32, -2.60, r"$\mathcal{C}(\mathbf{x})=\{c:\pi_c>\alpha_{c,t}\}$", 6.2, ha="left")
    T(ax, 14.32, -3.05, r"$\Pr[\mathrm{trust}\mid\mathsf{M}]\leq\alpha_{\mathsf{M}}$", 6.2, ha="left")
    # cross-row routes
    route(ax, [(15.10, -0.80), (15.10, -1.22), (10.20, -1.22), (10.20, -1.725)])
    route(ax, [(6.05, -1.725), (6.05, -1.05), (13.10, -1.05), (13.10, -0.80)])
    fig.savefig(path); ov = check_text(fig, ax)
    # formula right edge vs panel edge
    r = fig.canvas.get_renderer()
    right = max(t.get_window_extent(r).x1 for t in ax.texts if "pi_c" in t.get_text())
    panel_edge = ax.transData.transform((16.72, 0))[0]
    plt.close(fig); return ov, round(panel_edge - right, 1)


if __name__ == "__main__":
    print("fig1", fig1()); print("fig2", fig2())
