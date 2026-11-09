import json, os, sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


def style(frame="open"):
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "axes.spines.top": frame != "open", "axes.spines.right": frame != "open", "mathtext.fontset": "dejavusans",
                         "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

OUT = os.environ.get("CALITRUST_OUT", "results")
OUTFIG = sys.argv[1] if len(sys.argv) > 1 else "fig16_sybil_injection.png"
style("open")
RB={nm:json.load(open(os.path.join(OUT, f"rev_robust_{nm}.json"))) for nm in ["otc","alpha"]}
col={"otc":"#2166AC","alpha":"#E08214"}; lab={"otc":"Bitcoin-OTC","alpha":"Bitcoin-Alpha"}
ls={"refit":"-","frozen":"--"}
ks=[0,1,3,5,10,20]
def series(nm,key,proto,j,base_key=None):
    d=RB[nm][f"{key}_{proto}"]; out=[]
    for k in ks:
        if str(k) in d: out.append(d[str(k)][0][j])
        else: out.append(RB[nm][f"{base_key}_{proto}"][str(k)][0][j])
    return np.array(out)
fig,axs=plt.subplots(1,3,figsize=(7.1,2.45),sharex=True)
xs=np.arange(len(ks))
for nm in ["otc","alpha"]:
    for proto in ["refit","frozen"]:
        axs[0].plot(xs,series(nm,"m0_fresh",proto,0)*100,ls=ls[proto],color=col[nm],marker="o",ms=3,lw=1.2)
        axs[1].plot(xs,series(nm,"m3_est",proto,0,"m3_fresh")*100,ls=ls[proto],color=col[nm],marker="s",ms=3,lw=1.2)
        axs[1].plot(xs,series(nm,"m3_fresh",proto,0)*100,ls=ls[proto],color=col[nm],marker="o",ms=2.5,lw=0.8,alpha=0.55)
        axs[2].plot(xs,series(nm,"m3_est",proto,2,"m3_fresh")*100,ls=ls[proto],color=col[nm],marker="s",ms=3,lw=1.2)
for a,t in zip(axs,["No admission control:\nfresh sybils","Admission control ($m=3$): established\n(squares) and fresh (circles) sybils","Admission control ($m=3$): honest\nusers isolated, established sybils"]):
    a.set_title(t,loc="left",fontsize=7.5); a.set_xticks(xs); a.set_xticklabels(ks); a.set_xlabel("Sybil accounts per attacker, $k$"); a.margins(x=0.05)
axs[0].set_ylabel("Malicious miss rate (%)"); axs[1].set_ylabel("Malicious miss rate (%)"); axs[2].set_ylabel("Benign isolation rate (%)")
for a in axs[:2]: a.axhline(5,color="#555555",lw=0.7,ls=":"); 
axs[2].axhline(10,color="#555555",lw=0.7,ls=":")
axs[0].text(5.1,6.5,"target 5%",fontsize=6,ha="right",color="#555555"); axs[2].text(5.1,11.5,"budget 10%",fontsize=6,ha="right",color="#555555")
axs[0].set_ylim(0,80); axs[1].set_ylim(0,35); axs[2].set_ylim(0,80)
from matplotlib.lines import Line2D
h=[Line2D([],[],color=col["otc"],lw=1.5),Line2D([],[],color=col["alpha"],lw=1.5),Line2D([],[],color="#333",ls="-",lw=1.2),Line2D([],[],color="#333",ls="--",lw=1.2)]
axs[2].legend(h,["Bitcoin-OTC","Bitcoin-Alpha","standardisation refitted","standardisation frozen"],loc="upper left",fontsize=6,frameon=False)
fig.tight_layout(w_pad=1.0)
for t_ in axs[2].texts:
    if "budget" in t_.get_text(): t_.set_position((5.1, 6.0))
fig.savefig(OUTFIG, dpi=300)
