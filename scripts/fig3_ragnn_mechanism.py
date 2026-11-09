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

OUTFIG = sys.argv[1] if len(sys.argv) > 1 else "fig3_ragnn_mechanism.png"
style("none")
C = dict(oor="#542788", clor="#E08214", cwor="#2166AC", sor="#878787", ip="#2166AC", im="#C51B7D")
fig=plt.figure(figsize=(7.1,3.3)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,14.2); ax.set_ylim(0,6.6); ax.axis("off")
def box(x,y,w,h,txt,fc="white",ec="#333333",fs=7,bold=False,lw=0.9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.08",fc=fc,ec=ec,lw=lw))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,fontweight="bold" if bold else "normal")
def arr(x1,y1,x2,y2,col="#333333",rad=0.0):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=8,color=col,lw=0.9,connectionstyle=f"arc3,rad={rad}"))
rels=[("OOR",C["oor"]),("C-LOR",C["clor"]),("C-WOR",C["cwor"]),("SOR",C["sor"]),("INT$^{+}$",C["ip"]),("INT$^{-}$",C["im"])]
ys=[5.55-0.78*k for k in range(6)]
ax.text(1.05,6.35,"Typed neighbourhoods",ha="center",fontsize=7.5,fontweight="bold")
ax.text(3.55,6.35,"Relation messages",ha="center",fontsize=7.5,fontweight="bold")
ax.text(6.35,6.35,"Node-level attention",ha="center",fontsize=7.5,fontweight="bold")
ax.text(9.25,6.35,"Fusion with self term",ha="center",fontsize=7.5,fontweight="bold")
ax.text(12.4,6.35,"Output and certification",ha="center",fontsize=7.5,fontweight="bold")
for (nm,col),y in zip(rels,ys):
    box(0.1,y-0.27,1.9,0.54,r"$\mathcal{N}_{"+nm.replace("$","")+r"}(i)$",fc="white",ec=col,lw=1.4)
    box(2.45,y-0.27,2.2,0.54,r"$\mathbf{z}_{i,r}=\mathbf{W}_r\,\overline{\mathbf{h}}_{\mathcal{N}_r(i)}$",ec=col,lw=1.0,fs=6.5)
    arr(2.0,y,2.45,y,col)
    box(5.15,y-0.27,2.4,0.54,r"$e_{i,r}=\mathbf{q}^{\top}\tanh(\mathbf{W}_a\mathbf{z}_{i,r})$",fs=6.5)
    arr(4.65,y,5.15,y)
    arr(7.55,y,8.2,3.6,"#777777")
box(8.2,3.0,2.1,1.2,"softmax over\nnon-empty relations\n"+r"$\rightarrow\ \beta_{i,r}$",fc="#EEF3FA",fs=6.5)
box(8.2,1.2,2.1,1.2,r"$\sum_r \beta_{i,r}\mathbf{z}_{i,r}+\mathbf{W}_s\mathbf{h}_i^{(\ell-1)}$"+"\nReLU, + skip\n"+r"$\rightarrow\ \mathbf{h}_i^{(\ell)}$",fc="#EEF3FA",fs=6.5)
arr(9.25,3.0,9.25,2.4)
# input
box(8.2,4.75,2.1,1.0,r"$\mathbf{x}_i$ (22 evidence"+"\n"+r"features) $\rightarrow\mathbf{h}_i^{(0)}$",fc="#F4F4F4",fs=6.5)
ax.annotate("",xy=(9.25,1.2),xytext=(9.25,0.55),arrowprops=dict(arrowstyle="-",color="#777777",lw=0.9))
ax.text(9.25,0.32,r"repeat for $\ell=1,\dots,L$ (here $L=2$)",ha="center",fontsize=6.5,color="#555555")
arr(10.3,1.8,11.1,1.8)
box(11.1,4.3,2.6,1.1,r"$\hat{\mathbf{p}}(\mathbf{x}_i)=\mathrm{softmax}(\mathbf{W}_o\mathbf{h}_i^{(L)})$",fs=6.5)
box(11.1,2.6,2.6,1.1,r"scores $s(\mathbf{x}_i,y)=1-\hat p_y(\mathbf{x}_i)$",fs=6.5)
box(11.1,0.9,2.6,1.1,"class-conditional\nconformal sets "+r"$\mathcal{C}(\mathbf{x}_i)$",fc="#E6F2E8",ec="#1B7837",fs=6.5,lw=1.3)
ax.add_patch(FancyArrowPatch((10.3,1.8),(11.1,4.85),arrowstyle="-|>",mutation_scale=8,color="#333333",lw=0.9,connectionstyle="arc3,rad=-0.3"))
arr(12.4,4.3,12.4,3.7); arr(12.4,2.6,12.4,2.0)
# self term arrow from input to fusion
ax.add_patch(FancyArrowPatch((10.3,5.25),(10.3,2.2),arrowstyle="-|>",mutation_scale=8,color="#777777",lw=0.9,ls="--",connectionstyle="arc3,rad=-0.35"))

for p_ in list(ax.patches):
    if isinstance(p_,FancyArrowPatch):
        pa=p_._posA_posB
        if pa and (abs(pa[0][0]-10.3)<1e-6 and abs(pa[0][1]-1.8)<1e-6): p_.remove()
for p_ in list(ax.patches):
    if isinstance(p_,FancyArrowPatch) and p_.get_connectionstyle().__class__.__name__=="Arc3" and getattr(p_.get_connectionstyle(),"rad",0)==-0.3: p_.remove()
ax.plot([10.3,10.72,10.72],[1.9,1.9,4.85],color="#333333",lw=0.9)
arr(10.72,4.85,11.1,4.85)
fig.savefig(OUTFIG, dpi=300)
