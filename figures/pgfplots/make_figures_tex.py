"""Write figures_main.tex, a standalone wrapper around figs.tex (Figs. 4-15 of the paper).
The three numeric macros used by figs.tex are filled from data/hist_q.json and data/splits.json.
Compile with:  pdflatex figures_main.tex   (run inside figures/pgfplots)."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
hq = json.load(open(os.path.join(HERE, "data", "hist_q.json")))
sp = json.load(open(os.path.join(HERE, "data", "splits.json")))
tex = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2.3cm]{geometry}
\usepackage{amsmath,amssymb,bm}
\usepackage{graphicx,xcolor}
\usepackage{tikz,pgfplots}
\usetikzlibrary{arrows.meta,positioning,fit,calc,backgrounds}
\usepgfplotslibrary{groupplots}
\newcommand{\figsmall}{\fontsize{7}{8.5}\selectfont}
\newcommand{\figbase}{\fontsize{7.5}{9}\selectfont}
\pgfplotsset{compat=1.18, every axis/.append style={font=\figbase, grid=major, grid style={gray!18}, tick label style={font=\figsmall}}}
\definecolor{cCali}{HTML}{1B7837}\definecolor{cMarg}{HTML}{C51B7D}\definecolor{cDaps}{HTML}{E08214}
\definecolor{cEmp}{HTML}{542788}\definecolor{cGCN}{HTML}{2166AC}\definecolor{cMLP}{HTML}{7FBC41}
\definecolor{cSL}{HTML}{B35806}\definecolor{cBRS}{HTML}{878787}\definecolor{cET}{HTML}{D6604D}
\newcommand{\B}{\mathsf{B}}\newcommand{\M}{\mathsf{M}}\newcommand{\C}{\mathcal{C}}
\newcommand{\qM}{%(qM).3f}\newcommand{\thrBonsM}{%(thrB).3f}\newcommand{\splitsymax}{%(ymax).3f}
\begin{document}
\input{figs.tex}
\end{document}
""" % dict(qM=hq["qM"], thrB=hq["thrB_on_sM"], ymax=sp["ymax"] * 1.05)
open(os.path.join(HERE, "figures_main.tex"), "w").write(tex)
print("wrote figures_main.tex")
