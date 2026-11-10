#!/usr/bin/env python3
"""
그림 C — 대표값 2D 지도 3쌍 (행 = facet), 구조 1개 = 점 1개.
  열 1 S_surf(avg) × S_sub(avg) · 열 2 α_in 표면 m1 × m2 · 열 3 α_in 내부 m1 × m2
점 = 601–1100 평균, 오차막대 = block_se. α 두 열은 같은 축 범위.

입력 : ../02_timeseries/block_se.csv
출력 : fig_C_pairmap.png · .pdf. 둘 다 있으면 멈춘다.
"""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots  # noqa: F401  (style 등록용)

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "02_timeseries"
OUT = HERE / "fig_C_pairmap.png"
FACETS = ("100", "110", "111")
O_ORDER = ("Opoor", "stoich", "Orich")
O_LAB = {"Opoor": "O-poor", "stoich": "stoich", "Orich": "O-rich"}
O_COL = {"Opoor": "#f0a070", "stoich": "#d95926", "Orich": "#8f3510"}   # plot_timeseries.py 와 같은 색
X_MK = {0.25: "o", 0.50: "s", 0.75: "^"}
INK2, GRID = "#52514e", "#e4e3df"
A_LIM = (-0.40, 0.45)   # α 두 열 공통 범위
PAIRS = [   # (x 계열, y 계열, 열 제목, x 이름, y 이름, 축 범위)
    ("S_surf_avg", "S_sub_avg", "Depth: surface vs subsurface",
     r"$S_\mathrm{surf}$ (avg)", r"$S_\mathrm{sub}$ (avg)", None),
    ("a_in_surf_m1", "a_in_surf_m2", "In-layer order: surface",
     r"$\alpha_\mathrm{in}$ surface, m = 1", r"$\alpha_\mathrm{in}$ surface, m = 2", A_LIM),
    ("a_in_bulk_m1", "a_in_bulk_m2", "In-layer order: bulk",
     r"$\alpha_\mathrm{in}$ bulk, m = 1", r"$\alpha_\mathrm{in}$ bulk, m = 2", A_LIM),
]

if OUT.exists() and OUT.with_suffix(".pdf").exists():
    raise SystemExit("그림이 이미 있다 — 덮어쓰지 않는다")

plt.style.use(["science", "notebook"])
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Pretendard', 'Liberation Sans', 'DejaVu Sans', 'Arial'],
    'mathtext.fontset': 'dejavusans',
    'text.usetex': False,
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10.5,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'legend.fontsize': 9, 'legend.frameon': False,
    'figure.dpi': 100, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

KEYS = ["case_id", "facet", "o_condition", "x_Ir_nominal", "surf_id"]
se = pd.read_csv(SRC / "block_se.csv", dtype={"facet": str, "surf_id": str})
d = se.pivot(index=KEYS, columns="series", values="m_601").join(
    se.pivot(index=KEYS, columns="series", values="block_se"), rsuffix="_se").reset_index()
assert len(d) == 30
d["second"] = (d.facet == "111") & (d.o_condition == "stoich") & (d.surf_id == "S2")

fig, axes = plt.subplots(3, 3, figsize=(11.5, 10.5), sharex="col", sharey="col", constrained_layout=True)
for j, (xs, ys, title, xl, yl, lim) in enumerate(PAIRS):
    for i, fc in enumerate(FACETS):
        ax = axes[i, j]
        ax.axhline(0, color=GRID, lw=0.9, zorder=0)
        ax.axvline(0, color=GRID, lw=0.9, zorder=0)
        g = d[d.facet == fc]
        ok = g[xs].notna() & g[ys].notna()
        for p in g[ok].itertuples():
            col = O_COL[p.o_condition]
            x, y = getattr(p, xs), getattr(p, ys)
            ax.errorbar(x, y, xerr=getattr(p, f"{xs}_se"), yerr=getattr(p, f"{ys}_se"),
                        fmt="none", ecolor=col, elinewidth=0.8, capsize=0, zorder=2)
            ax.scatter(x, y, marker=X_MK[p.x_Ir_nominal], s=38, facecolor="white" if p.second else col,
                       edgecolor=col, lw=1.0, zorder=3)
        if (~ok).sum():
            ax.text(0.98, 0.03, f"{(~ok).sum()} undefined", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color=INK2)
        if lim:
            ax.set_xlim(lim)
            ax.set_ylim(lim)
        if i == 0:
            ax.set_title(title, fontweight="bold")
        if i == 2:
            ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        if j == 0:
            ax.annotate(f"({fc})", xy=(-0.42, 0.5), xycoords="axes fraction", ha="center", va="center",
                        fontsize=13, fontweight="bold", rotation=90)

handles = [Line2D([], [], color=O_COL[o], marker="o", ls="", ms=6, label=O_LAB[o]) for o in O_ORDER] + \
          [Line2D([], [], color=INK2, marker=X_MK[x], ls="", ms=6, label=fr"$x_\mathrm{{Ir}}$ = {x:.2f}") for x in X_MK] + \
          [Line2D([], [], color=INK2, marker="^", mfc="white", ls="", ms=6, label="(111) stoich S2"),
           Line2D([], [], color=INK2, marker="+", ms=9, mew=0.9, ls="", label="error bar = block SE")]
fig.legend(handles=handles, loc="outside lower center", ncol=3)
fig.suptitle("One point per structure = mean over MC step 601–1100", fontsize=11, color=INK2)
fig.savefig(OUT)
fig.savefig(OUT.with_suffix(".pdf"))
print("saved", OUT.name, OUT.with_suffix(".pdf").name)
