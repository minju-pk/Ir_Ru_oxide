#!/usr/bin/env python3
"""
그림 B1 — S_surf(avg) × S_sub(avg) 지도, 구조 1개 = 점 1개.
점 = 601–1100 평균, 오차막대 = block_se, 꼬리 = 101–200 블록 평균에서 출발.

입력 : ../02_timeseries/block_se.csv · block_means.csv
출력 : fig_B1.png · .pdf. 둘 다 있으면 멈춘다.
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
FACETS = ("100", "110", "111")
O_ORDER = ("Opoor", "stoich", "Orich")
O_LAB = {"Opoor": "O-poor", "stoich": "stoich", "Orich": "O-rich"}
O_COL = {"Opoor": "#f0a070", "stoich": "#d95926", "Orich": "#8f3510"}   # plot_timeseries.py 와 같은 색
X_MK = {0.25: "o", 0.50: "s", 0.75: "^"}
INK2, GRID = "#52514e", "#e4e3df"
TAIL_BLOCK = 1   # 101–200
XS, X_LAB = "S_surf_avg", r"$S_\mathrm{surf}$ (avg)"
YS, Y_LAB = "S_sub_avg", r"$S_\mathrm{sub}$ (avg)"
OUT = HERE / "fig_B1.png"

if OUT.exists() and OUT.with_suffix(".pdf").exists():
    raise SystemExit("그림이 이미 있다 — 덮어쓰지 않는다")

plt.style.use(["science", "notebook"])
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Pretendard', 'Liberation Sans', 'DejaVu Sans', 'Arial'],
    'mathtext.fontset': 'dejavusans',
    'text.usetex': False,
    'font.size': 11, 'axes.labelsize': 11.5, 'axes.titlesize': 12,
    'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
    'legend.fontsize': 9.5, 'legend.frameon': False,
    'figure.dpi': 100, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

KEYS = ["case_id", "facet", "o_condition", "x_Ir_nominal", "surf_id"]
se = pd.read_csv(SRC / "block_se.csv", dtype={"facet": str, "surf_id": str})
bm = pd.read_csv(SRC / "block_means.csv", dtype={"facet": str, "surf_id": str})
d = se.pivot(index=KEYS, columns="series", values="m_601").join(
    se.pivot(index=KEYS, columns="series", values="block_se"), rsuffix="_se").join(
    bm[bm.block == TAIL_BLOCK].pivot(index=KEYS, columns="series", values="mean"), rsuffix="_b1").reset_index()
assert len(d) == 30
d["second"] = (d.facet == "111") & (d.o_condition == "stoich") & (d.surf_id == "S2")

fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharex=True, sharey=True, constrained_layout=True)
for ax, fc in zip(axes, FACETS):
    ax.axhline(0, color=GRID, lw=0.9, zorder=0)
    ax.axvline(0, color=GRID, lw=0.9, zorder=0)
    g = d[d.facet == fc]
    n_na = g[YS].isna().sum()
    for p in g[g[YS].notna()].itertuples():
        col = O_COL[p.o_condition]
        x, y = getattr(p, XS), getattr(p, YS)
        x0, y0 = getattr(p, f"{XS}_b1"), getattr(p, f"{YS}_b1")
        ax.plot([x0, x], [y0, y], color=col, lw=0.9, alpha=0.7, zorder=1)
        ax.scatter(x0, y0, s=6, color=col, alpha=0.7, lw=0, zorder=1)
        ax.errorbar(x, y, xerr=getattr(p, f"{XS}_se"), yerr=getattr(p, f"{YS}_se"),
                    fmt="none", ecolor=col, elinewidth=0.8, capsize=0, zorder=2)
        ax.scatter(x, y, marker=X_MK[p.x_Ir_nominal], s=42, facecolor="white" if p.second else col,
                   edgecolor=col, lw=1.0, zorder=3)
    ax.set_title(f"({fc})" + (f"   [{n_na} undefined]" if n_na else ""))
    ax.set_xlabel(X_LAB)
axes[0].set_ylabel(Y_LAB)

# ncol=3 은 열 방향으로 채운다 → 열 1 = O 조건, 열 2 = x_Ir, 열 3 = 기타
handles = [Line2D([], [], color=O_COL[o], marker="o", ls="", ms=6, label=O_LAB[o]) for o in O_ORDER] + \
          [Line2D([], [], color=INK2, marker=X_MK[x], ls="", ms=6, label=fr"$x_\mathrm{{Ir}}$ = {x:.2f}") for x in X_MK] + \
          [Line2D([], [], color=INK2, marker="^", mfc="white", ls="", ms=6, label="(111) stoich S2"),
           Line2D([], [], color=INK2, marker="+", ms=9, mew=0.9, ls="", label="error bar = block SE")]
handles.append(Line2D([], [], color=INK2, lw=0.9, marker="o", ms=2.5, markevery=[0],
                      label="tail: from MC 101–200 mean"))
fig.legend(handles=handles, loc="outside lower center", ncol=3)
fig.suptitle("One point per structure = mean over MC step 601–1100", fontsize=10, color=INK2)
fig.savefig(OUT)
fig.savefig(OUT.with_suffix(".pdf"))
plt.close(fig)
print("saved", OUT.name, OUT.with_suffix(".pdf").name)
