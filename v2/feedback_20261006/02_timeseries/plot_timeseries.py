#!/usr/bin/env python3
"""
그림 F1 — 대표 descriptor 8개 + energy의 100 step 블록 평균 (facet별 3패널).

입력 : block_means.csv
출력 : fig_F1_<계열>.png · .pdf  9개씩. PNG · PDF가 모두 있으면 건너뛴다.
energy = 구조마다 마지막 블록 평균을 뺀 값(meV/atom). α 내부는 같은 m의 α 표면과 같은 y축(YLIM).
"""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import scienceplots  # noqa: F401  (style 등록용)

HERE = Path(__file__).resolve().parent
SRC = HERE / "block_means.csv"

SERIES = ["S_surf_avg", "S_sub_avg", "A_tb", "G_star",
          *[f"a_in_{r}_m{m}" for r in ("surf", "bulk") for m in (1, 2)], "energy_meV_atom"]
FACETS = ("100", "110", "111")
O_ORDER = ("Opoor", "stoich", "Orich")
O_LAB = {"Opoor": "O-poor", "stoich": "stoich", "Orich": "O-rich"}
O_COL = {"Opoor": "#f0a070", "stoich": "#d95926", "Orich": "#8f3510"}   # plot_v2_report.py 와 같은 색
X_MK = {0.25: "o", 0.50: "s", 0.75: "^"}
INK2, GRID, LOW_FILL = "#52514e", "#e4e3df", "#c8c7c3"
LOW_VALID = 0.5
# α 내부는 표면과 같은 y축으로 그린다 (0 근처가 표면과 비교되게). 값 = 표면 그림의 자동 범위.
YLIM = {"a_in_bulk_m1": (-0.352, 0.556), "a_in_bulk_m2": (-0.482, 0.509)}
ADAPTIVE_END, WIN_START = 100, 601

plt.style.use(["science", "notebook"])
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Pretendard', 'Liberation Sans', 'DejaVu Sans', 'Arial'],
    'mathtext.fontset': 'dejavusans',
    'text.usetex': False,
    'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 11.5,
    'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
    'legend.fontsize': 9.5, 'legend.frameon': False,
    'lines.linewidth': 1.0,
    'figure.dpi': 100, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


def label(s):
    """그림용 이름. 예: a_in_surf_m1 -> α_in surface, m = 1."""
    fixed = {"S_surf_avg": r"$S_\mathrm{surf}$ (avg)", "S_sub_avg": r"$S_\mathrm{sub}$ (avg)",
             "A_tb": r"$A_\mathrm{tb}$ (signed)", "G_star": r"$G^*$",
             "energy_meV_atom": r"$\Delta E$ per atom (meV), relative to last block"}
    if s in fixed:
        return fixed[s]
    _, _, r, m = s.split("_")
    return rf"$\alpha_\mathrm{{in}}$ {'surface' if r == 'surf' else r}, m = {m[1]}"


def style_legend():
    h = [Line2D([], [], color=O_COL[o], lw=1.6, label=O_LAB[o]) for o in O_ORDER]
    return h + [Line2D([], [], color=INK2, marker=X_MK[x], ls="", ms=5, label=fr"$x_\mathrm{{Ir}}$ = {x:.2f}") for x in X_MK]


# ---------------------------------------------------------------- F1
bm = pd.read_csv(SRC, dtype={"facet": str, "surf_id": str})
bm["x"] = (bm.step_lo + bm.step_hi) / 2
last = bm[bm.series == "energy_meV_atom"].groupby("case_id")["mean"].last()
e = bm.series == "energy_meV_atom"
bm.loc[e, "mean"] = bm.loc[e, "mean"] - bm.loc[e, "case_id"].map(last)
second = (bm.facet == "111") & (bm.o_condition == "stoich") & (bm.surf_id == "S2")

def f1_legend(has_low):
    """범례: 1열 O 조건(색) · 2열 x_Ir(기호) · 3열 이후 그 밖 — plot_F1_grid.py 와 같다."""
    h = style_legend() + [Line2D([], [], color=INK2, ls="--", label="(111) stoich S2")]
    if has_low:
        h += [Line2D([], [], color=INK2, marker="o", mfc=LOW_FILL, ls="", ms=5, label=f"valid fraction < {LOW_VALID}")]
    return h + [Line2D([], [], color=GRID, lw=6, label="adaptive (1–100)"),
                Line2D([], [], color=INK2, lw=0.8, ls=":", label="601 (averaging window start)")]


for s in SERIES:
    if (HERE / f"fig_F1_{s}.png").exists() and (HERE / f"fig_F1_{s}.pdf").exists():
        print(f"exists, skip fig_F1_{s}")
        continue
    d = bm[bm.series == s]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    for ax, f in zip(axes, FACETS):
        ax.axvspan(0, ADAPTIVE_END, color=GRID, lw=0, zorder=0)
        ax.axvline(WIN_START - 0.5, color=INK2, lw=0.8, ls=":", zorder=1)
        if s.startswith("a_in") or s in ("S_surf_avg", "S_sub_avg", "A_tb"):
            ax.axhline(0, color=INK2, lw=0.6, zorder=1)
        df = d[d.facet == f]
        n_none = 0
        for cid, g in df.groupby("case_id", sort=False):
            g = g.sort_values("block")
            if g["mean"].isna().all():
                n_none += 1
                continue
            col, mk = O_COL[g.o_condition.iloc[0]], X_MK[g.x_Ir_nominal.iloc[0]]
            ax.plot(g.x, g["mean"], color=col, ls="--" if second[g.index].iloc[0] else "-", zorder=3)
            low = g.valid_fraction < LOW_VALID
            ax.scatter(g.x[~low], g["mean"][~low], marker=mk, s=16, color=col, edgecolor="white", lw=0.4, zorder=4)
            ax.scatter(g.x[low], g["mean"][low], marker=mk, s=16, facecolor=LOW_FILL, edgecolor=col, lw=0.9, zorder=4)
        title = f"({f})  n = {df.case_id.nunique() - n_none}"
        ax.set_title(title + (f"  ({n_none} undefined)" if n_none else ""))
        ax.set_xlim(0, 1100)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_xlabel("MC step")
    axes[0].set_ylabel("block mean (100 steps)")
    if s in YLIM:
        axes[0].set_ylim(YLIM[s])
    fig.suptitle(label(s), x=0.125, ha="left", y=1.02, fontsize=12.5)
    legend = f1_legend((d.valid_fraction < LOW_VALID).any())
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=-(-len(legend) // 3))
    fig.savefig(HERE / f"fig_F1_{s}.png")
    fig.savefig(HERE / f"fig_F1_{s}.pdf")
    plt.close(fig)
    print(f"fig_F1_{s}.png")
