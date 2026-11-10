#!/usr/bin/env python3
"""
그림 F1 3×3 판 — plot_timeseries.py와 같은 데이터를 행 = facet, 열 = O 조건으로 배치.

입력 : block_means.csv
출력 : fig_F1grid_<계열>.png · .pdf  9개씩. PNG · PDF가 모두 있으면 건너뛴다.
energy · YLIM은 plot_timeseries.py와 같다.
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
SERIES = ["S_surf_avg", "S_sub_avg", "A_tb", "G_star",
          *[f"a_in_{r}_m{m}" for r in ("surf", "bulk") for m in (1, 2)], "energy_meV_atom"]
FACETS = ("100", "110", "111")
O_ORDER = ("Opoor", "stoich", "Orich")
O_LAB = {"Opoor": "O-poor", "stoich": "stoich", "Orich": "O-rich"}
O_COL = {"Opoor": "#f0a070", "stoich": "#d95926", "Orich": "#8f3510"}   # plot_timeseries.py 와 같은 색
X_MK = {0.25: "o", 0.50: "s", 0.75: "^"}
INK2, GRID, LOW_FILL = "#52514e", "#e4e3df", "#c8c7c3"
LOW_VALID = 0.5
# α 내부는 표면과 같은 y축으로 그린다 (0 근처가 표면과 비교되게). 값 = 표면 그림의 자동 범위.
YLIM = {"a_in_bulk_m1": (-0.352, 0.556), "a_in_bulk_m2": (-0.482, 0.509)}
ADAPTIVE_END, WIN_START = 100, 601

outs = [HERE / f"fig_F1grid_{s}.png" for s in SERIES]

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
    """그림용 이름 (plot_timeseries.py 와 같다). 예: a_in_surf_m1 -> α_in surface, m = 1."""
    fixed = {"S_surf_avg": r"$S_\mathrm{surf}$ (avg)", "S_sub_avg": r"$S_\mathrm{sub}$ (avg)",
             "A_tb": r"$A_\mathrm{tb}$ (signed)", "G_star": r"$G^*$",
             "energy_meV_atom": r"$\Delta E$ per atom (meV), relative to last block"}
    if s in fixed:
        return fixed[s]
    _, _, r, m = s.split("_")
    return rf"$\alpha_\mathrm{{in}}$ {'surface' if r == 'surf' else r}, m = {m[1]}"


bm = pd.read_csv(HERE / "block_means.csv", dtype={"facet": str, "surf_id": str})
bm["x"] = (bm.step_lo + bm.step_hi) / 2
e = bm.series == "energy_meV_atom"
last = bm[e].groupby("case_id")["mean"].last()
bm.loc[e, "mean"] = bm.loc[e, "mean"] - bm.loc[e, "case_id"].map(last)

def legend_for(has_low):
    """범례: 1열 O 조건(색) · 2열 x_Ir(기호) · 3열 이후 그 밖 — plot_timeseries.py F1 과 같다."""
    h = [Line2D([], [], color=O_COL[o], lw=1.6, label=O_LAB[o]) for o in O_ORDER]
    h += [Line2D([], [], color=INK2, marker=X_MK[x], ls="", ms=5, label=fr"$x_\mathrm{{Ir}}$ = {x:.2f}") for x in X_MK]
    h += [Line2D([], [], color=INK2, ls="--", label="(111) stoich S2")]
    if has_low:
        h += [Line2D([], [], color=INK2, marker="o", mfc=LOW_FILL, ls="", ms=5, label=f"valid fraction < {LOW_VALID}")]
    return h + [Line2D([], [], color=GRID, lw=6, label="adaptive (1–100)"),
                Line2D([], [], color=INK2, lw=0.8, ls=":", label="601 (averaging window start)")]

for s, out in zip(SERIES, outs):
    if out.exists() and out.with_suffix(".pdf").exists():
        print("exists, skip", out.stem)
        continue
    d = bm[bm.series == s]
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.6), sharex=True, sharey=True)
    for i, f in enumerate(FACETS):
        for j, o in enumerate(O_ORDER):
            ax = axes[i, j]
            ax.axvspan(0, ADAPTIVE_END, color=GRID, lw=0, zorder=0)
            ax.axvline(WIN_START - 0.5, color=INK2, lw=0.8, ls=":", zorder=1)
            if s != "G_star":
                ax.axhline(0, color=INK2, lw=0.6, zorder=1)
            n_none = 0
            for _, g in d[(d.facet == f) & (d.o_condition == o)].groupby("case_id"):
                g = g.sort_values("block")
                if g["mean"].isna().all():
                    n_none += 1
                    continue
                x_ir, col = g.x_Ir_nominal.iloc[0], O_COL[o]
                ls = "--" if (f == "111" and o == "stoich" and g.surf_id.iloc[0] == "S2") else "-"
                ax.plot(g.x, g["mean"], color=col, ls=ls, lw=1.2, zorder=3)
                low = g.valid_fraction < LOW_VALID
                ax.scatter(g.x[~low], g["mean"][~low], marker=X_MK[x_ir], s=16, color=col, edgecolor="white", lw=0.4, zorder=4)
                ax.scatter(g.x[low], g["mean"][low], marker=X_MK[x_ir], s=16, facecolor=LOW_FILL, edgecolor=col, lw=0.9, zorder=4)
            title = (O_LAB[o] if i == 0 else "") + (f"\n({n_none} undefined)" if (n_none and i == 0) else
                                                     f"({n_none} undefined)" if n_none else "")
            ax.set_title(title, fontsize=11.5 if i == 0 else 9.5)
            ax.yaxis.grid(True, color=GRID, lw=0.6)
            ax.set_xlim(0, 1100)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
            if i == 2:
                ax.set_xlabel("MC step")
        axes[i, 0].set_ylabel(f"({f})\nblock mean (100 steps)")
    if s in YLIM:
        axes[0, 0].set_ylim(YLIM[s])
    fig.suptitle(label(s), x=0.08, ha="left", y=1.0, fontsize=12.5)
    legend = legend_for((d.valid_fraction < LOW_VALID).any())
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=-(-len(legend) // 3))
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(out.name)
