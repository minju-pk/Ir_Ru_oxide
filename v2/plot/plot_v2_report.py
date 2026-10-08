#!/usr/bin/env python3
"""
v2 descriptor 그림 — 깊이별 조성 · 단거리 ordering · domain morphology.

입력 : ../descriptors_v2_domain_6/{per_frame,layer_profile,sro_long}.csv
출력 : figures_v2_report/<항목 폴더>/*.png (300 dpi) + *.pdf

  fig1a_layer_profile          x_layer 깊이 프로파일 (패널 = facet × x_Ir, 선 = O 조건)
  fig1b_depth_scalars          S_surf(avg) · S_sub(avg) · A_tb · |A_tb| · G*
  fig2a_alpha_in_m1 / _m2      α_in(R, m), region 5개
  fig2b_alpha_out_m1 / _m2     α_out 깊이 프로파일, Δlayer = 1
  fig2c_alpha_out_111_dlayer2  (111) Δlayer = 2 (최근접 cation 결합)
  fig3_domain_{TS,BS,bulk}_{size,perc} · figS3_domain_{TSS,BSS}_{size,perc}

집계 : MC 601–1100 스텝, 점유 스텝 가중 평균. 오차막대·띠 = 같은 가중의 표준편차(스텝 간 요동).
       NA 프레임은 평균에서 뺀다.
스칼라 그림 : 열 = facet, x축 = O 조건, 색 = x_Ir. (111) stoich S2는 속빈 마커(선 없음).
깊이 그림   : x축 = 슬랩 중심 기준 층 번호. (111) stoich S2는 점선.
percolation 방향 a, b = 면내 격자벡터.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import scienceplots  # noqa: F401  (style 등록용)

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "descriptors_v2_domain_6"
OUT = HERE / "figures_v2_report"
# 그림 이름 접두사 -> 항목 폴더
SUBDIRS = (("fig1", "01_Depth_composition"), ("fig2", "02_SRO"),
           ("fig3", "03_Domain_morphology"), ("figS3", "03_Domain_morphology"))

WIN = (601, 1100)                 # 평형 구간 (MC 스텝)

FACETS = ("100", "110", "111")
O_ORDER = ("Opoor", "stoich", "Orich")
XS = (0.25, 0.50, 0.75)
REGIONS = ("TS", "BS", "TSS", "BSS", "bulk")
KEYS = ["case_id", "facet", "o_condition", "x_Ir_nominal", "surf_id"]

O_LAB = {"Opoor": "O-poor", "stoich": "stoich", "Orich": "O-rich"}
FACET_MK = {"100": "o", "110": "s", "111": "^"}
X_COL = {0.25: "#86b6ef", 0.50: "#2a78d6", 0.75: "#104281"}          # x_Ir: 파랑 한 계열 (연 -> 진)
O_COL = {"Opoor": "#f0a070", "stoich": "#d95926", "Orich": "#8f3510"}  # O 조건: 주황 한 계열
INK2, GRID = "#52514e", "#e4e3df"

LINE_LW = 1.0
MARKER_S = 46
NUDGE = 0.13                      # (111) stoich S1/S2 좌우 간격
XPOS = {o: i for i, o in enumerate(O_ORDER)}

plt.style.use(["science", "notebook"])
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Pretendard', 'Liberation Sans', 'DejaVu Sans', 'Arial'],
    'mathtext.fontset': 'dejavusans',
    'text.usetex': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 12.5,
    'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'xtick.minor.size': 3, 'ytick.minor.size': 3,
    'legend.fontsize': 10.5, 'legend.frameon': False,
    'lines.linewidth': LINE_LW,
    'figure.dpi': 100, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# 축 범위 · 눈금 (None = 자동). 스칼라 그림의 x축은 O 조건 3칸 고정
AX_DEPTH = {
    "S_surf_avg": dict(ylim=None, ymajor=0.2, yminor=0.1),
    "S_sub_avg":  dict(ylim=None, ymajor=0.2, yminor=0.1),
    "A_tb":       dict(ylim=None, ymajor=0.05, yminor=None),
    "A_tb_abs":   dict(ylim=None, ymajor=0.05, yminor=None),
    "G_star":     dict(ylim=None, ymajor=0.05, yminor=None),
}
AX_ALPHA_IN = dict(ylim=None, ymajor=None, yminor=None)
AX_FRACTION = dict(ylim=(-0.05, 1.05), ymajor=0.5, yminor=0.25)       # D, H, P
AX_COUNT = dict(ylim=None, ymajor=None, yminor=None)                   # n_cluster
AX_INTERFACE = dict(ylim=None, ymajor=None, yminor=None)               # I_R
AX_LAYER_X = dict(ylim=(-0.03, 1.03), ymajor=0.25, yminor=None)        # fig1a
AX_ALPHA_OUT = dict(ylim=None, ymajor=None, yminor=None)               # fig2b, 2c
XLIM_O = (-0.45, 2.45)


def apply_y(ax, cfg):
    if cfg.get("ylim") is not None:
        ax.set_ylim(*cfg["ylim"])
    if cfg.get("ymajor") is not None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(cfg["ymajor"]))
    if cfg.get("yminor") is not None:
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(cfg["yminor"]))


def step_weight(df):
    """구조가 평형 구간 안에서 점유한 MC 스텝 수."""
    lo, hi = WIN
    return (np.minimum(df["step_to"], hi) - np.maximum(df["step_from"], lo) + 1).clip(lower=0)


def wstats(df, value, by):
    """by 그룹마다 점유 가중 평균 · 표준편차 · valid fraction (NA는 평균에서 제외)."""
    v = df[value].astype(float).to_numpy()
    w = df["w"].astype(float).to_numpy()
    ok = np.isfinite(v)
    vz = np.where(ok, v, 0.0)
    d = df[by].copy()
    d["_w"], d["_wv"] = w, np.where(ok, w, 0.0)
    d["_wx"], d["_wxx"] = d["_wv"] * vz, d["_wv"] * vz * vz
    g = d.groupby(by, sort=False, observed=True)[["_w", "_wv", "_wx", "_wxx"]].sum().reset_index()
    with np.errstate(invalid="ignore", divide="ignore"):
        g["mean"] = g["_wx"] / g["_wv"]
        g["sd"] = np.sqrt(np.clip(g["_wxx"] / g["_wv"] - g["mean"] ** 2, 0, None))
    g["valid"] = g["_wv"] / g["_w"]
    return g.drop(columns=["_w", "_wv", "_wx", "_wxx"])


def save(fig, name):
    out = OUT / next(d for p, d in SUBDIRS if name.startswith(p))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    print(f"  -> {name}.png / .pdf")


def header(fig, top_ax, title, notes=()):
    """그림 제목과 캡션 줄을 맨 위 패널 위에 놓는다."""
    h = fig.get_figheight()
    y1 = top_ax.get_position().y1
    y = y1 + (0.45 + 0.2 * len(notes)) / h
    fig.text(0.125, y, title, fontsize=13.5, ha="left", va="bottom")
    for i, t in enumerate(notes):
        fig.text(0.125, y - (i + 1) * 0.2 / h, t, fontsize=9.5, color=INK2, ha="left", va="bottom")


# ------------------------------------------------------------------ 스칼라 그림 (facet 패널)

def _draw_point(ax, xx, r, color, marker, hollow):
    ax.errorbar(xx, r.mean, yerr=r.sd, fmt="none", ecolor=color, elinewidth=0.9, alpha=0.55, zorder=2)
    ax.scatter(xx, r.mean, s=MARKER_S, marker=marker,
               facecolor="white" if hollow else color,
               edgecolor=color if hollow else "white",
               linewidth=1.2 if hollow else 0.7, zorder=4)


def scalar_row(axes, S, ylabel, zero_line, cfg):
    """한 행 = descriptor 하나. S: KEYS + mean, sd, valid."""
    for ax, facet in zip(axes, FACETS):
        if zero_line:
            ax.axhline(0, color=INK2, lw=0.8, zorder=1)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for x in XS:
            g = S[(S.facet == facet) & (S.x_Ir_nominal == x)]
            path = []
            for o in O_ORDER:
                go = g[g.o_condition == o].sort_values("surf_id")
                for k, r in enumerate(go.itertuples()):
                    if not np.isfinite(r.mean):
                        continue
                    xx = XPOS[o] + (k - (len(go) - 1) / 2) * NUDGE
                    _draw_point(ax, xx, r, X_COL[x], FACET_MK[facet], hollow=k > 0)
                    if k == 0:                       # S2는 선에서 제외
                        path.append((xx, r.mean))
            if len(path) > 1:
                ax.plot(*zip(*path), color=X_COL[x], lw=LINE_LW, zorder=3)
        ax.set_xlim(*XLIM_O)
        ax.set_xticks(range(len(O_ORDER)))
        ax.set_xticklabels([O_LAB[o] for o in O_ORDER])
        ax.tick_params(axis="x", labelbottom=True)   # 모든 행에 x 라벨 표시
        ax.xaxis.set_minor_locator(ticker.NullLocator())
    apply_y(axes[0], cfg)
    axes[0].set_ylabel(ylabel)


def scalar_figure(rows, name, title, col_titles=None, notes=()):
    """rows: [(S, ylabel, zero_line, cfg)] — 행마다 y축 공유, 열 = facet."""
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(10.5, 2.5 * n + 0.4), sharex=True, sharey="row",
                             squeeze=False, gridspec_kw=dict(wspace=0.08, hspace=0.2))
    for i, (S, ylabel, zero_line, cfg) in enumerate(rows):
        scalar_row(axes[i], S, ylabel, zero_line, cfg)
    for j, facet in enumerate(FACETS):
        axes[0, j].set_title(col_titles[j] if col_titles else f"({facet})")
    handles = [Line2D([], [], color=X_COL[x], marker="o", lw=LINE_LW, label=fr"$x_\mathrm{{Ir}}$ = {x:.2f}")
               for x in XS]
    handles += [Line2D([], [], color=INK2, marker="^", mfc="white", ls="", label="open: (111) stoich S2"),
                Line2D([], [], color=INK2, lw=0.9, marker="|", ls="", label="error bar: SD over steps")]
    axes[0, 2].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.03, 1.0))
    header(fig, axes[0, 0], title, notes)
    save(fig, name)


# ------------------------------------------------------------------ 깊이 그림

def depth_figure(rows, name, title, ylabel, cfg, zero_line=False, nominal=False, notes=()):
    """rows: [(row_label, P)] — P: KEYS + xpos, mean, sd. 열 = x_Ir, 선 = O 조건."""
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(12, 2.8 * n + 0.4), sharey=True, sharex="row",
                             squeeze=False, gridspec_kw=dict(wspace=0.08, hspace=0.34))
    for i, (label, P) in enumerate(rows):
        half = float(np.nanmax(np.abs(P["xpos"])))
        for j, x in enumerate(XS):
            ax = axes[i, j]
            if zero_line:
                ax.axhline(0, color=INK2, lw=0.8, zorder=1)
            if nominal:
                ax.axhline(x, color=INK2, lw=0.8, ls=":", zorder=1)
            ax.yaxis.grid(True, color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            g = P[P.x_Ir_nominal == x]
            for o in O_ORDER:
                for k, (_, gc) in enumerate(g[g.o_condition == o].groupby("surf_id", sort=True)):
                    gc = gc.sort_values("xpos")
                    ax.fill_between(gc["xpos"], gc["mean"] - gc["sd"], gc["mean"] + gc["sd"],
                                    color=O_COL[o], alpha=0.09, lw=0, zorder=2)
                    ax.plot(gc["xpos"], gc["mean"], color=O_COL[o], lw=LINE_LW, ls="-" if k == 0 else "--",
                            marker="o", ms=3.2, mec="white", mew=0.4, zorder=3)
            ax.set_xlim(-half - 0.6, half + 0.6)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1 if half < 6 else 5))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1) if half >= 6 else ticker.NullLocator())
            if i == 0:
                ax.set_title(fr"$x_\mathrm{{Ir}}$ = {x:.2f}")
            if i == n - 1 and j == 1:
                ax.set_xlabel("layer index from slab center  (bottom → top)")
        axes[i, 0].set_ylabel(f"{label}\n{ylabel}")
    apply_y(axes[0, 0], cfg)
    handles = [Line2D([], [], color=O_COL[o], lw=LINE_LW, marker="o", ms=3.2, label=O_LAB[o]) for o in O_ORDER]
    handles += [Line2D([], [], color=INK2, lw=LINE_LW, ls="--", label="dashed: (111) stoich S2"),
                Line2D([], [], color=INK2, lw=6, alpha=0.2, label="band: ±SD over steps")]
    if nominal:
        handles.append(Line2D([], [], color=INK2, lw=0.8, ls=":", label=r"nominal $x_\mathrm{Ir}$"))
    axes[0, 2].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.03, 1.0))
    header(fig, axes[0, 0], title, notes)
    save(fig, name)


# ================================================================== 데이터

t0 = time.time()
OUT.mkdir(exist_ok=True)

DOMAIN_COLS = [f"{q}_{e}_{r}" for r in REGIONS for e in ("Ir", "Ru")
               for q in ("D", "ncl", "H", "Pa", "Pb", "Pany")] + [f"I_{r}" for r in REGIONS]
pf = pd.read_csv(SRC / "per_frame.csv", dtype={"facet": str, "surf_id": str},
                 usecols=KEYS + ["frame", "step_from", "step_to", "n_atomic_layers",
                                 *AX_DEPTH.keys(), *DOMAIN_COLS])
pf["w"] = step_weight(pf)
pf = pf[pf["w"] > 0].reset_index(drop=True)
print(f"per_frame: 평형창 안 프레임 {len(pf)}개, 케이스 {pf.case_id.nunique()}개 ({time.time() - t0:.0f}s)")
frame_keys = pf[KEYS + ["frame", "w", "n_atomic_layers"]]


def case_stats(col, by=()):
    S = wstats(pf, col, KEYS + list(by))
    assert S.case_id.nunique() == 30, col
    return S


# ------------------------------------------------------------------ Fig 1 깊이별 조성
print("\n[Fig 1] 깊이별 조성")
lp = pd.read_csv(SRC / "layer_profile.csv", usecols=["case_id", "frame", "layer", "x"])
lp = lp.merge(frame_keys, on=["case_id", "frame"], how="inner")
P = wstats(lp, "x", KEYS + ["layer", "n_atomic_layers"])
P["xpos"] = P["layer"] - (P["n_atomic_layers"] - 1) / 2
depth_figure([(f"({f})", P[P.facet == f]) for f in FACETS], "fig1a_layer_profile",
             "Fig. 1a  Layer composition profile", r"$x_\ell$ (Ir fraction)", AX_LAYER_X, nominal=True,
             notes=["MC steps 601–1100, occupancy-weighted mean; layer 0 of each slab = bottom surface (BS), "
                    "last layer = top surface (TS)"])

depth_rows = [
    (case_stats("S_surf_avg"), r"$S_\mathrm{surf}^\mathrm{(avg)}$", True, AX_DEPTH["S_surf_avg"]),
    (case_stats("S_sub_avg"), r"$S_\mathrm{sub}^\mathrm{(avg)}$", True, AX_DEPTH["S_sub_avg"]),
    (case_stats("A_tb"), r"$A_\mathrm{tb}$", True, AX_DEPTH["A_tb"]),
    (case_stats("A_tb_abs"), r"$|A_\mathrm{tb}|$", False, AX_DEPTH["A_tb_abs"]),
    (case_stats("G_star"), r"$G^{*}$", False, AX_DEPTH["G_star"]),
]
scalar_figure(depth_rows, "fig1b_depth_scalars", "Fig. 1b  Depth-composition scalars",
              notes=["MC steps 601–1100, occupancy-weighted mean ± SD;  "
                     "S > 0: Ir-rich, S < 0: Ru-rich;  A_tb = x_TS − x_BS"])

# ------------------------------------------------------------------ Fig 2 단거리 ordering
print("\n[Fig 2] 단거리 ordering  (sro_long.csv 370 MB, 청크로 읽는다)")
parts = []
for chunk in pd.read_csv(SRC / "sro_long.csv", chunksize=500_000,
                         usecols=["case_id", "frame", "quantity", "region", "l", "l2", "dlayer",
                                  "m", "shell_r", "alpha"]):
    keep = (chunk.quantity == "alpha_in_region") | ((chunk.quantity == "alpha_out") & (chunk.dlayer <= 2))
    parts.append(chunk[keep])
sro = pd.concat(parts, ignore_index=True).merge(frame_keys, on=["case_id", "frame"], how="inner")
print(f"  alpha 행 {len(sro)}개 ({time.time() - t0:.0f}s)")

a_in = sro[sro.quantity == "alpha_in_region"]
A_IN = wstats(a_in, "alpha", KEYS + ["region", "m"])
r_in = a_in.groupby(["facet", "m"]).shell_r.median()
for m in (1, 2):
    cols = [f"({f})  m={m}: {r_in[(f, m)]:.2f} Å" for f in FACETS]
    rows = []
    for region in REGIONS:
        S = A_IN[(A_IN.region == region) & (A_IN.m == m)]
        assert S.case_id.nunique() == 30, (region, m)
        rows.append((S, fr"$\alpha_\mathrm{{in}}$ ({region})", True, AX_ALPHA_IN))
    scalar_figure(rows, f"fig2a_alpha_in_m{m}", f"Fig. 2a  In-layer ordering  α_in(R, m={m})",
                  col_titles=cols,
                  notes=["α = 0: random at fixed layer composition,  α > 0: like-pair clustering,  "
                         "α < 0: Ir–Ru ordering;  shell distance in panel title"])

a_out = sro[sro.quantity == "alpha_out"]
A_OUT = wstats(a_out, "alpha", KEYS + ["l", "l2", "dlayer", "m", "n_atomic_layers"])
A_OUT["xpos"] = (A_OUT["l"] + A_OUT["l2"]) / 2 - (A_OUT["n_atomic_layers"] - 1) / 2
r_out = a_out.groupby(["facet", "dlayer", "m"]).shell_r.median()
for m in (1, 2):
    rows = [(f"({f})  {r_out[(f, 1, m)]:.2f} Å", A_OUT[(A_OUT.facet == f) & (A_OUT.dlayer == 1) & (A_OUT.m == m)])
            for f in FACETS]
    depth_figure(rows, f"fig2b_alpha_out_m{m}", f"Fig. 2b  Inter-layer registry  α_out (Δlayer = 1, m={m})",
                 r"$\alpha_\mathrm{out}$", AX_ALPHA_OUT, zero_line=True,
                 notes=["x = midpoint of the layer pair;  α_out > 0: like species stacked,  "
                        "α_out < 0: Ir–Ru alternation;  Δlayer = 1 is the spec's 'adjacent layer'"])
rows = [(f"(111)  m={m}: {r_out[('111', 2, m)]:.2f} Å",
         A_OUT[(A_OUT.facet == "111") & (A_OUT.dlayer == 2) & (A_OUT.m == m)]) for m in (1, 2)]
depth_figure(rows, "fig2c_alpha_out_111_dlayer2", "Fig. 2c  (111) α_out at Δlayer = 2  (beyond spec)",
             r"$\alpha_\mathrm{out}$", AX_ALPHA_OUT, zero_line=True,
             notes=["(111): the nearest cation–cation bond (3.14 Å) connects layers two apart; "
                    "the spec's 'adjacent layer' (Δlayer = 1) misses it"])

# ------------------------------------------------------------------ Fig 3 domain morphology
print("\n[Fig 3] domain morphology")
DOMAIN_NOTES = ["k = 4 symmetrized in-plane graph fixed at the initial slab; (110): 4th-neighbour tie "
                "included (6 neighbours)",
                "(111): 16 sites per layer — strong finite-size limits on cluster / percolation statistics"]
for region in REGIONS:
    prefix = "fig3" if region in ("TS", "BS", "bulk") else "figS3"
    size_rows = []
    # mathtext 아래첨자 한 번 -> n_cluster는 "cluster,Ir"로 쓴다
    for q, sym, sub, cfg in (("D", "D", "", AX_FRACTION), ("ncl", "n", "cluster,", AX_COUNT),
                             ("H", "H", "", AX_FRACTION)):
        for e in ("Ir", "Ru"):
            size_rows.append((case_stats(f"{q}_{e}_{region}"), fr"${sym}_{{\mathrm{{{sub}{e}}}}}$", False, cfg))
    size_rows.append((case_stats(f"I_{region}"), r"$I_R$", False, AX_INTERFACE))
    scalar_figure(size_rows, f"{prefix}_domain_{region}_size",
                  f"Fig. {prefix[3:]}  Domain size & interface — region {region}",
                  notes=DOMAIN_NOTES + ["I_R = hetero-edge fraction on the domain graph (no composition baseline)"])
    perc_rows = []
    for e in ("Ir", "Ru"):
        for q, lab in (("Pa", "a"), ("Pb", "b"), ("Pany", "any")):
            perc_rows.append((case_stats(f"{q}_{e}_{region}"), fr"$P_{{\mathrm{{{e}}},{lab}}}$", False, AX_FRACTION))
    scalar_figure(perc_rows, f"{prefix}_domain_{region}_perc",
                  f"Fig. {prefix[3:]}  Percolation probability — region {region}",
                  notes=DOMAIN_NOTES + ["P = fraction of MC steps in which an Ir (Ru) domain wraps the cell "
                                        "along lattice vector a / b;  any = a OR b"])

print(f"\n완료 {time.time() - t0:.0f}s  -> {OUT}")
