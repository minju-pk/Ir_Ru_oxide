#!/usr/bin/env python3
"""
figures_v2_report/ 의 그림 17장에 찍힌 점을 표 두 개로 내보낸다 (그림은 다시 그리지 않는다).

입력 : ../descriptors_v2_domain_6/{per_frame,layer_profile,sro_long}.csv   (읽기만 한다)
출력 : figure_data_v2_report/figure_data_scalar.csv   x축 = O 조건인 그림 (fig1b, fig2a, fig3/figS3)
       figure_data_v2_report/figure_data_depth.csv    x축 = 층 번호인 그림 (fig1a, fig2b, fig2c)

집계는 plot_v2_report.py 와 같은 코드다 (WIN · step_weight · wstats 를 그대로 옮겼다).
plot_v2_report.py 는 import 하면 그림을 다시 저장하므로 import 하지 않는다.

그림에 없던 열 두 개:
  valid_fraction    = 값이 정의된 점유 스텝 / 전체 점유 스텝 (스펙의 N_valid / N_requested)
  single_atom_share = 정의된 스텝 중 값이 수식상 고정되는 스텝의 비율
                      alpha_in (단일 층 region) : 그 층의 소수 원소가 1개   -> alpha = 0
                      D_a, H_a                  : region 안 원소 a 가 1개   -> D = 1, H = 0
                      alpha_out                 : 두 층 중 하나가 한 원소뿐 -> alpha = 0
                      그 밖의 양은 해당 없음 (빈칸). 고정 여부는 아래 assert 로 데이터에서 확인한다.

random seed: 난수를 쓰지 않는다.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "descriptors_v2_domain_6"
OUT = HERE / "figure_data_v2_report"

WIN = (601, 1100)
REGIONS = ("TS", "BS", "TSS", "BSS", "bulk")
SINGLE_LAYER = ("TS", "BS", "TSS", "BSS")
KEYS = ["case_id", "facet", "o_condition", "x_Ir_nominal", "surf_id"]
TOL = 1e-12

OUT_SCALAR = OUT / "figure_data_scalar.csv"
OUT_DEPTH = OUT / "figure_data_depth.csv"
if OUT_SCALAR.exists() or OUT_DEPTH.exists():
    raise SystemExit(f"{OUT} 에 결과가 이미 있다 — 덮어쓰지 않는다")


def step_weight(df):
    lo, hi = WIN
    return (np.minimum(df["step_to"], hi) - np.maximum(df["step_from"], lo) + 1).clip(lower=0)


def wstats(df, value, by, deg=None):
    """plot_v2_report.wstats 와 같은 집계 + single_atom_share (deg: 고정 스텝 표시 bool 배열)."""
    v = df[value].astype(float).to_numpy()
    w = df["w"].astype(float).to_numpy()
    ok = np.isfinite(v)
    vz = np.where(ok, v, 0.0)
    d = df[by].copy()
    d["_w"], d["_wv"] = w, np.where(ok, w, 0.0)
    d["_wx"], d["_wxx"] = d["_wv"] * vz, d["_wv"] * vz * vz
    d["_wd"] = np.where(ok & deg, w, 0.0) if deg is not None else 0.0
    g = d.groupby(by, sort=False, observed=True)[["_w", "_wv", "_wx", "_wxx", "_wd"]].sum().reset_index()
    with np.errstate(invalid="ignore", divide="ignore"):
        g["mean"] = g["_wx"] / g["_wv"]
        g["sd"] = np.sqrt(np.clip(g["_wxx"] / g["_wv"] - g["mean"] ** 2, 0, None))
        g["single_atom_share"] = g["_wd"] / g["_wv"] if deg is not None else np.nan
    g["valid_fraction"] = g["_wv"] / g["_w"]
    return g.drop(columns=["_w", "_wv", "_wx", "_wxx", "_wd"])


def check_fixed(values, deg, target, label):
    """고정 스텝에서 값이 정말 target 인지 확인."""
    x = values[deg]
    assert np.all(np.abs(x - target) < TOL), f"{label}: 고정 규칙 위반 (최대 편차 {np.abs(x - target).max()})"
    print(f"  고정 확인 {label:<12} {int(deg.sum()):>8} 프레임행, 전부 {target}")


t0 = time.time()
OUT.mkdir(exist_ok=True)

DOMAIN_Q = [f"{q}_{e}_{r}" for r in REGIONS for e in ("Ir", "Ru")
            for q in ("D", "ncl", "H", "Pa", "Pb", "Pany")] + [f"I_{r}" for r in REGIONS]
DEPTH_Q = ["S_surf_avg", "S_sub_avg", "A_tb", "A_tb_abs", "G_star"]
pf = pd.read_csv(SRC / "per_frame.csv", dtype={"facet": str, "surf_id": str}, low_memory=False,
                 usecols=KEYS + ["frame", "step_from", "step_to", "n_atomic_layers", *DEPTH_Q, *DOMAIN_Q]
                 + [f"x_{r}" for r in REGIONS] + [f"n_{r}" for r in REGIONS])
pf["w"] = step_weight(pf)
pf = pf[pf["w"] > 0].reset_index(drop=True)
for r in REGIONS:                                   # region 안 원소별 개수
    pf[f"nIr_{r}"] = (pf[f"x_{r}"] * pf[f"n_{r}"]).round()
    pf[f"nRu_{r}"] = pf[f"n_{r}"] - pf[f"nIr_{r}"]
print(f"per_frame: 평형창 안 프레임 {len(pf)}개 ({time.time() - t0:.0f}s)")
frame_keys = pf[KEYS + ["frame", "w", "n_atomic_layers"]]

scalar = []


def add_scalar(S, figure, quantity, region="", m=np.nan, shell_r=np.nan):
    assert S.case_id.nunique() == 30, quantity
    S = S.assign(figure=figure, quantity=quantity, region=region)
    if "m" not in S:
        S["m"] = m
    if "shell_r" not in S:
        S["shell_r"] = shell_r
    scalar.append(S)


# ---------------------------------------------------------------- fig1b
for q in DEPTH_Q:
    add_scalar(wstats(pf, q, KEYS), "fig1b_depth_scalars", q)

# ---------------------------------------------------------------- fig3 / figS3
print("\n[domain] 고정 규칙 확인")
for r in REGIONS:
    prefix = "fig3" if r in ("TS", "BS", "bulk") else "figS3"
    for e in ("Ir", "Ru"):
        one = (pf[f"n{e}_{r}"] == 1).to_numpy()
        check_fixed(pf[f"D_{e}_{r}"].to_numpy(), one, 1.0, f"D_{e}_{r}")
        check_fixed(pf[f"H_{e}_{r}"].to_numpy(), one, 0.0, f"H_{e}_{r}")
        for q, name, deg in (("D", f"D_{e}", one), ("ncl", f"n_cluster_{e}", None), ("H", f"H_{e}", one)):
            add_scalar(wstats(pf, f"{q}_{e}_{r}", KEYS, deg), f"{prefix}_domain_{r}_size", name, r)
        for q, lab in (("Pa", "a"), ("Pb", "b"), ("Pany", "any")):
            add_scalar(wstats(pf, f"{q}_{e}_{r}", KEYS), f"{prefix}_domain_{r}_perc", f"P_{e}_{lab}", r)
    add_scalar(wstats(pf, f"I_{r}", KEYS), f"{prefix}_domain_{r}_size", "I_R", r)

# ---------------------------------------------------------------- sro_long (fig2a, 2b, 2c)
print("\n[sro] sro_long.csv 청크로 읽는 중")
parts = []
for chunk in pd.read_csv(SRC / "sro_long.csv", chunksize=500_000,
                         usecols=["case_id", "frame", "quantity", "region", "l", "l2", "dlayer",
                                  "m", "shell_r", "alpha"]):
    keep = (chunk.quantity == "alpha_in_region") | ((chunk.quantity == "alpha_out") & (chunk.dlayer <= 2))
    parts.append(chunk[keep])
sro = pd.concat(parts, ignore_index=True).merge(frame_keys, on=["case_id", "frame"], how="inner")
print(f"  alpha 행 {len(sro)}개 ({time.time() - t0:.0f}s)")

# fig2a — alpha_in(R, m). 단일 층 region 은 소수 원소 1개면 alpha = 0
a_in = sro[sro.quantity == "alpha_in_region"].reset_index(drop=True)
minor = pf[["case_id", "frame"]].copy()
for r in SINGLE_LAYER:
    minor[r] = np.minimum(pf[f"nIr_{r}"], pf[f"nRu_{r}"])
minor = minor.melt(id_vars=["case_id", "frame"], var_name="region", value_name="minor")
a_in = a_in.merge(minor, on=["case_id", "frame", "region"], how="left")   # bulk 는 minor = NaN
deg_in = (a_in["minor"] == 1).to_numpy()
check_fixed(a_in["alpha"].to_numpy(), deg_in, 0.0, "alpha_in")
A_IN = wstats(a_in, "alpha", KEYS + ["region", "m"], deg_in)
A_IN.loc[A_IN.region == "bulk", "single_atom_share"] = np.nan             # 여러 층 region 은 해당 없음
r_in = a_in.groupby(["case_id", "region", "m"]).shell_r.first().rename("shell_r").reset_index()
A_IN = A_IN.merge(r_in, on=["case_id", "region", "m"])
for m in (1, 2):
    for r in REGIONS:
        S = A_IN[(A_IN.region == r) & (A_IN.m == m)].drop(columns="region")
        add_scalar(S, f"fig2a_alpha_in_m{m}", "alpha_in", r)

# fig2b / 2c — alpha_out. 두 층 중 하나가 한 원소뿐이면 alpha = 0
lp = pd.read_csv(SRC / "layer_profile.csv", usecols=["case_id", "frame", "layer", "x"])
lp = lp.merge(frame_keys[["case_id", "frame"]], on=["case_id", "frame"], how="inner")
a_out = sro[(sro.quantity == "alpha_out") & ((sro.dlayer == 1) | (sro.facet == "111"))].reset_index(drop=True)
a_out = (a_out.merge(lp.rename(columns={"layer": "l", "x": "x1"}), on=["case_id", "frame", "l"])
              .merge(lp.rename(columns={"layer": "l2", "x": "x2"}), on=["case_id", "frame", "l2"]))
pure = lambda x: (x == 0) | (x == 1)
deg_out = (pure(a_out.x1) | pure(a_out.x2)).to_numpy()
check_fixed(a_out["alpha"].to_numpy(), deg_out & np.isfinite(a_out["alpha"].to_numpy()), 0.0, "alpha_out")
by_out = KEYS + ["l", "l2", "dlayer", "m", "n_atomic_layers"]
A_OUT = wstats(a_out, "alpha", by_out, deg_out)
A_OUT = A_OUT.merge(a_out.groupby(["case_id", "l", "l2", "m"]).shell_r.first().rename("shell_r").reset_index(),
                    on=["case_id", "l", "l2", "m"])
A_OUT["xpos"] = (A_OUT["l"] + A_OUT["l2"]) / 2 - (A_OUT["n_atomic_layers"] - 1) / 2
A_OUT["figure"] = np.where(A_OUT.dlayer == 1, "fig2b_alpha_out_m" + A_OUT.m.astype(str),
                           "fig2c_alpha_out_111_dlayer2")
A_OUT["quantity"] = "alpha_out"

# fig1a — x_layer
lp = lp.merge(frame_keys, on=["case_id", "frame"], how="inner")
P = wstats(lp, "x", KEYS + ["layer", "n_atomic_layers"])
P["xpos"] = P["layer"] - (P["n_atomic_layers"] - 1) / 2
P = P.rename(columns={"layer": "l"}).assign(l2=np.nan, dlayer=np.nan, m=np.nan, shell_r=np.nan,
                                            figure="fig1a_layer_profile", quantity="x_layer")

# ---------------------------------------------------------------- 쓰기
LEAD = ["figure", "case_id", "facet", "x_Ir_nominal", "o_condition", "surf_id"]
VALS = ["mean", "sd", "valid_fraction", "single_atom_share"]
S = pd.concat(scalar, ignore_index=True)[LEAD + ["region", "quantity", "m", "shell_r"] + VALS]
D = pd.concat([P, A_OUT], ignore_index=True)
D = D.rename(columns={"l": "layer", "l2": "layer2"})[
    LEAD + ["quantity", "layer", "layer2", "dlayer", "xpos", "m", "shell_r"] + VALS]
for T in (S, D):
    T["x_Ir_nominal"] = T["x_Ir_nominal"].map("{:.2f}".format)
S = S.sort_values(["figure", "facet", "x_Ir_nominal", "o_condition", "surf_id", "region", "quantity", "m"])
D = D.sort_values(["figure", "facet", "x_Ir_nominal", "o_condition", "surf_id", "m", "layer", "layer2"])
S.to_csv(OUT_SCALAR, index=False, float_format="%.6g")
D.to_csv(OUT_DEPTH, index=False, float_format="%.6g")
print(f"\n-> {OUT_SCALAR.name}  {len(S)}행 · 그림 {S.figure.nunique()}장")
print(f"-> {OUT_DEPTH.name}   {len(D)}행 · 그림 {D.figure.nunique()}장")
print(f"완료 {time.time() - t0:.0f}s")
