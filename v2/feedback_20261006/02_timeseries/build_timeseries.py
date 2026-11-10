#!/usr/bin/env python3
"""
대표 descriptor 시계열 — 100 step 블록 평균과 601–1100 평균의 오차(block_se).

입력 : ../../results/{per_frame,sro_long}.csv
출력 : block_means.csv   구조 × 계열 × 블록: 점유 가중 평균 · SD · valid fraction
       block_se.csv      구조 × 계열(energy 제외): 601–1100 평균 m_601 · block_se · n_blocks · n_changes · no_change

계열 : S_surf_avg · S_sub_avg · A_tb · G_star · a_in_{TS,BS,surf,bulk}_m{1,2} · energy_meV_atom
       a_in_surf = TS · BS 층을 한 region으로 묶은 α_in = 1 − Σ|E|f_obs / Σ|E|f_0
블록 : 0 = 1–100(adaptive), 1–10 = 101–200 … 1001–1100. 블록과 겹치는 점유 step 수로 가중
오차 : block_se = 블록 6–10(601–1100) 평균의 SD(ddof=1) / √5. 정의된 블록이 5개가 아니면 NA
       n_changes = 601–1100 동안 값이 바뀐 횟수. no_change = 한 번도 안 바뀜 (block_se = 0은 "오차 없음"이 아니라 "잴 정보 없음")
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "results"

KEYS = ["case_id", "facet", "o_condition", "x_Ir_nominal", "surf_id"]
DEPTH = ["S_surf_avg", "S_sub_avg", "A_tb", "G_star"]
A_REG = ("TS", "BS", "bulk")
ALPHA = [f"a_in_{r}_m{m}" for r in ("TS", "BS", "surf", "bulk") for m in (1, 2)]
SERIES = DEPTH + ALPHA + ["energy_meV_atom"]
BLOCKS = [(1, 100)] + [(lo, lo + 99) for lo in range(101, 1101, 100)]
TOL = 1e-9

SE_SERIES = SERIES[:-1]                               # energy 는 오차를 내지 않는다
WIN = (601, 1100)
WIN_BLOCKS = range(6, 11)                             # 601–700 … 1001–1100

OUT = HERE / "block_means.csv"
OUT_SE = HERE / "block_se.csv"
if OUT.exists() or OUT_SE.exists():
    raise SystemExit(f"{HERE} 에 결과가 이미 있다 — 덮어쓰지 않는다")


# ---------------------------------------------------------------- 집계
def step_weight(df, win):
    lo, hi = win
    return (np.minimum(df["step_to"], hi) - np.maximum(df["step_from"], lo) + 1).clip(lower=0)


def wstats(df, value, by, wcol):
    """by 그룹마다 점유 가중 평균 · 표준편차 · valid fraction (NA는 평균에서 제외)."""
    v = df[value].astype(float).to_numpy()
    w = df[wcol].astype(float).to_numpy()
    ok = np.isfinite(v)
    vz = np.where(ok, v, 0.0)
    d = df[by].copy()
    d["_w"], d["_wv"] = w, np.where(ok, w, 0.0)
    d["_wx"], d["_wxx"] = d["_wv"] * vz, d["_wv"] * vz * vz
    g = d.groupby(by, sort=False, observed=True)[["_w", "_wv", "_wx", "_wxx"]].sum().reset_index()
    with np.errstate(invalid="ignore", divide="ignore"):
        g["mean"] = g["_wx"] / g["_wv"]
        g["sd"] = np.sqrt(np.clip(g["_wxx"] / g["_wv"] - g["mean"] ** 2, 0, None))
    g["valid_fraction"] = g["_wv"] / g["_w"]
    return g.drop(columns=["_w", "_wv", "_wx", "_wxx"])


# ---------------------------------------------------------------- 프레임별 값
t0 = time.time()
pf = pd.read_csv(SRC / "per_frame.csv", dtype={"facet": str, "surf_id": str}, low_memory=False,
                 usecols=KEYS + ["frame", "step_from", "step_to", "energy", "n_atoms_actual", *DEPTH])
pf["energy_meV_atom"] = pf["energy"] / pf["n_atoms_actual"] * 1000
print(f"per_frame: {len(pf)} 프레임 · {pf.case_id.nunique()} 구조 ({time.time() - t0:.0f}s)")

parts = []
for chunk in pd.read_csv(SRC / "sro_long.csv", chunksize=500_000,
                         usecols=["case_id", "frame", "quantity", "region", "m", "n_pairs", "f_obs", "f_0", "alpha"]):
    keep = (chunk.quantity == "alpha_in_region") & chunk.region.isin(A_REG) & chunk.m.isin((1, 2))
    parts.append(chunk[keep].drop(columns="quantity"))
sro = pd.concat(parts, ignore_index=True)
assert not sro.duplicated(["case_id", "frame", "region", "m"]).any()

# 저장된 α 가 스펙 식 1 − f_obs/f_0 (f_0 = 0 이면 NA) 와 같은지 — 아래 TS∪BS 계산이 같은 식을 쓰는 근거
calc = np.where(sro.f_0 > 0, 1 - sro.f_obs / sro.f_0.where(sro.f_0 > 0), np.nan)
assert np.array_equal(np.isnan(calc), sro.alpha.isna().to_numpy())
assert np.nanmax(np.abs(calc - sro.alpha)) < TOL

# TS∪BS: α_in(R,m) = 1 − Σ|E| f_obs / Σ|E| f_0. f_0 = 0 인 pure layer 는 분자 · 분모에 0 을 더한다
tb = sro[sro.region.isin(("TS", "BS"))]
assert tb[["n_pairs", "f_obs", "f_0"]].notna().all().all()
tb = tb.assign(num=tb.n_pairs * tb.f_obs, den=tb.n_pairs * tb.f_0).groupby(["case_id", "frame", "m"])[["num", "den"]].sum()
tb["alpha"] = np.where(tb.den > 0, 1 - tb.num / tb.den.where(tb.den > 0), np.nan)
surf = tb.reset_index().assign(region="surf")[["case_id", "frame", "region", "m", "alpha"]]

a = pd.concat([sro[["case_id", "frame", "region", "m", "alpha"]], surf], ignore_index=True)
a["name"] = "a_in_" + a.region + "_m" + a.m.astype(str)
a = a.pivot(index=["case_id", "frame"], columns="name", values="alpha").reset_index()
pf = pf.merge(a, on=["case_id", "frame"], how="left", validate="one_to_one")
missing = pf[ALPHA].isna().all(axis=1).sum()
print(f"sro_long: α 행 {len(sro)}개, α가 하나도 없는 프레임 {missing}개 ({time.time() - t0:.0f}s)")

# ---------------------------------------------------------------- 블록 평균
rows = []
for b, (lo, hi) in enumerate(BLOCKS):
    pf["w"] = step_weight(pf, (lo, hi))
    sub = pf[pf.w > 0]
    cover = sub.groupby("case_id").w.sum()
    if b == 0:      # 첫 수락 전 step(MC 시작 상태)은 accepted_states.xyz 에 없다 — adaptive 블록만 해당
        short = (hi - lo + 1 - cover)[lambda x: x > 0]
        print(f"블록 0: 첫 수락 전 step이 빠진 구조 {len(short)}개 (빠진 step {short.min()}–{short.max()})")
    else:
        assert len(cover) == 30 and (cover == hi - lo + 1).all(), f"블록 {lo}–{hi}: 수락 상태가 모든 step 을 덮지 않음"
    for s in SERIES:
        rows.append(wstats(sub, s, KEYS, "w").assign(series=s, block=b, step_lo=lo, step_hi=hi))
bm = pd.concat(rows, ignore_index=True)[KEYS + ["series", "block", "step_lo", "step_hi", "mean", "sd", "valid_fraction"]]
bm.to_csv(OUT, index=False)

na = bm[bm["mean"].isna()].groupby("series").case_id.nunique()
print(f"블록 평균 {len(bm)}행 → {OUT.name}")
print("값이 없는 블록이 있는 구조 수:", na.to_dict() if len(na) else "없음")

# ---------------------------------------------------------------- 601–1100 평균 · 값이 바뀐 횟수 (1 step 계열)
steps = np.arange(WIN[0], WIN[1] + 1)
rows = []
for cid, g in pf.groupby("case_id"):
    g = g.sort_values("step_from")
    pos = np.searchsorted(g.step_from.to_numpy(), steps, side="right") - 1   # 각 step 을 점유한 프레임
    assert (pos >= 0).all() and (g.step_to.to_numpy()[pos] >= steps).all()
    for s in SE_SERIES:
        x = g[s].to_numpy(float)[pos]
        ok = np.isfinite(x)
        rows.append({**g.iloc[0][KEYS].to_dict(), "series": s,
                     "m_601": x[ok].mean() if ok.any() else np.nan,
                     "n_changes": int((np.diff(x[ok]) != 0).sum())})
win = pd.DataFrame(rows)

# 1 step 계열의 601–1100 평균이 점유 가중 평균과 같은지
w601 = pd.concat([wstats(pf.assign(w=step_weight(pf, WIN))[lambda d: d.w > 0], s, ["case_id"], "w").assign(series=s)
                  for s in SE_SERIES]).set_index(["case_id", "series"])["mean"]
chk = win.set_index(["case_id", "series"]).m_601
assert ((chk - w601.reindex(chk.index)).abs().max() < TOL)

# ---------------------------------------------------------------- block_se
wb = bm[bm.block.isin(WIN_BLOCKS) & bm.series.isin(SE_SERIES)]
assert (wb.step_lo.min(), wb.step_hi.max()) == WIN
gb = wb.groupby(["case_id", "series"])["mean"]
se = pd.DataFrame({"n_blocks": gb.count(), "block_se": gb.std(ddof=1) / np.sqrt(len(WIN_BLOCKS))})
se.loc[se.n_blocks < len(WIN_BLOCKS), "block_se"] = np.nan

out = win.set_index(["case_id", "series"]).join(se, validate="one_to_one").reset_index()
out["no_change"] = out.m_601.notna() & (out.n_changes == 0)      # α 가 아예 정의 안 된 구조는 제외
assert out[(out.n_blocks > 0) & (out.n_blocks < len(WIN_BLOCKS))].no_change.all()   # block_se 가 NA 인 계열(값 있음) = 값이 안 바뀐 계열
out = out[KEYS + ["series", "m_601", "block_se", "n_blocks", "n_changes", "no_change"]]
out.to_csv(OUT_SE, index=False)

print(f"block_se {len(out)}행 → {OUT_SE.name}  (NA {out.block_se.isna().sum()}건, no_change {out.no_change.sum()}건)")
print(f"끝 ({time.time() - t0:.0f}s)")
