#!/usr/bin/env python3
"""
601–1100 평균의 오차 block_se = 블록 6–10(100 step × 5) 평균의 SD(ddof=1) / √5.

입력 : block_means.csv · halves.csv (build_timeseries.py)
출력 : block_se.csv   구조 × 계열: m_601 · block_se · n_blocks · n_changes · no_change
       정의된 블록이 5개가 아니면 block_se = NA.
       no_change = 601–1100 동안 값이 한 번도 안 바뀜 (block_se = 0은 "오차 없음"이 아니라 "잴 정보 없음")
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
KEYS = ["case_id", "facet", "o_condition", "x_Ir_nominal", "surf_id"]
WIN_BLOCKS = range(6, 11)                                  # 601–700 … 1001–1100
N_BLOCKS = len(WIN_BLOCKS)
TOL = 1e-9

OUT = HERE / "block_se.csv"
if OUT.exists():
    raise SystemExit(f"{OUT} 가 이미 있다 — 덮어쓰지 않는다")

bm = pd.read_csv(HERE / "block_means.csv", dtype={"facet": str, "surf_id": str})
hv = pd.read_csv(HERE / "halves.csv", dtype={"facet": str, "surf_id": str})

win = bm[bm.block.isin(WIN_BLOCKS) & bm.series.isin(hv.series.unique())]
assert (win.step_lo.min(), win.step_hi.max()) == (601, 1100)
g = win.groupby(["case_id", "series"])["mean"]
se = pd.DataFrame({"n_blocks": g.count(), "block_se": g.std(ddof=1) / np.sqrt(N_BLOCKS), "mean_of_blocks": g.mean()})
se.loc[se.n_blocks < N_BLOCKS, "block_se"] = np.nan

out = hv.set_index(["case_id", "series"])[KEYS[1:] + ["m_601", "n_changes"]].join(se, validate="one_to_one").reset_index()
assert len(out) == len(hv) == 360
out["no_change"] = out.m_601.notna() & (out.n_changes == 0)      # α 가 아예 정의 안 된 구조는 제외

# 확인 1: 블록이 빠짐없이 정의된 계열은 블록 평균 5개의 평균 = m_601 (블록마다 100 step 이라 같은 무게)
full = out[out.n_blocks == N_BLOCKS]
vf_full = win.groupby(["case_id", "series"]).valid_fraction.min().eq(1).rename("vf_full")
same = full.join(vf_full, on=["case_id", "series"])
d = (same.mean_of_blocks - same.m_601).abs()
assert d[same.vf_full].max() < TOL
print(f"블록 평균의 평균 = m_601: 모든 블록이 100% 정의된 {same.vf_full.sum()}건 일치. "
      f"α 정의 step 이 빠진 블록이 있는 {(~same.vf_full).sum()}건은 최대 차이 {d[~same.vf_full].max():.3f}")
# 확인 2: block_se 가 NA 인 계열(값 있음)은 모두 값이 안 바뀐 계열
part = out[(out.n_blocks > 0) & (out.n_blocks < N_BLOCKS)]
assert part.no_change.all()

out = out[KEYS + ["series", "m_601", "block_se", "n_blocks", "n_changes", "no_change"]]
out.to_csv(OUT, index=False)

sd_between = hv.groupby("series").m_601.std()
summ = out.groupby("series", sort=False).agg(n_se=("block_se", "count"), median_se=("block_se", "median"),
                                             max_se=("block_se", "max"), n_no_change=("no_change", "sum"))
summ["median_se/sd_between"] = summ.median_se / sd_between.reindex(summ.index)
print(f"{len(out)}행 → {OUT.name}  (block_se NA {out.block_se.isna().sum()}건, no_change {out.no_change.sum()}건)")
print(summ.round(4).to_string())
