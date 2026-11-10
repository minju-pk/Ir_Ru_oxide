#!/usr/bin/env python3
"""
α_in / α_out 구현 검증 (화면 출력만).

[1] 껍질 목록      tolerance 규칙(폭 ≤ tol, 껍질 간격 > tol)을 지키고 실측 거리·pair 수와 맞는가
[2] 스펙 범위 필터  넓은 범위로 계산한 결과의 in_spec 행 == 스펙 범위로 직접 계산한 결과
[3] C_lat 항등식   같은 pair 집합에서 (1 − α_in) = (1 − C_lat)(N − 1)/N
[4] 인공 배열      무작위 셔플 ≈ 0, 동종 분리 > 0, 이종 교대 < 0
[5] 결측 규약      pure layer(f_0 = 0) → NA, region α가 가중합 식과 같은가
[6] 외부 대조      MC 산출물의 SRO 표(analysis_*/sro_*_table.csv)와 이종 pair 비율 비교
                   (배치 산출물 OUTDIR이 없으면 SKIP)
"""
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from descriptors_v2.regions import compute_layer_id, get_regions  # noqa: E402
from descriptors_v2.sro import (build_shell_pairs, compute_alpha_frames,  # noqa: E402
                                layer_region_map, SPEC_R_MAX, SPEC_DLAYER_MAX)

MC_DIR = PROJECT_ROOT / "Ir_Ru_oxide" / "correct" / "MC_final_30cases_20260806" / "mc_runs"
OUTDIR = HERE / "results"   # 검증 [6]이 읽는 배치 산출물
CATION = ("Ir", "Ru")
TOL = 0.18
SRO_PARAMS = dict(r_max_inplane=6.6, r_max_inter=6.0, tol=TOL, n_shell=2, dlayer_max=3)
SEED = 42

# 초기 구조 실측값 (x_Ir = 0.50): (facet, dlayer, m) -> (거리 Å, 총 pair 수).
# pair 수는 층 수에 따라 달라지므로 층 수가 같은 케이스만 대조한다.
EXPECT_SHELLS = {
    ("100", 0, 1): (3.154, 350), ("100", 0, 2): (4.532, 350),
    ("100", 1, 1): (3.572, 1260), ("100", 1, 2): (5.714, 1260),
    ("110", 0, 1): (3.154, 392), ("110", 0, 2): (3.572, 784),
    ("110", 1, 1): (3.572, 672), ("110", 1, 2): (4.532, 672),
    ("111", 0, 1): (5.522, 672), ("111", 1, 1): (3.572, 960),
    ("111", 2, 1): (3.154, 304), ("111", 2, 2): (4.532, 608),
}


def load_case(case_dir):
    """초기 구조에서 (atoms, cation index, cation 원소, 층 라벨, 껍질 목록, 층→region)."""
    atoms = read(str(case_dir / "surface.xyz"))
    symbols = np.array(atoms.get_chemical_symbols())
    cat = np.where(np.isin(symbols, CATION))[0]
    layer_id = compute_layer_id(atoms, cat)
    shells = build_shell_pairs(atoms.get_positions()[cat], atoms.cell, atoms.pbc,
                               layer_id, **SRO_PARAMS)
    regions5 = get_regions(atoms, layer_id)
    return atoms, cat, symbols[cat], layer_id, shells, layer_region_map(layer_id, regions5)


def facet_of(name):
    return name.split("_")[4]


CASES = sorted(d for d in MC_DIR.glob("ir_ru_sqs_*") if d.is_dir())


# ---------------------------------------------------------------- [1] 껍질 목록
def check_1_shell_lists():
    """껍질이 tolerance 규칙을 지키고, 실측값을 재현하는가."""
    bad_width, bad_gap, dup_images = [], [], []
    seen_expect = {}

    for case_dir in CASES:
        atoms, cat, _, layer_id, shells, _ = load_case(case_dir)
        facet = facet_of(case_dir.name)

        # (i) 껍질 폭 <= tolerance
        w = shells['shell_w'][np.isfinite(shells['shell_w'])]
        if (w > TOL).any():
            bad_width.append((case_dir.name, float(w.max())))

        # (ii) 같은 목록의 껍질 간격 > tolerance
        for l, l2 in {(a, b) for a, b in zip(shells['l'], shells['l2'])}:
            sel = (shells['l'] == l) & (shells['l2'] == l2)
            r = np.sort(shells['shell_r'][sel][np.isfinite(shells['shell_r'][sel])])
            if len(r) > 1 and np.diff(r).min() <= TOL:
                bad_gap.append((case_dir.name, l, l2, float(np.diff(r).min())))

        # (iii) 같은 (i,j)가 서로 다른 이미지로 두 번 이상 잡힌 건수
        key = shells['pair_i'].astype(np.int64) * (len(cat) + 1) + shells['pair_j']
        _, counts = np.unique(key, return_counts=True)
        if (counts > 1).any():
            dup_images.append((case_dir.name, int((counts - 1).sum())))

        # (iv) 실측값 대조
        for dl in (0, 1, 2):
            for m in (1, 2):
                want = EXPECT_SHELLS.get((facet, dl, m))
                if want is None:
                    continue
                sel = (shells['dlayer'] == dl) & (shells['m'] == m) & (shells['n_pairs'] > 0)
                if not sel.any():
                    continue
                r = shells['shell_r'][sel]
                got = (float(r.mean()), int(shells['n_pairs'][sel].sum()))
                seen_expect.setdefault((facet, dl, m), []).append((case_dir.name, got, want))

    print("[1] 껍질 목록")
    print(f"  껍질 폭 > {TOL} A 인 케이스        : {len(bad_width)}건")
    print(f"  껍질 1-2 간격 <= {TOL} A 인 목록   : {len(bad_gap)}건")
    print(f"  중복 이미지 pair                  : {len(dup_images)}케이스"
          + (f" {dup_images[:3]}" if dup_images else " (0건)"))

    n_match = n_total = 0
    for (facet, dl, m), rows in sorted(seen_expect.items()):
        r_want, n_want = rows[0][2]
        hit = [(name, got) for name, got, _ in rows if got[1] == n_want]
        n_total += 1
        r_got = np.mean([got[0] for _, got, _ in rows])
        ok = bool(hit) and abs(r_got - r_want) < 0.02
        n_match += ok
        print(f"  ({facet}) dlayer={dl} m={m}: r={r_got:.3f} (실측 {r_want:.3f}) · "
              f"pair {n_want} 인 케이스 {len(hit)}/{len(rows)}  {'OK' if ok else '<<< 불일치'}")

    ok = not bad_width and not bad_gap and n_match == n_total
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


# --------------------------------------------------------- [2] 스펙 범위 필터
def check_2_superset_contains_spec():
    """넓은 범위 결과의 in_spec 행이 스펙 범위로 직접 계산한 결과와 같은가 (entry · 거리 · pair 수 · α)."""
    mismatch, placeholder_only = [], []
    rng = np.random.default_rng(SEED)

    for case_dir in CASES:
        atoms, cat, csym, layer_id, shells, layer_region = load_case(case_dir)
        spec = build_shell_pairs(atoms.get_positions()[cat], atoms.cell, atoms.pbc, layer_id,
                                 r_max_inplane=SPEC_R_MAX, r_max_inter=SPEC_R_MAX,
                                 tol=TOL, n_shell=SRO_PARAMS['n_shell'],
                                 dlayer_max=SPEC_DLAYER_MAX)
        s = (csym == "Ir").astype(float)[None, :]
        a_super = compute_alpha_frames(shells, s, layer_id)['alpha'][0]
        a_spec = compute_alpha_frames(spec, s, layer_id)['alpha'][0]

        keep = shells['in_spec'] & (shells['n_pairs'] > 0)
        key_super = list(zip(shells['l'][keep], shells['l2'][keep], shells['m'][keep]))
        spec_keep = spec['n_pairs'] > 0
        key_spec = list(zip(spec['l'][spec_keep], spec['l2'][spec_keep], spec['m'][spec_keep]))

        if key_super != key_spec:
            mismatch.append((case_dir.name, "entry 목록이 다르다"))
            continue
        same_r = np.allclose(shells['shell_r'][keep], spec['shell_r'][spec_keep])
        same_n = (shells['n_pairs'][keep] == spec['n_pairs'][spec_keep]).all()
        same_a = np.allclose(a_super[keep], a_spec[spec_keep], equal_nan=True)
        if not (same_r and same_n and same_a):
            mismatch.append((case_dir.name, f"r={same_r} n={same_n} alpha={same_a}"))

        # 스펙 범위 계산에만 있는 빈 슬롯 (6.0 Å 안에 껍질이 모자란 목록)
        n_empty = int((~spec_keep).sum())
        if n_empty:
            placeholder_only.append((facet_of(case_dir.name), n_empty))

    print("[2] 넓은 범위 결과의 in_spec 필터 == 스펙 범위로 직접 계산")
    print(f"  대조한 케이스 : {len(CASES)}개")
    print(f"  불일치        : {len(mismatch)}건" + (f" {mismatch[:3]}" if mismatch else ""))
    by_facet = {}
    for facet, n in placeholder_only:
        by_facet[facet] = by_facet.get(facet, 0) + n
    print(f"  스펙 쪽에만 있는 빈 슬롯(NA 행): {by_facet if by_facet else '없음'}")
    print("    ↳ 스펙 상한 6.0 A 안에 껍질이 모자란 목록이다. 상위집합에서는 그 껍질이")
    print("      6.0 A 밖에서 실제로 잡히므로(in_spec=False) 필터 뷰에서 행이 사라진다.")
    print("      shell_pairs.csv의 n_shell_found_spec 열과 러너 요약이 이것을 명시한다.")
    ok = not mismatch
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


# ------------------------------------------------------------- [3] C_lat 항등식
def clat_v1(symbols, pairs):
    """v1 C_lat (개념서 식 24–26): 무작위 기준이 복원 추출 x² + (1 − x)². [3]의 대조용."""
    if len(pairs) == 0:
        return np.nan
    f_same = np.mean(symbols[pairs[:, 0]] == symbols[pairs[:, 1]])
    x = np.mean(symbols == "Ir")
    f_rand = x**2 + (1 - x)**2
    return (f_same - f_rand) / (1 - f_rand) if f_rand < 1 else np.nan


def check_3_clat_identity():
    """같은 pair 집합에서 (1 − α_in) == (1 − C_lat)(N − 1)/N."""
    worst = 0.0
    n_checked = 0
    for case_dir in CASES:
        atoms, cat, csym, layer_id, shells, _ = load_case(case_dir)
        s = (csym == "Ir").astype(float)[None, :]
        alpha = compute_alpha_frames(shells, s, layer_id)['alpha'][0]

        for e in np.where(shells['is_in'] & (shells['n_pairs'] > 0))[0]:
            layer = shells['l'][e]
            mask = layer_id == layer
            idx = np.where(mask)[0]
            local = {g: k for k, g in enumerate(idx)}
            p0, p1 = shells['pair_start'][e], shells['pair_start'][e + 1]
            pairs = np.array([[local[i], local[j]]
                              for i, j in zip(shells['pair_i'][p0:p1], shells['pair_j'][p0:p1])])
            c_lat = clat_v1(csym[mask], pairs)
            n_l = int(mask.sum())
            if not np.isfinite(c_lat) or not np.isfinite(alpha[e]):
                continue
            worst = max(worst, abs((1 - alpha[e]) - (1 - c_lat) * (n_l - 1) / n_l))
            n_checked += 1

    print("[3] C_lat 항등식  (1-alpha_in) == (1-C_lat)(N-1)/N")
    print(f"  대조한 (케이스, 층, 껍질) : {n_checked}건")
    print(f"  최대 오차                 : {worst:.3e}")
    ok = worst < 1e-12
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


# ----------------------------------------------------------------- [4] 인공 배열
def check_4_synthetic():
    """(110) 한 층의 m=1 pair 위에서 인공 배열의 α 부호를 확인한다.

    무작위 셔플 → ≈ 0,  좌표로 반씩 가른 동종 분리 → > 0,  pair 그래프 2색 칠하기 → < 0
    """
    case_dir = [d for d in CASES if "_110_" in d.name and "x0p50" in d.name][0]
    atoms, cat, csym, layer_id, shells, _ = load_case(case_dir)
    e = np.where(shells['is_in'] & (shells['m'] == 1) & (shells['n_pairs'] > 0))[0][0]
    layer = shells['l'][e]
    mask = layer_id == layer
    n_l = int(mask.sum())
    p0, p1 = shells['pair_start'][e], shells['pair_start'][e + 1]
    pi, pj = shells['pair_i'][p0:p1], shells['pair_j'][p0:p1]

    def alpha_of(s_full):
        return compute_alpha_frames(shells, s_full[None, :], layer_id)['alpha'][0, e]

    # (a) 무작위 셔플 500회 (조성 고정)
    rng = np.random.default_rng(SEED)
    base = np.zeros(len(cat))
    base[np.where(mask)[0][: n_l // 2]] = 1.0
    samples = []
    for _ in range(500):
        s = base.copy()
        s[mask] = rng.permutation(s[mask])
        samples.append(alpha_of(s))
    samples = np.array(samples)
    sem = samples.std(ddof=1) / np.sqrt(len(samples))

    # (b) 동종 분리
    pos = atoms.get_positions()[cat][mask]
    order = np.argsort(pos[:, 0] + 1e-6 * pos[:, 1])
    s_split = np.zeros(len(cat))
    s_split[np.where(mask)[0][order[: n_l // 2]]] = 1.0
    a_split = alpha_of(s_split)

    # (c) 이종 교대: 면내 최근접 그래프를 BFS로 2색 칠하기
    color = {}
    adj = {}
    for i, j in zip(pi, pj):
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    for start in adj:
        if start in color:
            continue
        color[start], queue = 0, [start]
        while queue:
            u = queue.pop()
            for v in adj[u]:
                if v not in color:
                    color[v] = 1 - color[u]
                    queue.append(v)
    s_alt = np.zeros(len(cat))
    for node, c in color.items():
        s_alt[node] = float(c)
    a_alt = alpha_of(s_alt)

    print("[4] 인공 배열  (케이스", case_dir.name.split("_dO")[0], f"층 {layer}, N={n_l})")
    print(f"  무작위 셔플 500회 : mean {samples.mean():+.4f} (sem {sem:.4f}) "
          f"-> |mean| < 3 sem ? {abs(samples.mean()) < 3 * sem}")
    print(f"  동종 분리         : alpha {a_split:+.4f}  -> > 0 ? {a_split > 0}")
    print(f"  이종 교대(2색칠)  : alpha {a_alt:+.4f}  -> < 0 ? {a_alt < 0}")
    ok = abs(samples.mean()) < 3 * sem and a_split > 0 and a_alt < 0
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


# ------------------------------------------------------- [5] 결측 규약 · region 집계
def check_5_na_and_region():
    """pure layer가 NA인가, region α가 스펙의 가중합 식과 같은가."""
    case_dir = [d for d in CASES if "_110_" in d.name and "x0p50" in d.name][0]
    atoms, cat, csym, layer_id, shells, layer_region = load_case(case_dir)

    # (i) 한 층을 전부 Ir로 → f_0 = 0 → NA
    s = (csym == "Ir").astype(float)
    pure_layer = int(shells['l'][shells['is_in']][0])
    s_pure = s.copy()
    s_pure[layer_id == pure_layer] = 1.0
    res = compute_alpha_frames(shells, s_pure[None, :], layer_id, layer_region=layer_region)
    sel = shells['is_in'] & (shells['l'] == pure_layer)
    na_ok = bool(np.isnan(res['alpha'][0, sel]).all()) and not res['valid'][0, sel].any()

    # (ii) region α를 스펙 식으로 다시 계산
    res2 = compute_alpha_frames(shells, s[None, :], layer_id, layer_region=layer_region)
    worst, worst_gap = 0.0, 0.0
    for c, (region, m) in enumerate(zip(res2['region_names'], res2['region_m'])):
        member = np.where(shells['is_in'] & (shells['m'] == m)
                          & np.array([layer_region[x] == region for x in shells['l']]))[0]
        w = shells['n_pairs'][member].astype(float)
        f_obs, f_0 = res2['f_obs'][0, member], res2['f_0'][0, member]
        ok = np.isfinite(f_obs) & (w > 0)
        if not ok.any():
            continue
        manual = 1 - (w[ok] * f_obs[ok]).sum() / (w[ok] * f_0[ok]).sum()
        worst = max(worst, abs(manual - res2['region_alpha'][0, c]))
        # 참고: 층별 α 단순평균과의 차이
        naive = np.nanmean(res2['alpha'][0, member])
        worst_gap = max(worst_gap, abs(naive - res2['region_alpha'][0, c]))

    print("[5] 결측 규약 · region 집계")
    print(f"  pure layer(f_0=0) -> NA 이고 valid=False ? {na_ok}")
    print(f"  region alpha 재계산 최대 오차            : {worst:.3e}")
    print(f"  (참고) 층별 alpha 단순평균과의 최대 차이  : {worst_gap:.4f}"
          f"  <- 이만큼 달라지므로 가중합이어야 한다")
    ok = na_ok and worst < 1e-12
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok



# ------------------------------------------------------- [6] 외부 대조 (MC 산출물 SRO 표)
def check_6_crosscheck_mc_tables():
    """MC 산출물의 SRO 표와 이종 pair 비율 f_IrRu(= 표의 pair_frac_AB)가 같은가.

    α는 비교하지 않는다 — 표의 α는 전역 조성을, 우리 α는 층 조성을 기준(f_0)으로 쓴다.
    표의 그래프(cutoff 3.4 Å)는 각 facet의 최근접 껍질 하나에 해당한다:
      (100)/(110): 면내 m = 1 → 전역 표 · 층별 표 모두 대조
      (111)      : Δlayer = 2 m = 1 → 전역 표만 대조 (층별 표는 수직 간선을 양쪽 층에 넣는다)
    프레임은 global_step으로 맞추고, 수락된 스텝만 쓴다.
    """
    import json

    import pandas as pd

    sro_path, pf_path = OUTDIR / "sro_long.csv", OUTDIR / "per_frame.csv"
    print("[6] 외부 대조 — MC 산출물 SRO 표")
    if not (sro_path.exists() and pf_path.exists()):
        print(f"  [건너뜀] 배치 산출물이 없다: {OUTDIR.name}/")
        print("           run_descriptors_v2.py 를 먼저 돌린 뒤 이 검증을 다시 실행한다.")
        print("  -> SKIP\n")
        return None

    cols = ['case_id', 'frame', 'quantity', 'l', 'dlayer', 'm', 'n_pairs', 'f_obs', 'shell_r']
    ours = pd.concat(
        [c[c.quantity.isin(['alpha_in', 'alpha_out']) & (c.m == 1)]
         for c in pd.read_csv(sro_path, usecols=cols, chunksize=400_000)],
        ignore_index=True)
    pf = pd.read_csv(pf_path, usecols=['case_id', 'frame', 'global_step',
                                       'n_atomic_layers', 'facet'], dtype={'facet': str})
    ours = ours.merge(pf, on=['case_id', 'frame'])
    ours['layer_theirs'] = ours.n_atomic_layers - 1 - ours.l   # 표의 층 번호는 역순

    n_layer_rows = n_global_rows = 0
    worst_layer = worst_global = 0.0
    bad_pairs = []
    for case_dir in CASES:
        o = ours[ours.case_id == case_dir.name]
        if not len(o):
            bad_pairs.append((case_dir.name, "우리 산출물에 이 케이스가 없다"))
            continue
        analysis = sorted(case_dir.glob("analysis_*"))[0]
        summary = json.load(open(analysis / "analysis_summary.json"))

        # 표의 cutoff 그래프에 대응하는 껍질 = 최근접 껍질
        near = o[np.isclose(o.shell_r, o.shell_r.min(), atol=0.02)]
        in_plane = bool((near.dlayer == 0).all())

        # (i) 전역 표 (pair 수부터 대조)
        n_ours = int(near.groupby('frame').n_pairs.sum().iloc[0])
        if n_ours != summary['n_cation_neighbor_edges']:
            bad_pairs.append((case_dir.name,
                              f"pair 수 {n_ours} vs 표 {summary['n_cation_neighbor_edges']}"))
            continue
        g = pd.read_csv(analysis / "sro_global_table.csv")
        g = g[g.accepted]
        agg = (near.assign(w=near.n_pairs * near.f_obs)
                   .groupby(['frame', 'global_step'])
                   .agg(num=('w', 'sum'), den=('n_pairs', 'sum')).reset_index())
        j = agg.assign(f_obs=agg.num / agg.den).merge(
            g[['global_step', 'pair_frac_AB']], on='global_step', how='inner')
        worst_global = max(worst_global, float((j.f_obs - j.pair_frac_AB).abs().max()))
        n_global_rows += len(j)

        # (ii) 층별 표 (면내 그래프인 facet만)
        if in_plane:
            t = pd.read_csv(analysis / "sro_layer_table.csv")
            t = t[t.accepted].rename(columns={'layer': 'layer_theirs'})
            jl = near.merge(t[['global_step', 'layer_theirs', 'n_edges', 'pair_frac_AB']],
                            on=['global_step', 'layer_theirs'], how='inner')
            if not (jl.n_pairs == jl.n_edges).all():
                bad_pairs.append((case_dir.name, "층별 pair 수 불일치"))
                continue
            worst_layer = max(worst_layer, float((jl.f_obs - jl.pair_frac_AB).abs().max()))
            n_layer_rows += len(jl)

    print(f"  전역 표 (30케이스) : {n_global_rows:,}행 대조 · f_obs 최대오차 {worst_global:.3e}")
    print(f"  층별 표 ((100)/(110)) : {n_layer_rows:,}행 대조 · f_obs 최대오차 {worst_layer:.3e}")
    print(f"  pair 수 불일치 : {len(bad_pairs)}건" + (f" {bad_pairs[:3]}" if bad_pairs else ""))
    print("    ↳ (111)은 층별 표와 대조하지 않는다 — 표의 (111) 그래프는 면내가 아니라")
    print("      dlayer=2 수직 결합이고, 층별 표는 그 간선을 양쪽 층에 중복으로 넣는다.")
    ok = not bad_pairs and worst_global < 1e-12 and worst_layer < 1e-12
    print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    return ok


def main():
    print(f"데이터: {MC_DIR}")
    print(f"케이스 {len(CASES)}개 · 껍질 설정 {SRO_PARAMS}\n")
    results = [
        ("[1] 껍질 목록", check_1_shell_lists()),
        ("[2] 스펙 범위 필터", check_2_superset_contains_spec()),
        ("[3] C_lat 항등식", check_3_clat_identity()),
        ("[4] 인공 배열", check_4_synthetic()),
        ("[5] 결측 규약 · region 집계", check_5_na_and_region()),
        ("[6] 외부 대조 (MC 산출물 표)", check_6_crosscheck_mc_tables()),
    ]
    print("=" * 70)
    for name, ok in results:
        print(f"  {name:30s} {'SKIP' if ok is None else ('PASS' if ok else 'FAIL')}")
    print("=" * 70)
    # SKIP은 실패로 보지 않는다
    return 0 if all(ok is not False for _, ok in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
