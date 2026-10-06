#!/usr/bin/env python3
"""
초기 구조(surface.xyz)의 cation–cation 거리 껍질을 facet별로 실측한다 (화면 출력만).

    E_in(l, m) · E_out(l, k, m) 설정: cation shell 2개, tolerance 0.18 Å, 최대 6.0 Å

확인하는 것: (1) 껍질이 앉는 거리  (2) 6.0 Å 안에 껍질이 2개 잡히는가
            (3) tolerance 0.18 Å로 껍질이 깨끗이 갈라지는가 (간격 > tol, 폭 < tol)
Δlayer 3까지, 면내는 6.6 Å까지도 함께 출력한다.
출력의 k는 층 간격(Δlayer)이다.
"""
import re
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list

HERE = Path(__file__).resolve().parent   # v2/
PROJECT_ROOT = HERE.parent               # 원본 데이터 위치
sys.path.insert(0, str(HERE))

from descriptors_v2.regions import compute_layer_id  # noqa: E402

MC_DIR = PROJECT_ROOT / "Ir_Ru_oxide" / "correct" / "MC_final_30cases_20260806" / "mc_runs"

CATION_SPECIES = ("Ir", "Ru")
R_MAX = 6.0            # 스펙: 최대 6.0 A
TOL = 0.18             # 스펙: shell tolerance 0.18 A
N_SHELL_WANTED = 2     # 스펙: cation shell 2개
DLAYER_MAX = 3         # 출력할 Δlayer 상한
R_MAX_INPLANE = 6.6    # 면내 확장 상한
VACUUM_GAP_THRESHOLD = 5.0
LAYER_GAP_THRESHOLD = 0.5

CASE_RE = re.compile(
    r"x0p(?P<x>\d+)_(?P<facet>\d+)_(?P<surf_id>S\d+)_(?P<o_condition>[A-Za-z]+)"
)


def find_shells(distances, tol=TOL):
    """거리를 정렬해 간격이 tol을 넘는 곳에서 껍질을 나눈다.

    반환: [(중심, 폭, 개수, 다음 껍질까지의 간격), ...] 거리 오름차순. 폭 = 껍질 안 최대 − 최소 거리.
    """
    if len(distances) == 0:
        return []
    d = np.sort(np.asarray(distances, dtype=float))
    cut = np.where(np.diff(d) > tol)[0]
    groups = np.split(d, cut + 1)

    shells = []
    for k, g in enumerate(groups):
        gap_next = float(groups[k + 1][0] - g[-1]) if k + 1 < len(groups) else np.nan
        shells.append((float(g.mean()), float(g[-1] - g[0]), int(len(g)), gap_next))
    return shells


def fmt_shells(shells, n_show=3):
    """껍질 목록을 한 줄 문자열로. '중심(폭, 개수)' 형태."""
    if not shells:
        return "없음"
    parts = [f"{c:.3f}(w{w:.3f}, n={n})" for c, w, n, _ in shells[:n_show]]
    if len(shells) > n_show:
        parts.append(f"...+{len(shells) - n_show}")
    return "  ".join(parts)


case_dirs = sorted(d for d in MC_DIR.iterdir() if d.is_dir())
print(f"데이터: {MC_DIR}")
print(f"케이스 {len(case_dirs)}개 · R_MAX={R_MAX} A · TOL={TOL} A · 목표 shell {N_SHELL_WANTED}개")
print()

rows = []
for case_dir in case_dirs:
    meta = CASE_RE.search(case_dir.name)
    ref_path = case_dir / "surface.xyz"
    if meta is None or not ref_path.exists():
        print(f"  [skip] {case_dir.name}")
        continue

    facet = meta.group("facet")
    x_ir = int(meta.group("x")) / 100.0
    o_cond = meta.group("o_condition")

    atoms = read(str(ref_path))
    symbols = np.array(atoms.get_chemical_symbols())
    cat_idx = np.where(np.isin(symbols, CATION_SPECIES))[0]

    layer_id = compute_layer_id(atoms, cat_idx,
                                vacuum_gap_threshold=VACUUM_GAP_THRESHOLD,
                                layer_gap_threshold=LAYER_GAP_THRESHOLD)

    # cation만 남긴 구조 (원소 종류는 이웃 탐색에 무관하므로 더미 번호)
    cat = Atoms(numbers=np.ones(len(cat_idx), dtype=int),
                positions=atoms.get_positions()[cat_idx],
                cell=atoms.cell, pbc=atoms.pbc)
    # 큰 상한으로 찾은 뒤 목록마다 자기 상한으로 거른다 (면내 6.6 Å, 층간 6.0 Å)
    i, j, d = neighbor_list("ijd", cat, max(R_MAX, R_MAX_INPLANE))
    upper = i < j
    i, j, d = i[upper], j[upper], d[upper]

    dlayer = np.abs(layer_id[i] - layer_id[j])
    in_plane = find_shells(d[(dlayer == 0) & (d <= R_MAX)])
    inter_1 = find_shells(d[(dlayer == 1) & (d <= R_MAX)])

    row = dict(case=case_dir.name, facet=facet, x_ir=x_ir, o_cond=o_cond,
               n_layers=int(layer_id.max() + 1), n_cat=len(cat_idx),
               in_plane=in_plane, inter_1=inter_1)

    for k in range(2, DLAYER_MAX + 1):
        row[f"inter_{k}"] = find_shells(d[(dlayer == k) & (d <= R_MAX)])
    if R_MAX_INPLANE > R_MAX:
        row["in_plane_ext"] = find_shells(d[dlayer == 0])

    rows.append(row)

# ---- 케이스별 상세 ----
print("=" * 110)
print("케이스별 — 6.0 A 안의 껍질 (중심, 폭 w, pair 개수 n)")
print("※ 출력의 k 는 층 간격(Delta layer)이다. 스펙 E_out(l,k,m) 의 k(상대 층 번호)와 다르다.")
print("=" * 110)
cur_facet = None
for r in rows:
    if r["facet"] != cur_facet:
        cur_facet = r["facet"]
        print(f"\n--- ({cur_facet}) ---")
    print(f"  x={r['x_ir']:.2f} {r['o_cond']:7s} 층{r['n_layers']:3d} cation{r['n_cat']:5d}")
    print(f"      면내   E_in      : {fmt_shells(r['in_plane'])}")
    print(f"      층간   E_out(k=1): {fmt_shells(r['inter_1'])}")
    for k in range(2, DLAYER_MAX + 1):
        print(f"      층간   E_out(k={k}): {fmt_shells(r[f'inter_{k}'])}")
    if R_MAX_INPLANE > R_MAX:
        print(f"      면내   E_in(<={R_MAX_INPLANE}): {fmt_shells(r['in_plane_ext'])}")

# ---- facet별 판정 ----
print()
print("=" * 110)
print(f"판정 — 6.0 A 안에 껍질 {N_SHELL_WANTED}개가 잡히는가 / tolerance {TOL} A로 갈라지는가")
print("=" * 110)
print(f"{'facet':>6} {'종류':>10} {'껍질수':>7} {'최소간격':>9} {'최대폭':>8}  판정")
print("-" * 110)

for facet in ["100", "110", "111"]:
    sub = [r for r in rows if r["facet"] == facet]
    if not sub:
        continue
    kinds = [("면내 E_in", "in_plane"), ("층간 E_out", "inter_1")]
    kinds += [(f"층간 E_out k={k}", f"inter_{k}") for k in range(2, DLAYER_MAX + 1)]
    if R_MAX_INPLANE > R_MAX:
        kinds += [(f"면내 <={R_MAX_INPLANE}", "in_plane_ext")]

    for kind, key in kinds:
        n_shells = [len(r[key]) for r in sub]
        gaps = [g for r in sub for *_, g in r[key] if np.isfinite(g)]
        widths = [w for r in sub for _, w, _, _ in r[key]]

        n_min, n_max = min(n_shells), max(n_shells)
        n_str = f"{n_min}" if n_min == n_max else f"{n_min}~{n_max}"
        min_gap = min(gaps) if gaps else np.nan
        max_w = max(widths) if widths else np.nan

        flags = []
        if n_min < N_SHELL_WANTED:
            flags.append(f"껍질 {N_SHELL_WANTED}개 미만")
        if np.isfinite(min_gap) and min_gap <= TOL:
            flags.append("껍질 간격 <= TOL (합쳐짐)")
        if np.isfinite(max_w) and max_w > TOL:
            flags.append("껍질 폭 > TOL (쪼개짐)")
        verdict = "OK" if not flags else "확인 필요: " + ", ".join(flags)
        if n_max == 0:
            # 그 층 간격에 pair 자체가 없다 (예: (110)의 Δlayer ≥ 2)
            verdict = "pair 없음 (구조상 정상)"

        print(f"{facet:>6} {kind:>10} {n_str:>7} "
              f"{min_gap:>9.3f} {max_w:>8.3f}  {verdict}")
