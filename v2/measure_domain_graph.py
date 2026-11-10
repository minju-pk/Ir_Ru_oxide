#!/usr/bin/env python3
"""
초기 구조(surface.xyz)에서 domain graph(층별 면내 대칭화 k=4 그래프)를 facet별로 실측한다 (화면 출력만).

  [1] 4번째·5번째 이웃 거리 — 동률이면 "가장 가까운 4개"가 유일하지 않다
  [2] 대칭화 후 degree와 층당 간선 수 (동률 포함 vs strict)
  [3] 최소이미지 검증 — domain.py(ase.neighborlist) vs 면내 5×5 이미지 전수 탐색
"""
import sys
from pathlib import Path

import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent   # v2/
PROJECT_ROOT = HERE.parent               # 원본 데이터 위치
sys.path.insert(0, str(HERE))

from descriptors_v2.regions import compute_layer_id, find_vacuum_axis   # noqa: E402
from descriptors_v2.domain import (build_domain_graph, inplane_neighbor_entries,  # noqa: E402
                                   DEFAULT_K, DEFAULT_TIE_TOL, DEFAULT_R_CUT)

MC_DIR = PROJECT_ROOT / "Ir_Ru_oxide" / "correct" / "MC_final_30cases_20260806" / "mc_runs"

TIE_ATOL = 1e-6      # 거리 동률 판정


def brute_force_inplane(positions, cell, pbc, axis, r_max):
    """면내 이미지 (−2..2)² 를 전부 만들어 거리를 직접 잰다 (독립 구현).

    반환: d_min (N,N 최소이미지 거리, 대각 inf) · n_multi (r_max 안에 같은 상대의 이미지가
          2개 이상인 i<j 쌍 수) · d_self (자기 주기 이미지까지 최소 거리)
    """
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    in_plane = [k for k in range(3) if k != axis]
    n = len(positions)

    shifts = []
    for na in (-2, -1, 0, 1, 2):
        for nb in (-2, -1, 0, 1, 2):
            vec = np.zeros(3)
            if pbc[in_plane[0]]:
                vec = vec + na * cell[in_plane[0]]
            elif na:
                continue
            if pbc[in_plane[1]]:
                vec = vec + nb * cell[in_plane[1]]
            elif nb:
                continue
            shifts.append(vec)
    shifts = np.array(shifts)

    d_all = np.linalg.norm(
        positions[:, None, None, :] - (positions[None, :, None, :] + shifts[None, None, :, :]),
        axis=-1)                                            # (N, N, 9)

    zero_shift = np.linalg.norm(shifts, axis=-1) == 0
    eye = np.eye(n, dtype=bool)
    # 자기 자신의 0-shift 이미지는 제외, 나머지 자기 이미지는 d_self에 쓴다
    d_self = float(d_all[eye][:, ~zero_shift].min()) if n else np.inf
    d_pair = d_all.copy()
    d_pair[eye] = np.inf
    d_min = d_pair.min(axis=-1)

    close = (d_all <= r_max)
    close[eye] = False
    n_multi = int((close.sum(axis=-1) >= 2)[np.triu_indices(n, 1)].sum())
    return d_min, n_multi, d_self


def main():
    case_dirs = sorted(d for d in MC_DIR.glob("ir_ru_sqs_*") if d.is_dir())
    print(f"케이스 {len(case_dirs)}개 · k={DEFAULT_K} · tie_tol={DEFAULT_TIE_TOL} A")
    print(f"입력: {MC_DIR}\n")

    per_facet = {}
    mismatch_total = 0
    multi_total = 0
    self_min = np.inf

    for case_dir in case_dirs:
        ref = case_dir / "surface.xyz"
        if not ref.exists():
            print(f"  [skip] surface.xyz 없음: {case_dir.name}")
            continue
        atoms = read(str(ref))
        symbols = np.array(atoms.get_chemical_symbols())
        cat = np.where(np.isin(symbols, ("Ir", "Ru")))[0]
        axis = find_vacuum_axis(atoms)
        layer_id = compute_layer_id(atoms, cat)
        positions = atoms.get_positions()[cat]

        graph = build_domain_graph(positions, atoms.cell, atoms.pbc, layer_id, axis)
        ei, ej, ed = inplane_neighbor_entries(
            positions, atoms.cell, atoms.pbc, layer_id, axis)[:3]

        facet = case_dir.name.split("_")[4]
        rec = per_facet.setdefault(facet, dict(
            cases=0, layers=0, d4=[], d5=[], gap=[], n_tie=[], n_nodes=[],
            edges=[], edges_strict=[], deg_min=[], deg_max=[], deg_mean=[]))
        rec["cases"] += 1

        for layer in range(graph["n_layers"]):
            n_node = graph["layer_n_nodes"][layer]
            if n_node < 2:
                continue
            rec["layers"] += 1
            rec["n_nodes"].append(n_node)
            rec["d4"].append(graph["layer_d4"][layer])
            rec["d5"].append(graph["layer_d5"][layer])
            rec["gap"].append(graph["layer_d5"][layer] - graph["layer_d4"][layer])
            rec["n_tie"].append(graph["layer_n_tie"][layer])
            rec["edges"].append(graph["layer_n_edges"][layer])
            rec["edges_strict"].append(graph["layer_n_edges_strict"][layer])
            rec["deg_min"].append(graph["layer_deg_min"][layer])
            rec["deg_max"].append(graph["layer_deg_max"][layer])
            rec["deg_mean"].append(graph["layer_deg_mean"][layer])

            idx = np.where(layer_id == layer)[0]
            local = {g: l for l, g in enumerate(idx)}
            n_loc = len(idx)
            # domain.py가 쓰는 이웃 목록을 층별 거리 행렬로
            sel = layer_id[ei] == layer
            d = np.full((n_loc, n_loc), np.inf)
            for a, b, dist in zip(ei[sel], ej[sel], ed[sel]):
                la, lb = local[a], local[b]
                d[la, lb] = min(d[la, lb], dist)

            # 5×5 이미지 전수 탐색과 대조
            r_probe = float(np.nanmax(graph["layer_d5"]))
            d_bf, n_multi, d_self = brute_force_inplane(
                positions[idx], atoms.cell, atoms.pbc, axis, r_probe)
            off_diag = ~np.eye(n_loc, dtype=bool)
            # 같은 반경(r_cut) 안에서만 비교
            both = off_diag & (d_bf <= DEFAULT_R_CUT)
            if not np.allclose(d[both], d_bf[both], atol=1e-9):
                mismatch_total += 1
                print(f"  <<< 최소이미지 불일치: {case_dir.name} 층 {layer} "
                      f"최대차 {np.abs(d[both] - d_bf[both]).max():.2e} A")
            multi_total += n_multi
            self_min = min(self_min, d_self)

    # ---------------------------------------------------------------- 요약
    print("=" * 100)
    print("[1] k=4 가 잘 정의되는가 — 4번째(d4)와 5번째(d5) 이웃 거리")
    print(f"{'facet':7s} {'케이스':>5s} {'층':>5s} {'층당원자':>7s} "
          f"{'d4 중앙':>8s} {'d5 중앙':>8s} {'d5-d4 최소':>10s} {'동률 층':>8s}  판정")
    for facet in sorted(per_facet):
        r = per_facet[facet]
        gap_min = float(np.nanmin(r["gap"]))
        n_tie_layers = int(np.sum(np.array(r["n_tie"]) > 0))
        verdict = ("⚠️ 동률 — k=4 가 동등한 이웃을 자른다" if gap_min < TIE_ATOL
                   else f"동률 없음 (여유 {gap_min:.3f} A)")
        print(f"({facet})  {r['cases']:5d} {r['layers']:5d} "
              f"{int(np.median(r['n_nodes'])):7d} "
              f"{np.nanmedian(r['d4']):8.3f} {np.nanmedian(r['d5']):8.3f} "
              f"{gap_min:10.3f} {n_tie_layers:4d}/{r['layers']:<3d}  {verdict}")

    print()
    print("[2] 대칭화 후 degree 와 간선 수 — 동률 포함(채택) vs strict k=4(스펙 문구)")
    print(f"{'facet':7s} {'degree(포함)':>14s} {'간선/층(포함)':>14s} "
          f"{'간선/층(strict)':>16s} {'버려지는 간선':>13s}")
    for facet in sorted(per_facet):
        r = per_facet[facet]
        e, es = float(np.mean(r["edges"])), float(np.mean(r["edges_strict"]))
        drop = f"{e - es:.1f} ({(e - es) / e * 100:.0f}%)" if e else "-"
        deg_span = f"{int(min(r['deg_min']))}~{int(max(r['deg_max']))}"
        print(f"({facet})  {deg_span:>14s} {e:14.1f} {es:16.1f} {drop:>13s}")

    print()
    print("[3] 최소이미지 검증 — descriptors_v2/domain.py (ase.neighborlist) vs 5x5 이미지 전수")
    print(f"  좌표 불일치 층: {mismatch_total}개  "
          f"{'<<< 확인 필요' if mismatch_total else '→ PASS'}")
    print(f"  같은 상대의 이미지가 2개 이상 이웃 반경에 들어온 쌍: {multi_total}개  "
          f"{'<<< round 방식이 놓친다 — 확인 필요' if multi_total else '→ PASS'}")
    print(f"  자기 자신의 주기 이미지까지 최소 거리: {self_min:.3f} A  "
          f"(이웃 반경 ~6.4 A 보다 충분히 크면 자가 간선이 생기지 않는다)")

    print()
    print("=" * 100)
    tie_facets = [f for f in sorted(per_facet)
                  if float(np.nanmin(per_facet[f]["gap"])) < TIE_ATOL]
    if tie_facets:
        print(f"결론: 동률이 있는 facet = {', '.join('(' + f + ')' for f in tie_facets)}."
              f"  나머지는 동률이 없어 두 규칙이 **같은 그래프**를 준다.")
        print("      strict k=4로 계산하려면 tie_tol=0.0 으로 두면 되고,")
        print(f"      값이 바뀌는 것은 {', '.join('(' + f + ')' for f in tie_facets)} 케이스뿐이다.")
    else:
        print("결론: 동률 없음 — k=4 가 애매함 없이 정의된다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
