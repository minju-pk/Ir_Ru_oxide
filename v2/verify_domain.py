#!/usr/bin/env python3
"""
Domain morphology 구현 검증 (화면 출력만). 대표 케이스 = facet마다 첫 케이스.

  [1] 그래프 재구성   면내 5×5 이미지 전수 탐색으로 그래프를 다시 만들어 간선 집합 대조
  [2] 인공 배열       전부 Ir · 고립 원자 1개 · 줄무늬(한 방향만 관통)
  [3] 항등식          Σ_q p_q = 1 · 0 < D ≤ 1 · 0 ≤ H ≤ 1 · 층 합 = region 합 · n_cluster = 1 → D = 1
  [4] I_R 재계산      간선 목록에서 이종 비율을 직접 센다
  [5] 결측 규약       원소 없는 region → D·H = NA (다른 region으로 번지지 않음) · n_cluster = 1 → H = 0
  [6] v1 비교         cutoff 3.4 Å 그래프의 사슬·고립 원자가 k=4 그래프에서 사라지는가
  [7] 회귀            배치 산출물의 공통 칼럼이 이전 산출물과 같은가 (산출물이 없으면 SKIP)
                      내부 회귀 확인용 — 기준값(descriptors_v2_sro_5/)이 공유본에 없어 공유본에서는 항상 SKIP
  [8] 손계산 대조     덩어리 크기 목록에 스펙 식을 직접 적용해 per_frame.csv와 비교.
                      P는 복제 셀이라는 다른 알고리즘으로 판정
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import iread, read
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from descriptors_v2.regions import (compute_layer_id, find_vacuum_axis,      # noqa: E402
                                    get_regions)
from descriptors_v2.sro import layer_region_map, REGION_ORDER                # noqa: E402
from descriptors_v2.core import get_cation_pairs_within_cutoff, compute_ddom  # noqa: E402
from descriptors_v2.domain import (build_domain_graph, compute_domain_frames,  # noqa: E402
                                   DEFAULT_K, DEFAULT_TIE_TOL, DEFAULT_R_CUT,
                                   ELEM_ORDER)

MC_DIR = PROJECT_ROOT / "Ir_Ru_oxide" / "correct" / "MC_final_30cases_20260806" / "mc_runs"
OUTDIR = HERE / "descriptors_v2_domain_6"      # 배치 산출물
PREV_OUTDIR = HERE / "descriptors_v2_sro_5"    # 검증 [7]의 기준값
V1_CUTOFF = 3.4

CASES = sorted(d for d in MC_DIR.glob("ir_ru_sqs_*") if d.is_dir())
REP = {}          # facet -> 대표 케이스 (facet당 첫 번째)
for _d in CASES:
    REP.setdefault(_d.name.split("_")[4], _d)


def load_case(case_dir):
    """초기 구조에서 그래프 · region · 층 라벨을 만든다."""
    atoms = read(str(case_dir / "surface.xyz"))
    symbols = np.array(atoms.get_chemical_symbols())
    cat = np.where(np.isin(symbols, ("Ir", "Ru")))[0]
    axis = find_vacuum_axis(atoms)
    layer_id = compute_layer_id(atoms, cat)
    positions = atoms.get_positions()[cat]
    graph = build_domain_graph(positions, atoms.cell, atoms.pbc, layer_id, axis)
    regions5 = get_regions(atoms, cat, "slab", layer_id=layer_id,
                           split_top_bottom=True, axis=axis)
    layer_region = layer_region_map(layer_id, regions5)
    return dict(atoms=atoms, cat=cat, axis=axis, layer_id=layer_id, positions=positions,
                graph=graph, regions5=regions5, layer_region=layer_region,
                s0=(symbols[cat] == "Ir"))


def final_labels(case_dir, n_cat):
    """마지막 구조(final_structure.xyz)의 occupation. 없거나 cation 수가 다르면 None."""
    finals = sorted(case_dir.glob("run_*/final_structure.xyz"))
    if not finals:
        return None
    atoms = read(str(finals[0]))
    symbols = np.array(atoms.get_chemical_symbols())
    cat = np.where(np.isin(symbols, ("Ir", "Ru")))[0]
    if len(cat) != n_cat:
        return None
    return symbols[cat] == "Ir"


# ---------------------------------------------------------------- [1]
def brute_force_graph(positions, cell, pbc, layer_id, axis,
                      k=DEFAULT_K, tie_tol=DEFAULT_TIE_TOL, r_cut=DEFAULT_R_CUT):
    """면내 5×5 이미지를 직접 깔아 같은 규칙(k개 + 동률 포함, 대칭화 OR)으로 간선 집합을 만든다."""
    cell = np.asarray(cell, dtype=float)
    in_plane = [a for a in range(3) if a != axis]
    shifts, offs = [], []
    for na in range(-2, 3):
        for nb in range(-2, 3):
            shifts.append(na * cell[in_plane[0]] + nb * cell[in_plane[1]])
            offs.append((na, nb))
    shifts, offs = np.array(shifts), np.array(offs)

    edges = set()
    for layer in np.unique(layer_id):
        idx = np.where(layer_id == layer)[0]
        p = positions[idx]
        n = len(idx)
        # (n, n, 25) 거리와 오프셋
        d = np.linalg.norm(p[:, None, None, :] - (p[None, :, None, :] + shifts[None, None]),
                           axis=-1)
        self_zero = np.zeros((n, n, len(offs)), dtype=bool)
        self_zero[np.eye(n, dtype=bool)] = (np.abs(offs).sum(axis=1) == 0)
        d = np.where(self_zero, np.inf, d)

        cand = d <= r_cut
        src, tgt, sh = np.where(cand)
        dist = d[src, tgt, sh]
        # 원자별 k번째 거리
        d_k = np.full(n, np.inf)
        for a in range(n):
            da = np.sort(dist[src == a])
            if len(da) >= k:
                d_k[a] = da[k - 1]
        sel = dist <= d_k[src] + tie_tol
        for a, b, sh_i in zip(src[sel], tgt[sel], sh[sel]):
            ga, gb = idx[a], idx[b]
            off = offs[sh_i]
            if ga < gb:
                edges.add((int(ga), int(gb), int(off[0]), int(off[1])))
            else:
                edges.add((int(gb), int(ga), int(-off[0]), int(-off[1])))
    return edges


def check_1_graph():
    print("\n[1] 그래프 재구성 — 5x5 이미지 전수로 다시 만들어 간선 집합 대조")
    ok = True
    for facet, case_dir in REP.items():
        c = load_case(case_dir)
        g = c["graph"]
        mine = set(zip(g["edge_i"].tolist(), g["edge_j"].tolist(),
                       g["edge_off"][:, 0].tolist(), g["edge_off"][:, 1].tolist()))
        theirs = brute_force_graph(c["positions"], c["atoms"].cell, c["atoms"].pbc,
                                   c["layer_id"], c["axis"])
        same = mine == theirs
        deg_ok = int(g["node_deg"].min()) >= DEFAULT_K
        intra = bool((c["layer_id"][g["edge_i"]] == c["layer_id"][g["edge_j"]]).all())
        upper = bool((g["edge_i"] < g["edge_j"]).all())
        print(f"  ({facet}) 간선 {g['n_edges']:5d}  집합 일치 {same}  "
              f"degree>={DEFAULT_K} {deg_ok} (min {int(g['node_deg'].min())})  "
              f"층 내부만 {intra}  i<j {upper}")
        if not same:
            print(f"        <<< 우리에만 {len(mine - theirs)}개, 전수에만 {len(theirs - mine)}개")
        ok &= same and deg_ok and intra and upper
    return ok


# ---------------------------------------------------------------- [2]
def check_2_synthetic():
    print("\n[2] 인공 배열 — 답을 아는 배열 3종")
    ok = True
    for facet, case_dir in REP.items():
        c = load_case(case_dir)
        g, lr = c["graph"], c["layer_region"]
        n = g["n_nodes"]
        ir_i, ru_i = 0, 1        # ELEM_ORDER = ('Ir', 'Ru')

        # (a) 전부 Ir: D = 1, Ru는 NA, I = 0, 전부 관통
        s_all = np.ones(n, dtype=bool)
        r = compute_domain_frames(g, s_all[None, :], lr)
        a_ok = (np.allclose(r["D"][0, :, ir_i], 1.0)
                and np.all(np.isnan(r["D"][0, :, ru_i]))
                and np.all(np.isnan(r["H"][0, :, ru_i]))
                and np.allclose(r["I"][0], 0.0)
                and bool(r["P"][0, :, ir_i].all()))
        # n_cluster = 1 → H = 0
        single = r["n_cluster"][0, :, ir_i] == 1
        a_ok &= bool(np.all(r["H"][0, single, ir_i] == 0.0))

        # (b) 표면층 원자 하나만 Ir
        ts_layer = [l for l in range(g["n_layers"]) if lr[l] == "TS"][0]
        one = np.where(g["node_layer"] == ts_layer)[0][0]
        s_one = np.zeros(n, dtype=bool)
        s_one[one] = True
        r = compute_domain_frames(g, s_one[None, :], lr)
        ts = REGION_ORDER.index("TS")
        b_ok = (r["D"][0, ts, ir_i] == 1.0 and r["n_cluster"][0, ts, ir_i] == 1
                and r["H"][0, ts, ir_i] == 0.0
                and not r["P"][0, ts, ir_i].any())

        # (c) 줄무늬: 첫 격자방향 기준 절반만 Ir
        frac = c["positions"] @ np.linalg.inv(np.asarray(c["atoms"].cell))
        in_plane = [a for a in range(3) if a != c["axis"]]
        s_str = (frac[:, in_plane[0]] % 1.0) < 0.5
        r = compute_domain_frames(g, s_str[None, :], lr)
        # b 방향으로만 관통해야 한다
        p_ir = r["P"][0, ts, ir_i]
        c_ok = bool(p_ir[1] and not p_ir[0])

        print(f"  ({facet}) 전부 Ir {a_ok}  고립 원자 {b_ok}  "
              f"줄무늬(한 방향만 관통) {c_ok}  [줄무늬 P = {p_ir.tolist()}]")
        ok &= a_ok and b_ok and c_ok
    return ok


# ---------------------------------------------------------------- [3]
def check_3_identities():
    print("\n[3] 항등식 — sum_q p_q = 1 · 0 < D <= 1 · 0 <= H <= 1 · 층 합 = region 합")
    ok = True
    rng = np.random.default_rng(42)
    for facet, case_dir in REP.items():
        c = load_case(case_dir)
        g, lr = c["graph"], c["layer_region"]
        n = g["n_nodes"]
        # 실제 배열 + 무작위 배열
        frames = [c["s0"]]
        fin = final_labels(case_dir, n)
        if fin is not None:
            frames.append(fin)
        for _ in range(8):
            frames.append(rng.permutation(c["s0"]))
        s = np.array(frames)
        r = compute_domain_frames(g, s, lr)

        region_index = {name: i for i, name in enumerate(REGION_ORDER)}
        layer_r = np.array([region_index[lr[l]] for l in range(g["n_layers"])])

        # Σ_q p_q = 1  ⇔  region의 n_a가 라벨로 센 개수와 같다
        n_a_ref = np.zeros((len(s), len(REGION_ORDER), 2))
        for ri, name in enumerate(REGION_ORDER):
            mask = c["regions5"][name]
            n_a_ref[:, ri, 0] = (s[:, mask]).sum(axis=1)
            n_a_ref[:, ri, 1] = (~s[:, mask]).sum(axis=1)
        e_pq = float(np.abs(r["n_a"] - n_a_ref).max())

        # 층 합 = region 합
        lay_sum = np.zeros_like(r["n_a"])
        np.add.at(lay_sum, (slice(None), layer_r, slice(None)), r["layer_n_a"])
        e_layer = float(np.abs(lay_sum - r["n_a"]).max())

        d, h = r["D"], r["H"]
        d_ok = bool(np.all((d[~np.isnan(d)] > 0) & (d[~np.isnan(d)] <= 1 + 1e-12)))
        h_ok = bool(np.all((h[~np.isnan(h)] >= -1e-12) & (h[~np.isnan(h)] <= 1 + 1e-12)))
        ncl_ok = bool(np.all(r["n_cluster"][r["n_a"] > 0] >= 1))
        one_cl = (r["n_cluster"] == 1) & (r["n_a"] > 0)
        d1_ok = bool(np.allclose(d[one_cl], 1.0))

        print(f"  ({facet}) 프레임 {len(s)}  n_a 오차 {e_pq:.1e}  층합-region합 {e_layer:.1e}  "
              f"0<D<=1 {d_ok}  0<=H<=1 {h_ok}  n_cluster>=1 {ncl_ok}  n_cluster=1->D=1 {d1_ok}")
        ok &= (e_pq == 0) and (e_layer == 0) and d_ok and h_ok and ncl_ok and d1_ok
    return ok


# ---------------------------------------------------------------- [4]
def check_4_interface_density():
    print("\n[4] I_R 독립 재계산 — 간선 목록에서 이종 비율을 직접 센다")
    ok = True
    rng = np.random.default_rng(7)
    for facet, case_dir in REP.items():
        c = load_case(case_dir)
        g, lr = c["graph"], c["layer_region"]
        s = np.array([c["s0"]] + [rng.permutation(c["s0"]) for _ in range(5)])
        r = compute_domain_frames(g, s, lr)

        region_index = {name: i for i, name in enumerate(REGION_ORDER)}
        edge_r = np.array([region_index[lr[l]] for l in g["edge_layer"]])

        # (1/|E|) Σ |s_i − s_j|
        err_i = 0.0
        for t in range(len(s)):
            hetero = np.abs(s[t][g["edge_i"]].astype(float)
                            - s[t][g["edge_j"]].astype(float))
            for ri in range(len(REGION_ORDER)):
                m = edge_r == ri
                ref = hetero[m].mean() if m.any() else np.nan
                got = r["I"][t, ri]
                if np.isnan(ref) and np.isnan(got):
                    continue
                err_i = max(err_i, abs(ref - got))

        print(f"  ({facet}) I_R 최대오차 {err_i:.2e}  (간선 {g['n_edges']}개 × 프레임 {len(s)})")
        ok &= (err_i < 1e-12)
    return ok


# ---------------------------------------------------------------- [5]
def check_5_na():
    print("\n[5] 결측 규약 — 원소 없는 region -> NA · n_cluster=1 -> H=0")
    ok = True
    for facet, case_dir in REP.items():
        c = load_case(case_dir)
        g, lr = c["graph"], c["layer_region"]
        ts = REGION_ORDER.index("TS")
        ir_i, ru_i = 0, 1

        # TS를 전부 Ru로 → TS의 Ir은 NA
        s = c["s0"].copy()
        s[c["regions5"]["TS"]] = False
        r = compute_domain_frames(g, s[None, :], lr)
        na_ok = (np.isnan(r["D"][0, ts, ir_i]) and np.isnan(r["H"][0, ts, ir_i])
                 and r["n_cluster"][0, ts, ir_i] == 0
                 and not np.isnan(r["D"][0, ts, ru_i]))
        # 다른 region은 값이 있어야 한다
        bulk = REGION_ORDER.index("bulk")
        spill_ok = not np.isnan(r["D"][0, bulk, ir_i])
        h_ok = bool(np.all(r["H"][0][r["n_cluster"][0] == 1] == 0.0))
        print(f"  ({facet}) 원소 없는 region NA {na_ok}  NA 번짐 없음 {spill_ok}  "
              f"n_cluster=1 -> H=0 {h_ok}")
        ok &= na_ok and spill_ok and h_ok
    return ok


# ---------------------------------------------------------------- [6]
def check_6_v1_pathology():
    print("\n[6] v1 비교 — cutoff 3.4 A 그래프의 사슬·고립 원자가 k=4 로 없어지는가")
    ok = True
    for facet, case_dir in REP.items():
        c = load_case(case_dir)
        g, lr = c["graph"], c["layer_region"]
        s = c["s0"]
        symbols = np.where(s, "Ir", "Ru")

        # v1: region 안 cutoff 3.4 Å 그래프
        v1 = {}
        for name in ("TS", "TSS"):
            mask = c["regions5"][name]
            pos_r = c["positions"][mask]
            pairs = get_cation_pairs_within_cutoff(pos_r, c["atoms"].cell,
                                                   c["atoms"].pbc, V1_CUTOFF)
            v1[name] = (compute_ddom(symbols[mask], pairs)[0], len(pairs))

        r = compute_domain_frames(g, s[None, :], lr)
        ts, tss = REGION_ORDER.index("TS"), REGION_ORDER.index("TSS")
        new_ts = (r["D"][0, ts, 0], r["D"][0, ts, 1])

        if facet == "111":
            # (111): cutoff 안 pair 0개 → 모든 원자 고립
            path_ok = (v1["TS"][1] == 0 and v1["TSS"][1] == 0)
            fixed_ok = int(g["node_deg"].min()) >= DEFAULT_K
            print(f"  (111) v1 표면 pair {v1['TS'][1]}개 -> D_dom {v1['TS'][0]:.3f} "
                  f"(병리 재현 {path_ok})   k=4 최소 degree {int(g['node_deg'].min())} "
                  f"-> 고립 없음 {fixed_ok}   새 D(Ir,Ru) = "
                  f"({new_ts[0]:.3f}, {new_ts[1]:.3f})")
            ok &= path_ok and fixed_ok
        else:
            # (100)/(110): 면내 이웃 2개 → 1차원 사슬
            deg_v1 = 2 * v1["TS"][1] / max(int(c["regions5"]["TS"].sum()), 1)
            chain_ok = deg_v1 <= 2.01
            print(f"  ({facet}) v1 표면 pair {v1['TS'][1]:4d}개 -> 원자당 {deg_v1:.2f} "
                  f"(사슬 재현 {chain_ok})   k=4 degree "
                  f"{int(g['node_deg'].min())}~{int(g['node_deg'].max())}   새 D(Ir,Ru) = "
                  f"({new_ts[0]:.3f}, {new_ts[1]:.3f})")
            ok &= chain_ok
    return ok


# ---------------------------------------------------------------- [7]
def check_7_regression():
    print("\n[7] 회귀 — 단계 2·3 산출물이 한 자리도 안 바뀌었는가")
    if not (OUTDIR / "per_frame.csv").exists():
        print(f"  SKIP: {OUTDIR.name}/ 에 아직 결과가 없다 (배치를 먼저 돌린다)")
        return None
    if not (PREV_OUTDIR / "per_frame.csv").exists():
        print(f"  SKIP: 기준값 {PREV_OUTDIR.name}/ 이 없다 (내부 회귀 확인용 — 공유본에서는 정상)")
        return None

    ok = True
    for fname, keys in (("per_frame.csv", ["case_id", "frame"]),
                        ("sro_long.csv", None),
                        ("shell_pairs.csv", None)):
        old = pd.read_csv(PREV_OUTDIR / fname, float_precision="round_trip")
        new = pd.read_csv(OUTDIR / fname, float_precision="round_trip")
        common = [c for c in old.columns if c in new.columns]
        if keys:
            old = old.sort_values(keys).reset_index(drop=True)
            new = new.sort_values(keys).reset_index(drop=True)
        same_rows = len(old) == len(new)
        equal = same_rows and old[common].equals(new[common])
        added = [c for c in new.columns if c not in old.columns]
        print(f"  {fname:16s} 행 {len(old)} vs {len(new)}  공통칼럼 {len(common)}개 일치 {equal}"
              + (f"  (새 칼럼 {len(added)}개)" if added else ""))
        if not equal and same_rows:
            for col in common:
                if not old[col].equals(new[col]):
                    print(f"        <<< 불일치 칼럼: {col}")
        ok &= equal
    return ok


# ---------------------------------------------------------------- [8]
def _cluster_sizes(graph, s, layer, is_ir):
    """한 층에서 한 원소의 덩어리 크기 목록 (내림차순). 고립 원자도 크기 1로 센다."""
    nodes = np.where((graph["node_layer"] == layer) & (s == is_ir))[0]
    if len(nodes) == 0:
        return []
    m = (np.isin(graph["edge_i"], nodes) & np.isin(graph["edge_j"], nodes)
         & (graph["edge_layer"] == layer))
    loc = {g: i for i, g in enumerate(nodes)}
    rows = [loc[a] for a in graph["edge_i"][m]]
    cols = [loc[b] for b in graph["edge_j"][m]]
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(nodes),) * 2)
    n_comp, labels = connected_components(adj, directed=False)
    return sorted(np.bincount(labels, minlength=n_comp).tolist(), reverse=True)


def _percolation_replica(graph, s, layer, is_ir, reach=2):
    """층을 (2·reach+1)² 복제 셀로 깔고, 원자가 자기 자신의 한 칸 옆 복제본과 연결되면 그 방향으로 관통."""
    nodes = np.where((graph["node_layer"] == layer) & (s == is_ir))[0]
    n = len(nodes)
    if n == 0:
        return (False, False)
    m = (np.isin(graph["edge_i"], nodes) & np.isin(graph["edge_j"], nodes)
         & (graph["edge_layer"] == layer))
    loc = {g: i for i, g in enumerate(nodes)}
    span = 2 * reach + 1
    index = lambda i, a, b: (a + reach) * span * n + (b + reach) * n + i
    rows, cols = [], []
    for u, v, off_a, off_b in zip(graph["edge_i"][m], graph["edge_j"][m],
                                  graph["edge_off"][m][:, 0], graph["edge_off"][m][:, 1]):
        iu, iv = loc[u], loc[v]
        for a in range(-reach, reach + 1):
            for b in range(-reach, reach + 1):
                a2, b2 = a + off_a, b + off_b
                if -reach <= a2 <= reach and -reach <= b2 <= reach:
                    rows.append(index(iu, a, b))
                    cols.append(index(iv, a2, b2))
    total = span * span * n
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(total, total))
    _, labels = connected_components(adj, directed=False)
    p_a = any(labels[index(i, 0, 0)] == labels[index(i, 1, 0)] for i in range(n))
    p_b = any(labels[index(i, 0, 0)] == labels[index(i, 0, 1)] for i in range(n))
    return (bool(p_a), bool(p_b))


def check_8_handcheck(frame=0, regions=("TS", "bulk")):
    """층별 덩어리 크기 목록에 D · n_cluster · H · I_R · P 식을 직접 적용해 per_frame.csv와 비교한다."""
    print("\n[8] 손계산 대조 — 덩어리 크기 목록에서 스펙 식을 직접 적용")
    if not (OUTDIR / "per_frame.csv").exists():
        print(f"  SKIP: {OUTDIR.name}/ 에 아직 결과가 없다 (배치를 먼저 돌린다)")
        return None
    per_frame = pd.read_csv(OUTDIR / "per_frame.csv", low_memory=False)

    ok = True
    for facet, case_dir in REP.items():
        c = load_case(case_dir)
        g, lr = c["graph"], c["layer_region"]
        runs = sorted(case_dir.glob("run_*/accepted_states.xyz"))
        if not runs:
            print(f"  ({facet}) SKIP: accepted_states.xyz 없음")
            continue
        atoms = next(iter(iread(str(runs[0]), index=f"{frame}:{frame + 1}")))
        symbols = np.array(atoms.get_chemical_symbols())
        cat = np.where(np.isin(symbols, ("Ir", "Ru")))[0]
        s = symbols[cat] == "Ir"
        sel = per_frame[(per_frame.case_id == case_dir.name) & (per_frame.frame == frame)]
        if len(sel) != 1:
            print(f"  ({facet}) SKIP: per_frame 에 frame {frame} 행이 없다")
            continue
        row = sel.iloc[0]

        print(f"  ({facet}) {case_dir.name}  frame {frame}"
              f"  층당 {int(g['layer_n_nodes'][0])} site · 간선/층 {int(g['layer_n_edges'][0])}")
        for region in regions:
            layers = [l for l in range(g["n_layers"]) if lr[l] == region]
            for elem, is_ir in (("Ir", True), ("Ru", False)):
                by_layer = [_cluster_sizes(g, s, l, is_ir) for l in layers]
                sizes = [x for part in by_layer for x in part]
                n_atoms = sum(sizes)
                if n_atoms == 0:            # NA ([5]에서 확인)
                    continue
                n_cl = len(sizes)
                num = sum(max(part) for part in by_layer if part)
                d_hand = num / n_atoms
                p = np.array(sizes) / n_atoms
                h_hand = 0.0 if n_cl == 1 else float(-(p * np.log(p)).sum() / np.log(n_cl))
                pa_hand, pb_hand = (False, False)
                for l in layers:
                    a, b = _percolation_replica(g, s, l, is_ir)
                    pa_hand, pb_hand = pa_hand or a, pb_hand or b

                d_csv, h_csv = row[f"D_{elem}_{region}"], row[f"H_{elem}_{region}"]
                n_csv = int(row[f"ncl_{elem}_{region}"])
                pa_csv, pb_csv = bool(row[f"Pa_{elem}_{region}"]), bool(row[f"Pb_{elem}_{region}"])
                good = (abs(d_hand - d_csv) < 1e-12 and abs(h_hand - h_csv) < 1e-12
                        and n_cl == n_csv and pa_hand == pa_csv and pb_hand == pb_csv)
                ok &= good
                shown = by_layer if len(by_layer) <= 3 else by_layer[:3] + ["..."]
                print(f"      {region:5s} {elem}  덩어리 {shown}"
                      f"  D={num}/{n_atoms}={d_hand:.6f}  n={n_cl}  H={h_hand:.6f}"
                      f"  P=({pa_hand},{pb_hand})   {'OK' if good else '<<< 불일치'}")
                if not good:
                    print(f"            CSV: D={d_csv:.6f} n={n_csv} H={h_csv:.6f} "
                          f"P=({pa_csv},{pb_csv})")

            edge_sel = np.isin(g["edge_layer"], layers)
            hetero = int((s[g["edge_i"][edge_sel]] != s[g["edge_j"][edge_sel]]).sum())
            total = int(edge_sel.sum())
            i_hand = hetero / total
            i_csv = row[f"I_{region}"]
            good = abs(i_hand - i_csv) < 1e-12
            ok &= good
            print(f"      {region:5s} I_R = 이종 {hetero} / 전체 {total} = {i_hand:.6f}"
                  f"   {'OK' if good else f'<<< 불일치 (CSV {i_csv:.6f})'}")
    return ok


def main():
    print(f"데이터: {MC_DIR}")
    print(f"케이스 {len(CASES)}개 · 대표 {list(REP)} · "
          f"k={DEFAULT_K} tie_tol={DEFAULT_TIE_TOL} r_cut={DEFAULT_R_CUT}")
    results = [
        ("[1] 그래프 재구성", check_1_graph()),
        ("[2] 인공 배열", check_2_synthetic()),
        ("[3] 항등식", check_3_identities()),
        ("[4] I_R 독립 재계산", check_4_interface_density()),
        ("[5] 결측 규약", check_5_na()),
        ("[6] v1 병리 재현", check_6_v1_pathology()),
        ("[7] 회귀", check_7_regression()),
        ("[8] 손계산 대조", check_8_handcheck()),
    ]
    print("\n" + "=" * 70)
    for name, ok in results:
        print(f"  {name:26s} {'SKIP' if ok is None else ('PASS' if ok else 'FAIL')}")
    print("=" * 70)
    return 0 if all(ok is not False for _, ok in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
