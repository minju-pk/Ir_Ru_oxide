#!/usr/bin/env python3
"""
MC 30 케이스에 descriptors_v2를 적용해 "케이스 × 프레임 × descriptor" 표를 만든다.

입력 : ../Ir_Ru_oxide/correct/MC_final_30cases_20260806/mc_runs/<case>/
         surface.xyz (초기 구조) · run_*/accepted_states.xyz (수락된 프레임) · run_*/results_every_step.json
출력 : results/
  per_frame.csv     프레임 = 한 행. 깊이별 조성 · region 수준 domain morphology · x_layer(';'로 이은 문자열)
  layer_profile.csv (케이스, 프레임, 층) = 한 행. x_layer
  sro_long.csv      (케이스, 프레임, entry) = 한 행. α_in · α_in_region · α_out
  shell_pairs.csv   케이스의 껍질 pair 목록
  domain_layer.csv  (케이스, 프레임, 층, 원소) = 한 행. 층별 덩어리 · percolation
  domain_graph.csv  케이스의 domain graph (층별)
  run_params.json   실행 설정

층 라벨 · region · 껍질 pair 목록 · domain graph는 surface.xyz에서 한 번 정해 모든 프레임에 고정한다.
출력 폴더에 결과가 있으면 덮어쓰지 않고 멈춘다.
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import iread, read

HERE = Path(__file__).resolve().parent   # v2/
PROJECT_ROOT = HERE.parent               # 원본 데이터 위치
sys.path.insert(0, str(HERE))
from descriptors_v2.depth import compute_depth_frames  # noqa: E402
from descriptors_v2.regions import compute_layer_id, get_regions, find_vacuum_axis  # noqa: E402
from descriptors_v2.sro import (build_shell_pairs, compute_alpha_frames,  # noqa: E402
                                layer_region_map, REGION_ORDER)
from descriptors_v2.domain import (build_domain_graph, compute_domain_frames,  # noqa: E402
                                   ELEM_ORDER)

MC_DIR = PROJECT_ROOT / "Ir_Ru_oxide" / "correct" / "MC_final_30cases_20260806" / "mc_runs"
OUTDIR = HERE / "results"

PARAMS = dict(
    cation_species=("Ir", "Ru"),
    n_surf_layers=1,             # 표면 = 양 끝 1층
    n_sub_layers=1,              # 차표면 = 그 안쪽 1층
    vacuum_gap_threshold=5.0,    # Å
    layer_gap_threshold=0.5,     # Å. 이 간격 이내 = 같은 층
)

# α 껍질. sro_long.csv의 in_spec = 면내 ≤ 6.0 Å · Δlayer = 1 인 entry
SRO_PARAMS = dict(
    r_max_inplane=6.6,   # Å. (111) 면내 2번째 껍질(6.41 Å)까지
    r_max_inter=6.0,     # Å
    tol=0.18,            # Å
    n_shell=2,
    dlayer_max=3,        # (111) 최근접 cation(3.15 Å)이 Δlayer = 2
)

DOMAIN_PARAMS = dict(
    k=4,
    tie_tol=0.18,        # Å. k번째 이웃과 이 거리 안이면 모두 포함 ((110)만 해당, 0.0 = strict)
    r_cut=9.5,           # Å. 이웃 후보 반경
)

# 폴더명 접미사: (100)/(110)은 _zaligned, (111)은 _normalz
CASE_RE = re.compile(
    r"x0p(?P<x>\d+)_(?P<facet>\d+)_(?P<surf_id>S\d+)_(?P<o_condition>[A-Za-z]+)"
    r"_dO(?P<dO>[+-]?\d+)_N_(?P<n_atoms>\d+)_(?P<align>zaligned|normalz)"
)


def parse_case_name(name):
    """폴더명 -> 조성 · facet · O 조건 등. 구조에서 센 값과 대조하는 데 쓴다."""
    m = CASE_RE.search(name)
    if m is None:
        return None
    return dict(
        x_Ir_nominal=int(m.group("x")) / 100.0,
        facet=m.group("facet"),
        surf_id=m.group("surf_id"),
        o_condition=m.group("o_condition"),
        dO=int(m.group("dO")),
        n_atoms_nominal=int(m.group("n_atoms")),
        align=m.group("align"),
    )


def load_step_index(run_dir):
    """수락된 프레임마다 (global_step, stage, step_from, step_to).

    step_from ~ step_to = 그 구조가 점유한 MC 스텝 구간 (다음 수락 직전까지, 마지막은 런 끝까지).
    """
    path = run_dir / "results_every_step.json"
    if not path.exists():
        return None
    try:
        records = json.load(open(path))["records"]
    except Exception as exc:
        print(f"    [warn] results_every_step.json 읽기 실패: {exc}")
        return None

    accepted = [(r.get("global_step"), r.get("stage")) for r in records if r.get("accepted")]
    if not accepted:
        return []
    last_step = max(r.get("global_step") or 0 for r in records)

    steps = [s for s, _ in accepted]
    out = []
    for i, (step, stage) in enumerate(accepted):
        step_to = steps[i + 1] - 1 if i + 1 < len(steps) else last_step
        out.append((step, stage, step, step_to))
    return out


def build_sro_tables(case_name, run_id, frame_ids, shells, layer_region, res):
    """α 결과 -> (sro_long 행, shell_pairs 행).

    quantity: alpha_in (층 하나) · alpha_in_region (region) · alpha_out (층 쌍 l, l2)
    """
    n_frames, n_entries = res['alpha'].shape
    frame_col = np.repeat(np.asarray(frame_ids), n_entries)

    def tile(arr):
        return np.tile(np.asarray(arr), n_frames)

    is_in = shells['is_in']
    body = pd.DataFrame({
        'case_id': case_name, 'run_id': run_id, 'frame': frame_col,
        'quantity': tile(np.where(is_in, 'alpha_in', 'alpha_out')),
        'region': tile([layer_region[x] if in_ else '' for x, in_ in zip(shells['l'], is_in)]),
        'l': tile(shells['l']), 'l2': tile(shells['l2']),
        'dlayer': tile(shells['dlayer']), 'm': tile(shells['m']),
        'shell_r': tile(shells['shell_r']), 'n_pairs': tile(shells['n_pairs']),
        'f_obs': res['f_obs'].ravel(), 'f_0': res['f_0'].ravel(),
        'alpha': res['alpha'].ravel(), 'valid': res['valid'].ravel(),
        'in_spec': tile(shells['in_spec']),
    })

    # region 수준 행. 층 번호가 없으므로 l/l2 = -1
    n_region_cols = res['region_alpha'].shape[1]
    reg_r, reg_n, reg_spec = [], [], []
    for region, m in zip(res['region_names'], res['region_m']):
        sel = np.where(is_in & (shells['m'] == m)
                       & np.array([layer_region[x] == region for x in shells['l']]))[0]
        w = shells['n_pairs'][sel].astype(float)
        r = shells['shell_r'][sel]
        ok = np.isfinite(r) & (w > 0)
        # 껍질 거리 = pair 수 가중 평균
        reg_r.append(float((r[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan)
        reg_n.append(float(w.sum()))
        reg_spec.append(bool(len(sel)) and bool(shells['in_spec'][sel].all()))

    region_body = pd.DataFrame({
        'case_id': case_name, 'run_id': run_id,
        'frame': np.repeat(np.asarray(frame_ids), n_region_cols),
        'quantity': 'alpha_in_region',
        'region': np.tile(res['region_names'], n_frames),
        'l': -1, 'l2': -1, 'dlayer': 0,
        'm': np.tile(res['region_m'], n_frames),
        'shell_r': np.tile(reg_r, n_frames),
        'n_pairs': np.tile(reg_n, n_frames),
        'f_obs': res['region_f_obs'].ravel(), 'f_0': res['region_f_0'].ravel(),
        'alpha': res['region_alpha'].ravel(), 'valid': res['region_valid'].ravel(),
        'in_spec': np.tile(reg_spec, n_frames),
    })

    shell_table = pd.DataFrame({
        'case_id': case_name,
        'kind': np.where(is_in, 'in', 'out'),
        'region': [layer_region[x] if in_ else '' for x, in_ in zip(shells['l'], is_in)],
        'l': shells['l'], 'l2': shells['l2'], 'dlayer': shells['dlayer'],
        'm': shells['m'], 'shell_r': shells['shell_r'], 'shell_w': shells['shell_w'],
        'n_pairs': shells['n_pairs'], 'in_spec': shells['in_spec'],
        'n_shell_found': shells['n_shell_found'],
        'n_shell_found_spec': shells['n_shell_found_spec'],
    })

    return pd.concat([body, region_body], ignore_index=True), shell_table


def build_domain_tables(case_name, run_id, frame_ids, graph, layer_region, res):
    """domain 결과 -> (per_frame에 붙일 region 열, domain_layer 행, domain_graph 행)."""
    n_frames = len(frame_ids)
    n_layers = graph["n_layers"]

    # region 수준 (wide)
    cols = {"frame": np.asarray(frame_ids)}
    for ri, region in enumerate(REGION_ORDER):
        for ei, elem in enumerate(ELEM_ORDER):
            cols[f"D_{elem}_{region}"] = res["D"][:, ri, ei]
            cols[f"ncl_{elem}_{region}"] = res["n_cluster"][:, ri, ei]
            cols[f"H_{elem}_{region}"] = res["H"][:, ri, ei]
            # 방향 a, b = 면내 격자벡터
            cols[f"Pa_{elem}_{region}"] = res["P"][:, ri, ei, 0]
            cols[f"Pb_{elem}_{region}"] = res["P"][:, ri, ei, 1]
            cols[f"Pany_{elem}_{region}"] = res["P"][:, ri, ei].any(axis=-1)
        cols[f"I_{region}"] = res["I"][:, ri]
        cols[f"nedge_{region}"] = int(res["region_n_edges"][ri])
    region_df = pd.DataFrame(cols)

    # 층 수준 (long)
    n_rows = n_frames * n_layers * len(ELEM_ORDER)
    frame_col = np.repeat(np.asarray(frame_ids), n_layers * len(ELEM_ORDER))
    layer_col = np.tile(np.repeat(np.arange(n_layers), len(ELEM_ORDER)), n_frames)
    elem_col = np.tile(np.array(ELEM_ORDER), n_frames * n_layers)
    layer_df = pd.DataFrame({
        "case_id": case_name, "run_id": run_id,
        "frame": frame_col, "layer": layer_col,
        "region": np.tile(np.repeat([layer_region[l] for l in range(n_layers)],
                                    len(ELEM_ORDER)), n_frames),
        "element": elem_col,
        "n_a": res["layer_n_a"].reshape(n_rows),
        "max_cluster": res["layer_max"].reshape(n_rows),
        "n_cluster": res["layer_ncl"].reshape(n_rows),
        "P_a": res["layer_perc"][..., 0].reshape(n_rows),
        "P_b": res["layer_perc"][..., 1].reshape(n_rows),
        # 원소와 무관한 층 값 — 두 원소 행에 같은 값
        "I_layer": np.repeat(res["layer_I"].reshape(n_frames * n_layers), len(ELEM_ORDER)),
        "n_edges_layer": np.tile(np.repeat(res["layer_n_edges"], len(ELEM_ORDER)), n_frames),
    })

    # 그래프 (프레임 무관)
    graph_df = pd.DataFrame({
        "case_id": case_name,
        "layer": np.arange(n_layers),
        "region": [layer_region[l] for l in range(n_layers)],
        "n_nodes": graph["layer_n_nodes"],
        "n_edges": graph["layer_n_edges"],
        "n_edges_strict": graph["layer_n_edges_strict"],
        "deg_min": graph["layer_deg_min"], "deg_max": graph["layer_deg_max"],
        "deg_mean": graph["layer_deg_mean"],
        "d_k": graph["layer_d4"], "d_k1": graph["layer_d5"],
        "n_tie": graph["layer_n_tie"],
        "k": graph["k"], "tie_tol": graph["tie_tol"],
    })
    return region_df, layer_df, graph_df


def process_case(case_dir):
    name = case_dir.name
    meta = parse_case_name(name)
    if meta is None:
        print(f"  [skip] 폴더명 형식 불일치: {name}")
        return None

    runs = sorted(case_dir.glob("run_*/accepted_states.xyz"))
    if not runs:
        print(f"  [skip] accepted_states.xyz 없음: {name}")
        return None
    if len(runs) > 1:
        print(f"    [warn] run 폴더가 {len(runs)}개다. 첫 번째만 쓴다: {runs[0].parent.name}")
    traj_path = runs[0]
    run_dir = traj_path.parent
    steps = load_step_index(run_dir)

    # 초기 구조에서 한 번 정해 모든 프레임에 고정
    ref = read(str(case_dir / "surface.xyz"))
    ref_cat = np.where(np.isin(ref.get_chemical_symbols(), PARAMS["cation_species"]))[0]
    ref_positions = ref.get_positions()[ref_cat]
    axis = find_vacuum_axis(ref, PARAMS["vacuum_gap_threshold"])
    layer_id = compute_layer_id(ref, ref_cat, PARAMS["vacuum_gap_threshold"],
                                PARAMS["layer_gap_threshold"])
    regions5 = get_regions(ref, layer_id, axis=axis, n_surf_layers=PARAMS["n_surf_layers"],
                           n_sub_layers=PARAMS["n_sub_layers"])
    layer_region = layer_region_map(layer_id, regions5)
    shells = build_shell_pairs(ref_positions, ref.cell, ref.pbc, layer_id, **SRO_PARAMS)
    domain_graph = build_domain_graph(ref_positions, ref.cell, ref.pbc, layer_id, axis,
                                      **DOMAIN_PARAMS)

    # 프레임마다 메타데이터와 occupation만 읽는다
    rows, s_frames = [], []
    t0 = time.time()
    for i, atoms in enumerate(iread(str(traj_path), index=":")):
        symbols = np.array(atoms.get_chemical_symbols())
        n_ir = int((symbols == "Ir").sum())
        n_ru = int((symbols == "Ru").sum())
        n_cat = n_ir + n_ru
        if n_cat != len(layer_id):
            raise ValueError(f"프레임 {i}의 cation {n_cat}개 ≠ surface.xyz의 {len(layer_id)}개")

        row = dict(case_id=name, run_id=run_dir.name, frame=i, **meta)
        row["n_atoms_actual"] = len(atoms)
        row["n_Ir"] = n_ir
        row["n_Ru"] = n_ru
        row["x_Ir_actual"] = n_ir / n_cat
        row["name_matches_content"] = bool(
            len(atoms) == meta["n_atoms_nominal"]
            and abs(n_ir / n_cat - meta["x_Ir_nominal"]) < 0.02
        )
        try:
            row["energy"] = float(atoms.get_potential_energy())
        except Exception:
            row["energy"] = np.nan
        if steps is not None and i < len(steps):
            (row["global_step"], row["stage"],
             row["step_from"], row["step_to"]) = steps[i]
        else:
            row["global_step"], row["stage"] = np.nan, None
            row["step_from"], row["step_to"] = np.nan, np.nan
        rows.append(row)
        s_frames.append(symbols[np.isin(symbols, PARAMS["cation_species"])]
                        == PARAMS["cation_species"][0])

    df = pd.DataFrame(rows)
    s_arr = np.array(s_frames)
    frame_ids = np.arange(len(s_arr))

    # 깊이별 조성
    depth = compute_depth_frames(s_arr, layer_id, regions5)
    x_layer = depth.pop("x_layer")
    df["x_layer"] = [";".join(repr(float(v)) for v in xl) for xl in x_layer]
    for col, val in depth.items():
        df[col] = val
    n_layers = x_layer.shape[1]
    layer_df = pd.DataFrame({
        "case_id": name, "run_id": run_dir.name,
        "frame": np.repeat(frame_ids, n_layers),
        "layer": np.tile(np.arange(n_layers), len(frame_ids)),
        "x": x_layer.ravel(),
    })

    # 단거리 ordering · domain morphology
    res = compute_alpha_frames(shells, s_arr, layer_id, layer_region=layer_region)
    sro_df, shell_df = build_sro_tables(name, run_dir.name, frame_ids, shells, layer_region, res)
    res_dom = compute_domain_frames(domain_graph, s_arr, layer_region)
    region_df, domain_layer_df, domain_graph_df = build_domain_tables(
        name, run_dir.name, frame_ids, domain_graph, layer_region, res_dom)
    df = df.merge(region_df, on="frame", how="left")
    dt = time.time() - t0

    n_bad = int((~df["name_matches_content"]).sum())
    flag = f"  <<< 폴더명 불일치 {n_bad} 프레임" if n_bad else ""
    if steps is not None and len(steps) != len(df):
        flag += f"  <<< step 개수 불일치 (accepted 기록 {len(steps)} vs 프레임 {len(df)}) -> global_step 신뢰 불가"
    print(f"  {name:52s} {len(df):4d} frames  {dt:5.1f}s{flag}")
    return df, layer_df, sro_df, shell_df, domain_layer_df, domain_graph_df


def main():
    names = ("per_frame.csv", "layer_profile.csv", "sro_long.csv", "shell_pairs.csv",
             "domain_layer.csv", "domain_graph.csv")
    existing = [f for f in names if (OUTDIR / f).exists()]
    if existing:
        print(f"[중단] {OUTDIR.name}/ 에 이미 결과가 있다: {', '.join(existing)}\n"
              f"       덮어쓰지 않는다. 다시 만들려면 그 폴더를 옮기거나 지우고 실행할 것.")
        return 1

    OUTDIR.mkdir(exist_ok=True)
    case_dirs = sorted(d for d in MC_DIR.glob("ir_ru_sqs_*") if d.is_dir())
    print(f"케이스 {len(case_dirs)}개, 파라미터 {PARAMS}")
    print(f"단거리 ordering 껍질 설정 {SRO_PARAMS}"
          f"  (스펙 문구 = r_max_inplane 6.0 · dlayer_max 1, in_spec 열로 필터)")
    print(f"domain graph 설정 {DOMAIN_PARAMS}"
          f"  (스펙 문구 = tie_tol 0.0. 동률은 (110)에만 있다)\n")

    results = []
    t0 = time.time()
    for case_dir in case_dirs:
        try:
            result = process_case(case_dir)
        except Exception as exc:
            print(f"  [FAIL] {case_dir.name}: {type(exc).__name__}: {exc}")
            continue
        if result is not None:
            results.append(result)

    if not results:
        print("\n처리된 케이스가 없다.")
        return 1

    # 케이스별 표 6개를 이어 붙여 쓴다
    tables = [pd.concat(parts, ignore_index=True) for parts in zip(*results)]
    for fname, table in zip(names, tables):
        table.to_csv(OUTDIR / fname, index=False)
    out, layer_out, sro_out, shell_out, dom_layer_out, dom_graph_out = tables

    json.dump(
        dict(
            params={k: list(v) if isinstance(v, tuple) else v for k, v in PARAMS.items()},
            mc_dir=str(MC_DIR.relative_to(PROJECT_ROOT)),
            n_cases=len(results),
            n_rows=len(out),
            generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            module="descriptors_v2",
            n_layer_rows=len(layer_out),
            sro_params=SRO_PARAMS,
            n_sro_rows=len(sro_out),
            n_shell_rows=len(shell_out),
            domain_params=DOMAIN_PARAMS,
            n_domain_layer_rows=len(dom_layer_out),
            n_domain_graph_rows=len(dom_graph_out),
            note=("descriptor 값만 담긴 표다. 평형 구간 선택/평균/플롯은 plot/ 스크립트가 한다. "
                  "x_layer는 layer_profile.csv(long)에 있고 per_frame.csv에는 "
                  "';'로 이은 문자열 칼럼으로도 들어 있다 - 값은 동일하다. "
                  "alpha_in/alpha_out은 sro_long.csv(long), 그 껍질 목록은 "
                  "shell_pairs.csv에 있다. 스펙 문구 그대로의 값은 in_spec==True 필터. "
                  "domain morphology의 region 수준(D/ncl/H/Pany/I/nedge)은 "
                  "per_frame.csv의 칼럼이고, 층 수준은 domain_layer.csv(long), "
                  "그래프 자체는 domain_graph.csv에 있다."),
        ),
        open(OUTDIR / "run_params.json", "w"),
        indent=2,
        ensure_ascii=False,
    )

    print(f"\n총 {len(results)}개 케이스, {len(out)}행, {time.time() - t0:.1f}s")
    if len(results) != len(case_dirs):
        print(f"  <<< 경고: 폴더 {len(case_dirs)}개 중 {len(results)}개만 처리됐다. "
              f"위의 [skip]/[FAIL] 줄을 확인한다 (폴더명 접미사 등).")
    print("  facet별 케이스 수:", out.groupby("facet")["case_id"].nunique().to_dict())
    for fname, table in zip(names, tables):
        print(f"  -> {OUTDIR / fname}  ({len(table)}행)")

    facet_of = out.groupby("case_id")["facet"].first()
    sro_out["facet"] = sro_out["case_id"].map(facet_of)
    shell_out["facet"] = shell_out["case_id"].map(facet_of)

    # valid fraction = N_valid / N_requested
    print("\nalpha valid fraction (facet x quantity) — 전체 / 스펙(in_spec)만:")
    for (facet, quantity), grp in sro_out.groupby(["facet", "quantity"]):
        spec = grp[grp["in_spec"]]
        spec_txt = (f"{spec['valid'].mean():.3f} ({len(spec):,}행)" if len(spec)
                    else "행 없음")
        print(f"  ({facet}) {quantity:16s} {grp['valid'].mean():.3f} "
              f"({len(grp):,}행)   스펙만: {spec_txt}")

    # 스펙 상한 6.0 Å 안에 껍질이 n_shell개 안 잡히는 (케이스, 층 쌍)
    print("\n스펙 상한 6.0 A 안에 껍질이 2개 안 잡히는 목록:")
    lists = shell_out.drop_duplicates(["case_id", "l", "l2"])
    short = lists[lists["n_shell_found_spec"] < SRO_PARAMS["n_shell"]]
    if len(short):
        print(short.groupby(["facet", "kind"]).size().to_string())
    else:
        print("  없음")

    # facet별 최근접 cation 껍질이 스펙 범위 안에 있는가
    print("\n각 facet의 최근접 cation-cation 껍질이 어디에 있는가:")
    have = shell_out[shell_out["n_pairs"] > 0]
    nearest = have.loc[have.groupby("facet")["shell_r"].idxmin()]
    for _, r in nearest.iterrows():
        mark = "스펙 안" if r["in_spec"] else "<<< 스펙 밖 — k=1만 보면 누락된다"
        print(f"  ({r['facet']}) {r['shell_r']:.3f} A  dlayer={r['dlayer']} "
              f"m={r['m']} kind={r['kind']}  {mark}")

    dom_graph_out["facet"] = dom_graph_out["case_id"].map(facet_of)
    print("\ndomain graph — facet별 (층 하나 기준):")
    print(f"  {'facet':7s} {'층당원자':>7s} {'간선/층':>8s} {'strict':>8s} "
          f"{'degree':>9s} {'동률 원자':>9s}")
    for facet, grp in dom_graph_out.groupby("facet"):
        tie = grp["n_tie"].sum() / max(grp["n_nodes"].sum(), 1)
        print(f"  ({facet})  {grp['n_nodes'].median():7.0f} {grp['n_edges'].mean():8.1f} "
              f"{grp['n_edges_strict'].mean():8.1f} "
              f"{grp['deg_min'].min():.0f}~{grp['deg_max'].max():<7.0f} {tie:9.1%}")

    print("\ndomain morphology — 표면(TS/BS) 평균 (전 프레임, NA 제외):")
    for facet, grp in out.groupby("facet"):
        parts = []
        for region in ("TS", "BS"):
            for elem in ELEM_ORDER:
                parts.append(f"{region}·{elem} D={grp[f'D_{elem}_{region}'].mean():.3f}"
                             f" P={grp[f'Pany_{elem}_{region}'].mean():.2f}")
        print(f"  ({facet}) " + "  ".join(parts))

    bad = out.loc[~out["name_matches_content"], "case_id"].unique()
    print(f"\n폴더명 불일치 케이스: {len(bad)}개" + (f" {list(bad)}" if len(bad) else ""))

    print("\nfacet별 region 분할 (첫 프레임 기준):")
    first = out[out["frame"] == 0]
    cols = ["n_atomic_layers", "n_TS", "n_BS", "n_TSS", "n_BSS", "n_bulk"]
    print(first.groupby(["facet", "surf_id"])[cols].first().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
