"""
단거리 ordering — α_in / α_out (스펙 <단거리 ordering>).

    f_IrRu(E)      = (1/|E|) Σ_{i,j∈E} |s_i − s_j|
    f_0,in(l)      = 2 N_Ir,l N_Ru,l / [N_l (N_l − 1)]
    α_in(l,m)      = 1 − f_IrRu(E_in(l,m)) / f_0,in(l)
    f_0,out(l,k)   = x_l (1 − x_k) + (1 − x_l) x_k
    α_out          = 1 − f_IrRu(E_out(l,k,m)) / f_0,out(l,k)
    α_in(R,m)      = 1 − f_obs(R,m) / f_0(R,m)    (관측·기댓값을 각각 |E_l,m|로 가중합한 뒤 나눈다)

- pair 목록은 초기 구조에서 한 번 만들어 고정하고, 프레임마다 occupation s만 바꾼다.
- α_out은 층 번호 쌍 (l, l2)로 기록한다.
- 스펙 범위(면내 ≤ 6.0 Å, Δlayer = 1)보다 넓게(면내 ≤ 6.6 Å, Δlayer ≤ 3) 계산하고
  각 entry에 in_spec을 붙인다. 스펙 범위의 값은 in_spec == True 로 걸러낸다.
- E가 비었거나 f_0 = 0이면 α는 NA.
"""
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

# 스펙 범위 (in_spec 판정에만 쓴다)
SPEC_R_MAX = 6.0
SPEC_DLAYER_MAX = 1

DEFAULT_TOL = 0.18        # 스펙: shell tolerance 0.18 Å
DEFAULT_N_SHELL = 2       # 스펙: cation shell 2개
REGION_ORDER = ('TS', 'BS', 'TSS', 'BSS', 'bulk')   # 스펙 x_region 성분 순서


def _find_shells(distances, tol):
    """거리를 정렬해 간격이 tol을 넘는 곳에서 껍질을 나눈다.

    반환: (중심, 폭, 시작 index, 끝 index, 정렬 순서). 껍질은 거리 오름차순.
    """
    order = np.argsort(distances, kind='stable')
    d = distances[order]
    cut = np.where(np.diff(d) > tol)[0] + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut, [len(d)]])
    centers = np.array([d[a:b].mean() for a, b in zip(starts, ends)])
    widths = np.array([d[b - 1] - d[a] for a, b in zip(starts, ends)])
    return centers, widths, starts, ends, order


def build_shell_pairs(positions, cell, pbc, layer_id,
                      r_max_inplane=6.6, r_max_inter=6.0, tol=DEFAULT_TOL,
                      n_shell=DEFAULT_N_SHELL, dlayer_max=3):
    """초기 구조의 cation 좌표에서 (층 쌍 × 껍질)별 pair 목록을 만든다.

    positions, cell, pbc : 초기 구조의 cation 좌표와 셀
    layer_id             : cation 층 라벨
    r_max_inplane        : 같은 층 pair 상한 (스펙 6.0)
    r_max_inter          : 다른 층 pair 상한 (스펙 6.0)
    dlayer_max           : 볼 층 간격 상한 (스펙 1)

    반환 dict — entry(층 쌍 × 껍질 m) G개:
      pair_i, pair_j      (P,)   entry 순서로 이어 붙인 pair
      pair_start          (G+1,) entry별 시작 offset
      is_in, l, l2, dlayer, m, shell_r, shell_w, n_pairs, in_spec   (G,)
      n_shell_found, n_shell_found_spec  (G,) 찾은 껍질 수 (전체 / 6.0 Å 이내)
    m은 목록(층 쌍)마다 따로 매긴다 — 같은 m이라도 facet마다 거리가 다르므로 shell_r을 함께 본다.
    pair가 있는 목록에서 껍질이 n_shell개보다 적으면 모자란 m은 n_pairs = 0 행(NA)으로 남긴다.
    """
    positions = np.asarray(positions, dtype=float)
    layer_id = np.asarray(layer_id, dtype=int)
    if len(positions) != len(layer_id):
        raise ValueError(f"좌표 {len(positions)}개와 층 라벨 {len(layer_id)}개가 다르다.")

    probe = Atoms(numbers=np.ones(len(positions), dtype=int),
                  positions=positions, cell=cell, pbc=pbc)
    i, j, d = neighbor_list('ijd', probe, max(r_max_inplane, r_max_inter))
    upper = i < j
    i, j, d = i[upper], j[upper], d[upper]

    dlayer_all = np.abs(layer_id[i] - layer_id[j])
    r_limit = np.where(dlayer_all == 0, r_max_inplane, r_max_inter)
    keep = (dlayer_all <= dlayer_max) & (d <= r_limit)
    i, j, d, dlayer_all = i[keep], j[keep], d[keep], dlayer_all[keep]

    # 층 쌍 (l <= l2)으로 목록을 나눈다. E_in은 l == l2
    la = np.minimum(layer_id[i], layer_id[j])
    lb = np.maximum(layer_id[i], layer_id[j])
    keys = np.unique(np.stack([la, lb], axis=1), axis=0)

    pair_i, pair_j, pair_start = [], [], [0]
    cols = {k: [] for k in ('is_in', 'l', 'l2', 'dlayer', 'm', 'shell_r', 'shell_w',
                            'n_pairs', 'in_spec', 'n_shell_found', 'n_shell_found_spec')}

    for key_l, key_l2 in keys:
        sel = np.where((la == key_l) & (lb == key_l2))[0]
        centers, widths, starts, ends, order = _find_shells(d[sel], tol)
        n_found = len(centers)
        n_found_spec = int(np.sum(centers <= SPEC_R_MAX))
        dl = int(key_l2 - key_l)

        for m in range(1, n_shell + 1):
            if m <= n_found:
                idx = sel[order[starts[m - 1]:ends[m - 1]]]
                pair_i.append(i[idx])
                pair_j.append(j[idx])
                pair_start.append(pair_start[-1] + len(idx))
                shell_r, shell_w, n_p = centers[m - 1], widths[m - 1], len(idx)
                in_spec = bool(shell_r <= SPEC_R_MAX and dl <= SPEC_DLAYER_MAX)
            else:
                # 껍질이 모자란 슬롯 (NA)
                pair_start.append(pair_start[-1])
                shell_r, shell_w, n_p = np.nan, np.nan, 0
                in_spec = bool(dl <= SPEC_DLAYER_MAX)

            cols['is_in'].append(dl == 0)
            cols['l'].append(int(key_l))
            cols['l2'].append(int(key_l2))
            cols['dlayer'].append(dl)
            cols['m'].append(m)
            cols['shell_r'].append(shell_r)
            cols['shell_w'].append(shell_w)
            cols['n_pairs'].append(n_p)
            cols['in_spec'].append(in_spec)
            cols['n_shell_found'].append(n_found)
            cols['n_shell_found_spec'].append(n_found_spec)

    out = {k: np.array(v) for k, v in cols.items()}
    out['pair_i'] = (np.concatenate(pair_i) if pair_i else np.empty(0, dtype=int))
    out['pair_j'] = (np.concatenate(pair_j) if pair_j else np.empty(0, dtype=int))
    out['pair_start'] = np.array(pair_start, dtype=int)
    out['n_entries'] = len(out['m'])
    out['n_shell'] = n_shell
    return out


def layer_region_map(layer_id, regions5):
    """층 번호 -> region 이름 배열. 한 층이 두 region에 걸치면 ValueError."""
    layer_id = np.asarray(layer_id, dtype=int)
    n_layers = int(layer_id.max()) + 1
    mapping = np.empty(n_layers, dtype=object)
    for layer in range(n_layers):
        mask = layer_id == layer
        names = [name for name, region_mask in regions5.items() if (region_mask & mask).any()]
        if len(names) != 1:
            raise ValueError(f"층 {layer}이 region {names}에 걸쳐 있다. region 분할이 층 단위가 아니다.")
        mapping[layer] = names[0]
    return mapping


def compute_alpha_frames(shells, s_frames, layer_id, layer_region=None):
    """프레임 묶음의 α_in / α_out (프레임 축 벡터화).

    shells       : build_shell_pairs 결과
    s_frames     : (F, N) occupation, 1 = Ir, 0 = Ru
    layer_id     : (N,) 층 라벨 (shells와 같은 것)
    layer_region : layer_region_map 결과. 주면 region 수준 α_in도 낸다

    반환: f_obs · f_0 · alpha · valid (F, G)
          [layer_region을 주면] region_f_obs · region_f_0 · region_alpha · region_valid ·
          region_n_pairs (F, 5·n_shell), region_names · region_m
    """
    s = np.asarray(s_frames, dtype=np.float64)
    if s.ndim == 1:
        s = s[None, :]
    layer_id = np.asarray(layer_id, dtype=int)
    n_frames = s.shape[0]
    n_entries = shells['n_entries']
    n_pairs = shells['n_pairs'].astype(float)

    # f_IrRu: entry별 이종 pair 비율
    hetero = np.abs(s[:, shells['pair_i']] - s[:, shells['pair_j']])   # (F, P)
    if hetero.shape[1]:
        # 마지막 entry가 비어 offset이 P를 가리킬 수 있으므로 0 한 칸을 덧붙인다
        padded = np.concatenate([hetero, np.zeros((n_frames, 1))], axis=1)
        sums = np.add.reduceat(padded, shells['pair_start'][:-1], axis=1)
    else:
        sums = np.zeros((n_frames, n_entries))
    # reduceat는 빈 entry에 이웃 값을 넣으므로 0으로 덮는다
    sums[:, n_pairs == 0] = 0.0
    with np.errstate(invalid='ignore', divide='ignore'):
        f_obs = np.where(n_pairs > 0, sums / n_pairs, np.nan)

    # 층별 조성
    n_layers = int(layer_id.max()) + 1
    onehot = np.zeros((len(layer_id), n_layers))
    onehot[np.arange(len(layer_id)), layer_id] = 1.0
    n_ir = s @ onehot                      # (F, K)
    n_l = onehot.sum(axis=0)               # (K,)
    n_ru = n_l[None, :] - n_ir
    with np.errstate(invalid='ignore', divide='ignore'):
        x_layer = n_ir / n_l[None, :]
        f0_in_layer = 2.0 * n_ir * n_ru / (n_l * (n_l - 1.0))[None, :]

    # entry별 f_0 (같은 층: 초기하, 다른 층: 조성 곱)
    l, l2 = shells['l'], shells['l2']
    x_l, x_l2 = x_layer[:, l], x_layer[:, l2]
    f_0 = np.where(shells['is_in'][None, :],
                   f0_in_layer[:, l],
                   x_l * (1.0 - x_l2) + (1.0 - x_l) * x_l2)

    with np.errstate(invalid='ignore', divide='ignore'):
        alpha = np.where((f_0 > 0) & (n_pairs > 0)[None, :], 1.0 - f_obs / f_0, np.nan)
    valid = np.isfinite(alpha)

    result = dict(f_obs=f_obs, f_0=f_0, alpha=alpha, valid=valid)
    if layer_region is None:
        return result

    # region 수준 α_in: 층별 α의 평균이 아니라, 관측·기댓값을 pair 수로 가중합한 뒤 나눈다
    n_shell = shells['n_shell']
    in_idx = np.where(shells['is_in'])[0]
    region_names, region_m, columns = [], [], []
    for region in REGION_ORDER:
        for m in range(1, n_shell + 1):
            sel = in_idx[(np.array([layer_region[x] for x in l[in_idx]]) == region)
                         & (shells['m'][in_idx] == m)]
            columns.append(sel)
            region_names.append(region)
            region_m.append(m)

    w = np.where(n_pairs > 0, n_pairs, 0.0)[None, :]            # (1, G)
    w_obs = w * np.nan_to_num(f_obs, nan=0.0)
    w_exp = w * np.nan_to_num(f_0, nan=0.0)
    n_col = len(columns)
    num = np.zeros((n_frames, n_col))
    den = np.zeros((n_frames, n_col))
    weight = np.zeros((n_frames, n_col))
    for c, sel in enumerate(columns):
        if len(sel) == 0:
            continue
        num[:, c] = w_obs[:, sel].sum(axis=1)
        den[:, c] = w_exp[:, sel].sum(axis=1)
        weight[:, c] = w[:, sel].sum(axis=1)

    with np.errstate(invalid='ignore', divide='ignore'):
        region_f_obs = np.where(weight > 0, num / weight, np.nan)
        region_f_0 = np.where(weight > 0, den / weight, np.nan)
        region_alpha = np.where(den > 0, 1.0 - num / den, np.nan)

    result.update(region_f_obs=region_f_obs, region_f_0=region_f_0,
                  region_alpha=region_alpha, region_valid=np.isfinite(region_alpha),
                  region_n_pairs=weight,
                  region_names=np.array(region_names), region_m=np.array(region_m))
    return result
