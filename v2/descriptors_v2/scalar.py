"""
프레임(ase.Atoms) 하나 -> scalar descriptor dict.

slab에서는 두 계열을 함께 낸다:
  [v1] S_depth · G_layer · C_lat · D_dom     region 3분할 — v1 산출물과의 회귀 확인용
  [v2] x_region(5성분) · S_surf/S_sub(avg) · A_tb · G_star · x_layer    region 5분할 (스펙 <깊이별 조성>)
bulk에서는 C_lat · D_dom만 낸다.
region은 Å 두께가 아니라 "표면에서 몇 번째 원자층인가"로 정의한다.
"""
import numpy as np

from .core import get_cation_pairs_within_cutoff, compute_clat, compute_ddom
from .regions import (get_regions, find_vacuum_axis,
                       detect_atomic_layers, layer_coordinate, compute_layer_id)


def _frac_ir(cation_symbols, mask, ir_symbol):
    """마스크 안의 Ir 비율. 원자가 없으면 NaN (0과 구분)."""
    return float(np.mean(cation_symbols[mask] == ir_symbol)) if mask.sum() > 0 else np.nan


def _x_per_layer(layer_id, cation_symbols, ir_symbol):
    """층별 Ir 비율 x_layer = (x_0, ..., x_{K-1}). 원자가 없는 층은 NaN."""
    n_layers = int(layer_id.max()) + 1
    return np.array([_frac_ir(cation_symbols, layer_id == layer, ir_symbol)
                     for layer in range(n_layers)])


def _layer_total_variation(x_layer):
    """(Σ|x_{l+1} − x_l|, 유효 층 수). NaN 층은 건너뛴다. G_layer와 G_star가 공유한다."""
    valid = x_layer[~np.isnan(x_layer)]
    if len(valid) < 2:
        return np.nan, len(valid)
    return float(np.sum(np.abs(np.diff(valid)))), len(valid)


def compute_depth_profile(cation_symbols, ir_symbol, regions5, x_layer):
    """스펙 <깊이별 조성>. regions5 = {'TS','BS','TSS','BSS','bulk'} 마스크, x_layer = 층별 Ir 비율.

    반환 칼럼: x_TS · x_BS · x_TSS · x_BSS · x_bulk · n_TS … n_BSS ·
              S_surf_top/bottom/avg · S_sub_top/bottom/avg · A_tb · A_tb_abs · G_star · x_layer(배열)

    x_surf = (N_TS x_TS + N_BS x_BS)/(N_TS + N_BS) 는 TS∪BS 안의 Ir 비율과 같으므로 합친 마스크로 센다
    (한쪽 region이 비어도 정의된다). 빈 region의 x는 NaN이고 파생값으로 전파된다.
    G_star = 총변화량 / (K − 1), K = 원자가 있는 층 수.
    """
    x = {name: _frac_ir(cation_symbols, mask, ir_symbol) for name, mask in regions5.items()}
    x_bulk = x['bulk']

    x_surf = _frac_ir(cation_symbols, regions5['TS'] | regions5['BS'], ir_symbol)
    x_sub = _frac_ir(cation_symbols, regions5['TSS'] | regions5['BSS'], ir_symbol)

    A_tb = x['TS'] - x['BS']

    total_variation, n_valid = _layer_total_variation(x_layer)
    G_star = total_variation / (n_valid - 1) if n_valid > 1 else np.nan

    return {
        'x_TS': x['TS'], 'x_BS': x['BS'], 'x_TSS': x['TSS'], 'x_BSS': x['BSS'],
        'x_bulk': x_bulk,
        'n_TS': int(regions5['TS'].sum()), 'n_BS': int(regions5['BS'].sum()),
        'n_TSS': int(regions5['TSS'].sum()), 'n_BSS': int(regions5['BSS'].sum()),
        'S_surf_top': x['TS'] - x_bulk, 'S_surf_bottom': x['BS'] - x_bulk,
        'S_surf_avg': x_surf - x_bulk,
        'S_sub_top': x['TSS'] - x_bulk, 'S_sub_bottom': x['BSS'] - x_bulk,
        'S_sub_avg': x_sub - x_bulk,
        'A_tb': A_tb, 'A_tb_abs': abs(A_tb),
        'G_star': G_star,
        'x_layer': x_layer,
    }


def compute_depth_descriptors(atoms, cation_indices, cation_symbols, axis, ir_symbol,
                               regions, layer_gap_threshold=0.5,
                               w_surf=0.5, w_sub=0.5, layer_id=None, x_layer=None):
    """v1 S_depth (개념서 식 17–19) · G_layer (식 22). regions는 3분할 {'surf','sub','bulk'}."""
    if layer_id is None:
        coord = layer_coordinate(atoms, cation_indices, axis)
        layer_id = detect_atomic_layers(coord, layer_gap_threshold)

    x_surf = _frac_ir(cation_symbols, regions['surf'], ir_symbol)
    x_sub = _frac_ir(cation_symbols, regions['sub'], ir_symbol)
    x_bulk = _frac_ir(cation_symbols, regions['bulk'], ir_symbol)
    S_surf = x_surf - x_bulk
    S_sub = x_sub - x_bulk
    S_depth = w_surf * S_surf + w_sub * S_sub

    if x_layer is None:
        x_layer = _x_per_layer(layer_id, cation_symbols, ir_symbol)
    n_layers_found = len(x_layer)
    G_layer, _ = _layer_total_variation(x_layer)

    return {
        'x_surf': x_surf, 'x_sub': x_sub, 'x_bulk': x_bulk,
        'S_surf': S_surf, 'S_sub': S_sub, 'S_depth': S_depth,
        'G_layer': G_layer, 'n_atomic_layers': int(n_layers_found),
    }


def compute_frame_descriptors(atoms, cutoff, cation_species=('Ir', 'Ru'),
                               n_surf_layers=1, n_sub_layers=1,
                               vacuum_gap_threshold=5.0, layer_gap_threshold=0.5,
                               layer_id=None, ref_positions=None, ref_cell=None):
    """한 프레임의 descriptor dict. 'x_layer'만 배열이다 (길이 = 층 수, facet마다 다름).

    layer_id      : 기준 구조(초기 slab)에서 구한 고정 층 라벨. None이면 이 프레임에서 감지.
    ref_positions : 기준 구조의 cation 좌표. 주면 C_lat/D_dom의 이웃 pair를 여기서 잡는다
                    (swap MC는 site가 고정이므로 이웃 그래프도 고정한다). ref_cell은 그 셀.
    깊이별 조성은 고정 마스크 위에서 라벨만 세므로 좌표를 쓰지 않는다.
    """
    symbols = np.array(atoms.get_chemical_symbols())
    positions = atoms.get_positions()

    cation_mask = np.isin(symbols, cation_species)
    cation_indices = np.where(cation_mask)[0]
    if len(cation_indices) == 0:
        raise ValueError(f"cation_species={cation_species} 에 해당하는 원자가 없습니다.")

    cation_positions = positions[cation_indices]
    cation_symbols = symbols[cation_indices]

    # 이웃 pair를 잡을 좌표. 기준 좌표의 길이가 안 맞으면 이 프레임 좌표를 쓰고 neighbor_def에 남긴다
    pair_positions = cation_positions
    pair_cell = atoms.cell
    neighbor_def = 'per_frame'
    if ref_positions is not None and len(ref_positions) == len(cation_indices):
        pair_positions = np.asarray(ref_positions, dtype=float)
        if ref_cell is not None:
            pair_cell = ref_cell
        neighbor_def = 'fixed'

    axis = find_vacuum_axis(atoms, vacuum_gap_threshold)
    structure_type = 'slab' if axis is not None else 'bulk'
    regions = get_regions(atoms, cation_indices, structure_type,
                           n_surf_layers, n_sub_layers, vacuum_gap_threshold,
                           layer_gap_threshold, layer_id=layer_id)

    result = {'structure_type': structure_type, 'n_cations': int(len(cation_indices)),
              'neighbor_def': neighbor_def}

    for region_name, mask in regions.items():
        pos_r = pair_positions[mask]
        sym_r = cation_symbols[mask]
        result[f'n_{region_name}'] = int(mask.sum())
        if len(pos_r) < 2:
            result[f'C_lat_{region_name}'] = np.nan
            result[f'D_dom_{region_name}'] = np.nan
            continue
        pairs = get_cation_pairs_within_cutoff(pos_r, pair_cell, atoms.pbc, cutoff)
        result[f'C_lat_{region_name}'] = compute_clat(sym_r, pairs)
        d_dom, _ = compute_ddom(sym_r, pairs)
        result[f'D_dom_{region_name}'] = d_dom

    if structure_type == 'slab':
        ir_symbol = cation_species[0]

        # 층 라벨과 층 조성을 한 번만 정해 3분할·5분할 계열이 같은 것을 쓰게 한다
        resolved_layer_id = layer_id
        if resolved_layer_id is None:
            resolved_layer_id = detect_atomic_layers(
                layer_coordinate(atoms, cation_indices, axis), layer_gap_threshold)
        x_layer = _x_per_layer(resolved_layer_id, cation_symbols, ir_symbol)

        # [v1] 3분할
        result.update(compute_depth_descriptors(
            atoms, cation_indices, cation_symbols, axis, ir_symbol,
            regions, layer_gap_threshold, layer_id=resolved_layer_id, x_layer=x_layer,
        ))

        # [v2] 5분할
        regions5 = get_regions(atoms, cation_indices, structure_type,
                                n_surf_layers, n_sub_layers, vacuum_gap_threshold,
                                layer_gap_threshold, layer_id=resolved_layer_id,
                                split_top_bottom=True, axis=axis)
        result.update(compute_depth_profile(
            cation_symbols, ir_symbol, regions5, x_layer,
        ))

    return result
