"""
구조 유형(bulk / slab) 판별, 원자층 라벨, region 분할.

region 분할 (get_regions의 split_top_bottom):
  3분할 {surf, sub, bulk}         개념서 §2.2 식 6 — 위·아래 표면을 합친다 (v1 회귀 확인용)
  5분할 {TS, BS, TSS, BSS, bulk}  스펙 v2 — 위·아래를 분리한다
"""
import numpy as np


def _axis_vacuum_gap(atoms, axis):
    """해당 축에서 원자들 사이 최대 빈 구간(주기 wrap 포함, Å)."""
    cell_length = atoms.cell.lengths()[axis]
    frac = atoms.get_scaled_positions(wrap=True)[:, axis]
    coords = np.sort(frac) * cell_length
    if len(coords) < 2:
        return 0.0
    gaps = np.diff(coords)
    wrap_gap = (coords[0] + cell_length) - coords[-1]
    return float(max(gaps.max(), wrap_gap))


def find_vacuum_axis(atoms, vacuum_gap_threshold=5.0):
    """vacuum gap이 threshold를 넘는 축 중 가장 큰 것. 없으면 None."""
    best_axis, best_gap = None, vacuum_gap_threshold
    for axis in range(3):
        if not atoms.pbc[axis]:
            return axis
        gap = _axis_vacuum_gap(atoms, axis)
        if gap > best_gap:
            best_axis, best_gap = axis, gap
    return best_axis


def detect_structure_type(atoms, vacuum_gap_threshold=5.0):
    """'bulk' 또는 'slab' — 비주기 축이나 threshold(Å)를 넘는 빈 구간이 있으면 slab."""
    return 'slab' if find_vacuum_axis(atoms, vacuum_gap_threshold) is not None else 'bulk'


def top_side_is_max_layer(atoms, axis):
    """layer_id가 큰 쪽이 top인지 판정한다.

    스펙: "cross(a,b)를 +z로 정렬한 뒤 최대 z 쪽이 top".
    회전은 내적 부호를 바꾸지 않으므로  top_is_max ⇔ dot(cross(a, b), c) > 0  (c = vacuum 축).
    a, b는 vacuum 축 기준 순환순서로 잡는다 (vacuum 축이 1일 때 부호가 뒤집히지 않도록).
    """
    cell = np.asarray(atoms.cell)
    in_plane_a = cell[(axis + 1) % 3]
    in_plane_b = cell[(axis + 2) % 3]
    return bool(np.dot(np.cross(in_plane_a, in_plane_b), cell[axis]) > 0)


def layer_coordinate(atoms, indices, axis):
    """층 감지에 쓸 1차원 좌표(Å) = fractional 좌표 × 셀 길이.

    면내 격자벡터에 축 성분이 있는 셀에서도 성립하도록 cartesian 대신 fractional을 쓴다.
    슬랩이 셀 경계에 걸쳐 있으면 가장 큰 빈 구간(vacuum)이 경계에 오도록 좌표를 회전시킨다.
    """
    length = atoms.cell.lengths()[axis]
    coord = atoms.get_scaled_positions(wrap=True)[indices, axis] * length
    if len(coord) < 2:
        return coord
    sorted_coord = np.sort(coord)
    gaps = np.diff(sorted_coord)
    wrap_gap = (sorted_coord[0] + length) - sorted_coord[-1]
    if gaps.max() > wrap_gap:
        coord = np.mod(coord - sorted_coord[int(gaps.argmax()) + 1], length)
    return coord


def detect_atomic_layers(coord, layer_gap_threshold=0.5):
    """1차원 좌표를 원자층으로 묶는다. 간격 > layer_gap_threshold(Å)이면 층 경계.

    반환: layer index 배열 (0 = 좌표가 가장 작은 층).
    """
    order = np.argsort(coord)
    sorted_coord = coord[order]
    gaps = np.diff(sorted_coord)
    is_boundary = gaps > layer_gap_threshold
    layer_id_sorted = np.concatenate([[0], np.cumsum(is_boundary)])
    layer_id = np.empty_like(layer_id_sorted)
    layer_id[order] = layer_id_sorted
    return layer_id


def compute_layer_id(atoms, cation_indices, vacuum_gap_threshold=5.0,
                      layer_gap_threshold=0.5):
    """cation의 층 라벨. bulk(vacuum 축 없음)이면 None.

    초기 구조에서 한 번 계산해 get_regions(layer_id=...)로 넘기고 MC 전체에 고정한다 (스펙 규약).
    프레임마다 다시 감지하면 표면 rumpling으로 층이 합쳐져 region 경계가 흔들린다.
    """
    axis = find_vacuum_axis(atoms, vacuum_gap_threshold)
    if axis is None:
        return None
    coord = layer_coordinate(atoms, cation_indices, axis)
    return detect_atomic_layers(coord, layer_gap_threshold)


def get_regions(atoms, cation_indices, structure_type,
                 n_surf_layers=1, n_sub_layers=1,
                 vacuum_gap_threshold=5.0, layer_gap_threshold=0.5,
                 layer_id=None, split_top_bottom=False, axis=None):
    """region 이름 -> cation_indices에 대한 boolean mask.

    bulk: {'bulk'}.  slab: 표면에서 몇 번째 층인지(층 개수 기준)로 나눈다.
      split_top_bottom=False : {'surf', 'sub', 'bulk'}
      split_top_bottom=True  : {'TS', 'BS', 'TSS', 'BSS', 'bulk'} — top은 top_side_is_max_layer로 판정
    두 분할은 surf == TS|BS, sub == TSS|BSS 로 포개진다.

    layer_id : compute_layer_id로 미리 구한 고정 층 라벨. None이면 이 프레임에서 감지한다.
    axis     : vacuum 축 (5분할에서만 사용). None이면 여기서 찾는다.
    """
    n = len(cation_indices)
    if structure_type == 'bulk':
        return {'bulk': np.ones(n, dtype=bool)}

    if layer_id is None:
        layer_id = compute_layer_id(atoms, cation_indices,
                                     vacuum_gap_threshold, layer_gap_threshold)
    max_layer = layer_id.max()
    # 각 표면에서 몇 번째 층인가 (0 = 표면층)
    depth_from_max = max_layer - layer_id
    depth_from_min = layer_id

    if not split_top_bottom:
        layer_depth = np.minimum(depth_from_min, depth_from_max)
        surf_mask = layer_depth < n_surf_layers
        sub_mask = (~surf_mask) & (layer_depth < n_surf_layers + n_sub_layers)
        bulk_mask = ~surf_mask & ~sub_mask
        return {'surf': surf_mask, 'sub': sub_mask, 'bulk': bulk_mask}

    # 위·아래에 surface+subsurface를 따로 두려면 그 두 배의 층이 필요하다 (모자라면 마스크가 겹친다)
    n_shell = n_surf_layers + n_sub_layers
    n_layers = int(max_layer) + 1
    if n_layers < 2 * n_shell:
        raise ValueError(
            f"층이 {n_layers}개뿐이라 top/bottom을 나눌 수 없다 "
            f"(surface {n_surf_layers}층 + subsurface {n_sub_layers}층을 양쪽에 두려면 "
            f"{2 * n_shell}층 이상 필요). region 마스크가 겹치므로 진행하지 않는다."
        )

    if axis is None:
        axis = find_vacuum_axis(atoms, vacuum_gap_threshold)
    if top_side_is_max_layer(atoms, axis):
        depth_top, depth_bot = depth_from_max, depth_from_min
    else:
        depth_top, depth_bot = depth_from_min, depth_from_max

    ts_mask = depth_top < n_surf_layers
    bs_mask = depth_bot < n_surf_layers
    tss_mask = (~ts_mask) & (depth_top < n_shell)
    bss_mask = (~bs_mask) & (depth_bot < n_shell)
    bulk_mask = ~(ts_mask | bs_mask | tss_mask | bss_mask)

    # 스펙 x_region 성분 순서
    return {'TS': ts_mask, 'BS': bs_mask, 'TSS': tss_mask, 'BSS': bss_mask,
            'bulk': bulk_mask}
