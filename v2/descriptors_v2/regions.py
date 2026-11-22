"""
원자층 라벨과 region 5분할 {TS, BS, TSS, BSS, bulk}.

region은 Å 두께가 아니라 표면에서 몇 번째 원자층인가로 정한다. top = cross(a, b) 쪽 표면.
"""
import numpy as np


def _axis_vacuum_gap(atoms, axis):
    """해당 축에서 원자 사이 최대 빈 구간 (주기 wrap 포함, Å)."""
    cell_length = atoms.cell.lengths()[axis]
    frac = atoms.get_scaled_positions(wrap=True)[:, axis]
    coords = np.sort(frac) * cell_length
    if len(coords) < 2:
        return 0.0
    gaps = np.diff(coords)
    wrap_gap = (coords[0] + cell_length) - coords[-1]
    return float(max(gaps.max(), wrap_gap))


def find_vacuum_axis(atoms, vacuum_gap_threshold=5.0):
    """비주기 축, 또는 빈 구간이 threshold(Å)를 넘는 축 중 가장 큰 것. 없으면 None."""
    best_axis, best_gap = None, vacuum_gap_threshold
    for axis in range(3):
        if not atoms.pbc[axis]:
            return axis
        gap = _axis_vacuum_gap(atoms, axis)
        if gap > best_gap:
            best_axis, best_gap = axis, gap
    return best_axis


def top_side_is_max_layer(atoms, axis):
    """층 번호가 큰 쪽이 top인가 ⇔ dot(cross(a, b), c) > 0  (c = vacuum 축, a·b = 그 다음 순환 순서의 축)."""
    cell = np.asarray(atoms.cell)
    in_plane_a = cell[(axis + 1) % 3]
    in_plane_b = cell[(axis + 2) % 3]
    return bool(np.dot(np.cross(in_plane_a, in_plane_b), cell[axis]) > 0)


def layer_coordinate(atoms, indices, axis):
    """층 감지용 1차원 좌표(Å) = 표면 법선 방향 높이. 슬랩이 셀 경계에 걸치면 vacuum이 경계에 오게 옮긴다."""
    cell = np.asarray(atoms.cell)
    normal = np.cross(cell[(axis + 1) % 3], cell[(axis + 2) % 3])
    length = abs(np.dot(cell[axis], normal)) / np.linalg.norm(normal)   # 셀의 법선 방향 높이
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
    """간격 > layer_gap_threshold(Å)이면 층 경계. 반환: 층 번호 (0 = 좌표가 가장 작은 층)."""
    order = np.argsort(coord)
    sorted_coord = coord[order]
    gaps = np.diff(sorted_coord)
    is_boundary = gaps > layer_gap_threshold
    layer_id_sorted = np.concatenate([[0], np.cumsum(is_boundary)])
    layer_id = np.empty_like(layer_id_sorted)
    layer_id[order] = layer_id_sorted
    return layer_id


def compute_layer_id(atoms, cation_indices, vacuum_gap_threshold=5.0, layer_gap_threshold=0.5):
    """cation의 층 라벨. 초기 구조에서 한 번 구해 모든 프레임에 쓴다."""
    axis = find_vacuum_axis(atoms, vacuum_gap_threshold)
    if axis is None:
        raise ValueError("vacuum 축이 없다 — slab 구조가 아니다.")
    coord = layer_coordinate(atoms, cation_indices, axis)
    return detect_atomic_layers(coord, layer_gap_threshold)


def get_regions(atoms, layer_id, axis=None, n_surf_layers=1, n_sub_layers=1,
                vacuum_gap_threshold=5.0):
    """region 이름 -> (N,) bool 마스크 (TS · BS · TSS · BSS · bulk 순).

    TS/BS = 위·아래 표면 n_surf_layers층, TSS/BSS = 그 안쪽 n_sub_layers층, 나머지 = bulk.
    axis : vacuum 축. None이면 atoms에서 찾는다.
    """
    layer_id = np.asarray(layer_id, dtype=int)
    n_layers = int(layer_id.max()) + 1
    n_shell = n_surf_layers + n_sub_layers
    if n_layers < 2 * n_shell:
        raise ValueError(f"층이 {n_layers}개뿐이라 위·아래 region을 나눌 수 없다 ({2 * n_shell}층 이상 필요).")

    if axis is None:
        axis = find_vacuum_axis(atoms, vacuum_gap_threshold)
    depth_from_max = (n_layers - 1) - layer_id      # 층 번호가 큰 쪽 표면에서 몇 번째 층인가 (0 = 표면층)
    depth_from_min = layer_id
    if top_side_is_max_layer(atoms, axis):
        depth_top, depth_bot = depth_from_max, depth_from_min
    else:
        depth_top, depth_bot = depth_from_min, depth_from_max

    ts = depth_top < n_surf_layers
    bs = depth_bot < n_surf_layers
    tss = ~ts & (depth_top < n_shell)
    bss = ~bs & (depth_bot < n_shell)
    bulk = ~(ts | bs | tss | bss)
    return {'TS': ts, 'BS': bs, 'TSS': tss, 'BSS': bss, 'bulk': bulk}
