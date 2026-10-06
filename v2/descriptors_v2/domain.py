"""
Domain morphology — D_Ir/D_Ru · n_cluster · H · I_R · P (스펙 <Domain morphology>).

    D_a(R)         = [Σ_{l∈R} max_q |C_a,l,q|] / [Σ_{l∈R} N_a,l]
    n_cluster,a(R) = Σ_{l∈R} n_a,l
    p_q            = |C_a,q| / N_a,R              (q: region 안 모든 층의 성분을 통틀어)
    H_a            = −Σ_q p_q ln p_q / ln n_cluster,a     (n_cluster = 1 → 0,  N_a,R = 0 → NA)
    I_R            = (1/|E_R^dom|) Σ_{i,j∈E_R^dom} |s_i − s_j|
    P_a,d          = 1 if a-domain이 방향 d로 셀을 가로질러 연결, else 0;   P_a,any = P_a,a OR P_a,b

그래프: 층별 면내 대칭화 k=4 최근접 그래프, 초기 구조에서 한 번 만들어 고정.
- 대칭화는 OR (i의 k위 안에 j, 또는 j의 k위 안에 i).
- k번째 이웃과 같은 거리(tie_tol 이내)의 이웃은 모두 포함한다. (110)은 4번째 이웃이 4중 동률이라
  이웃 6개가 되고, (100)/(111)은 4개다. tie_tol = 0.0이면 strict k-최근접.
- percolation 방향은 면내 격자벡터 a, b 기준이다 ((111)은 면내 격자가 비스듬하다).
"""
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .sro import REGION_ORDER

DEFAULT_K = 4
DEFAULT_TIE_TOL = 0.18      # Å. 스펙 shell tolerance와 같은 값
DEFAULT_R_CUT = 9.5         # Å. 이웃 후보 반경 (k+1번째 이웃까지 들어와야 한다)
ELEM_ORDER = ('Ir', 'Ru')   # s = 1 (Ir) -> 0,  s = 0 (Ru) -> 1


def inplane_neighbor_entries(positions, cell, pbc, layer_id, axis, r_cut=DEFAULT_R_CUT):
    """같은 층 안의 이웃 후보 (방향 있는 목록, i 오름차순 → 거리 오름차순).

    반환: i, j, d, off(M,2 정수 셀 오프셋: i에서 본 j의 이미지 = f_j + off), rank(i 안의 거리 순위),
          counts, starts
    최소이미지는 ase.neighborlist(격자벡터 기준). 같은 층이면서 vacuum 축 오프셋이 0인 쌍만 남긴다.
    """
    positions = np.asarray(positions, dtype=float)
    layer_id = np.asarray(layer_id, dtype=int)
    n_nodes = len(positions)
    in_plane = [a for a in range(3) if a != axis]

    probe = Atoms(numbers=np.ones(n_nodes, dtype=int), positions=positions,
                  cell=cell, pbc=pbc)
    i, j, d, shift = neighbor_list('ijdS', probe, float(r_cut))

    keep = (layer_id[i] == layer_id[j]) & (shift[:, axis] == 0) & (i != j)
    i, j, d, shift = i[keep], j[keep], d[keep], shift[keep]

    order = np.lexsort((d, i))
    i, j, d = i[order], j[order], d[order]
    off = shift[order][:, in_plane].astype(np.int64)

    counts = np.bincount(i, minlength=n_nodes)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rank = np.arange(len(i)) - starts[i]
    return i, j, d, off, rank, counts, starts


def build_domain_graph(positions, cell, pbc, layer_id, axis,
                       k=DEFAULT_K, tie_tol=DEFAULT_TIE_TOL, r_cut=DEFAULT_R_CUT):
    """초기 구조에서 층별 면내 대칭화 k-최근접 그래프를 만든다.

    positions, cell, pbc : 초기 구조의 cation 좌표와 셀
    layer_id, axis       : cation 층 라벨, vacuum 축
    tie_tol              : k번째 이웃 거리 + tie_tol 이내는 모두 포함 (0.0 = strict)

    반환 dict:
      edge_i, edge_j (i<j) · edge_layer · edge_d · edge_off(P,2: u_j = u_i + off) · edge_strict
      node_layer · node_deg · n_layers · n_nodes · n_edges · k · tie_tol · r_cut
      층별 진단: layer_n_nodes · layer_n_edges · layer_n_edges_strict · layer_d4 · layer_d5 ·
                 layer_deg_min/max/mean · layer_n_tie
    간선의 신원은 (i, j, off) — 같은 (i,j)의 다른 이미지는 다른 간선이다.
    """
    positions = np.asarray(positions, dtype=float)
    layer_id = np.asarray(layer_id, dtype=int)
    if len(positions) != len(layer_id):
        raise ValueError(f"좌표 {len(positions)}개와 층 라벨 {len(layer_id)}개가 다르다.")

    n_nodes = len(positions)
    n_layers = int(layer_id.max()) + 1
    i, j, d, off, rank, counts, starts = inplane_neighbor_entries(
        positions, cell, pbc, layer_id, axis, r_cut)

    # k+1번째 이웃까지 후보에 있어야 d_k가 반경에 잘리지 않는다
    n_min = np.min(counts[np.bincount(layer_id, minlength=n_layers)[layer_id] > 1])
    if n_min < k + 1:
        raise ValueError(
            f"r_cut={r_cut} A 안에 이웃이 {n_min}개뿐인 원자가 있다 (k={k} 이라 최소 {k+1}개 필요). "
            f"r_cut 을 키울 것 - 지금 값으로는 k번째 이웃 거리가 반경에 잘린다.")

    # 원자별 k번째 / k+1번째 이웃 거리
    d_k = d[starts + (k - 1)]                       # (N,)
    d_k1 = d[starts + k]
    sel = d <= (d_k[i] + tie_tol)                   # 동률 포함
    sel_strict = rank < k                           # strict (진단용)

    # 방향 있는 목록 -> 무방향 간선 (i<j로 정규화한 (i, j, off))
    swap = i > j
    ci = np.where(swap, j, i)
    cj = np.where(swap, i, j)
    coff = np.where(swap[:, None], -off, off)
    key = np.stack([ci, cj, coff[:, 0], coff[:, 1]], axis=1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()

    keep = np.zeros(len(uniq), dtype=bool)
    np.logical_or.at(keep, inv, sel)                # 대칭화 = OR
    strict = np.zeros(len(uniq), dtype=bool)
    np.logical_or.at(strict, inv, sel_strict)
    edge_d = np.zeros(len(uniq))
    edge_d[inv] = d

    edge_i, edge_j = uniq[keep, 0], uniq[keep, 1]
    edge_off = uniq[keep, 2:4].astype(np.int64)
    edge_d = edge_d[keep]
    edge_strict = strict[keep]
    edge_layer = layer_id[edge_i]

    # 진단
    node_deg = (np.bincount(edge_i, minlength=n_nodes)
                + np.bincount(edge_j, minlength=n_nodes))
    diag = {}
    diag['layer_n_nodes'] = np.bincount(layer_id, minlength=n_layers)
    diag['layer_n_edges'] = np.bincount(edge_layer, minlength=n_layers)
    diag['layer_n_edges_strict'] = np.bincount(edge_layer[edge_strict], minlength=n_layers)
    for name, arr, fn in (('layer_d4', d_k, np.median), ('layer_d5', d_k1, np.median),
                          ('layer_deg_min', node_deg, np.min),
                          ('layer_deg_max', node_deg, np.max),
                          ('layer_deg_mean', node_deg, np.mean)):
        diag[name] = np.array([fn(arr[layer_id == l]) if (layer_id == l).any() else np.nan
                               for l in range(n_layers)], dtype=float)
    tie_atom = np.isclose(d_k1, d_k, atol=1e-6)
    diag['layer_n_tie'] = np.array([int(tie_atom[layer_id == l].sum())
                                    for l in range(n_layers)], dtype=int)

    out = dict(edge_i=edge_i, edge_j=edge_j, edge_layer=edge_layer, edge_d=edge_d,
               edge_off=edge_off, edge_strict=edge_strict,
               node_layer=layer_id, node_deg=node_deg,
               n_layers=n_layers, n_nodes=n_nodes, n_edges=len(edge_i),
               k=k, tie_tol=tie_tol, r_cut=r_cut)
    out.update(diag)
    return out


def _csr_from_edges(n_nodes, src, dst, off):
    """방향 간선 (src -> dst, u[dst] = u[src] + off)을 CSR 인접구조로 만든다."""
    order = np.argsort(src, kind='stable')
    src_s, dst_s, off_s = src[order], dst[order], off[order]
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.add.at(indptr, src_s + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, dst_s, off_s


def _unwrap_offsets(n_nodes, labels, indptr, dst, off):
    """연결성분마다 BFS spanning tree를 따라 unwrap 셀 오프셋 u (N,2)를 매긴다 (뿌리 u = 0)."""
    u = np.zeros((n_nodes, 2), dtype=np.int64)
    visited = np.zeros(n_nodes, dtype=bool)
    roots = np.unique(labels, return_index=True)[1]
    visited[roots] = True
    frontier = roots
    while frontier.size:
        counts = (indptr[frontier + 1] - indptr[frontier])
        total = int(counts.sum())
        if total == 0:
            break
        starts = np.repeat(indptr[frontier], counts)
        within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        e_idx = starts + within
        parent = np.repeat(frontier, counts)
        child = dst[e_idx]
        fresh = ~visited[child]
        if not fresh.any():
            break
        child_f = child[fresh]
        u_f = u[parent[fresh]] + off[e_idx[fresh]]
        # 같은 단계에서 여러 부모가 닿은 자식은 하나만 쓴다 (감김 판정은 tree 선택과 무관)
        uniq, first = np.unique(child_f, return_index=True)
        u[uniq] = u_f[first]
        visited[uniq] = True
        frontier = uniq
    return u


def _percolation(u, edge_i, edge_j, edge_off):
    """간선별 감김 여부 (P,2) bool = (u[j] − u[i] − off ≠ 0), 격자방향 a, b 각각."""
    if len(edge_i) == 0:
        return np.zeros((0, 2), dtype=bool)
    return (u[edge_j] - u[edge_i] - edge_off) != 0


def compute_domain_frames(graph, s_frames, layer_region):
    """프레임 묶음의 domain morphology.

    graph        : build_domain_graph 결과
    s_frames     : (F, N) occupation, 1 = Ir, 0 = Ru
    layer_region : layer_region_map 결과

    반환 (region 축 = REGION_ORDER, 원소 축 = ELEM_ORDER, 방향 축 = a, b):
      층     : layer_n_a · layer_max · layer_ncl (F,K,2) · layer_perc (F,K,2,2) · layer_I (F,K)
      region : D · n_cluster · H · n_a (F,5,2) · P (F,5,2,2) · I (F,5)
      고정량 : layer_n_edges · region_n_edges · layer_region_index
    I_R에는 조성 보정이 없다.
    """
    s = np.asarray(s_frames)
    if s.ndim == 1:
        s = s[None, :]
    s = s.astype(bool)
    n_frames = s.shape[0]

    ei, ej = graph['edge_i'], graph['edge_j']
    eoff, elayer = graph['edge_off'], graph['edge_layer']
    node_layer = graph['node_layer']
    n_nodes, n_layers = graph['n_nodes'], graph['n_layers']
    n_region = len(REGION_ORDER)

    region_index = {name: i for i, name in enumerate(REGION_ORDER)}
    layer_r = np.array([region_index[layer_region[l]] for l in range(n_layers)], dtype=int)
    edge_r = layer_r[elayer]

    layer_n_edges = np.bincount(elayer, minlength=n_layers)
    region_n_edges = np.bincount(edge_r, minlength=n_region).astype(float)

    out = dict(
        layer_n_a=np.zeros((n_frames, n_layers, 2)),
        layer_max=np.zeros((n_frames, n_layers, 2)),
        layer_ncl=np.zeros((n_frames, n_layers, 2), dtype=int),
        layer_perc=np.zeros((n_frames, n_layers, 2, 2), dtype=bool),
        layer_I=np.full((n_frames, n_layers), np.nan),
        D=np.full((n_frames, n_region, 2), np.nan),
        H=np.full((n_frames, n_region, 2), np.nan),
        n_cluster=np.zeros((n_frames, n_region, 2), dtype=int),
        n_a=np.zeros((n_frames, n_region, 2)),
        P=np.zeros((n_frames, n_region, 2, 2), dtype=bool),
        I=np.full((n_frames, n_region), np.nan),
    )

    ones = np.ones(len(ei))
    for t in range(n_frames):
        st = s[t]
        hetero = st[ei] != st[ej]
        homo = ~hetero

        # I_R: 이종 간선 비율
        num = np.bincount(edge_r, weights=hetero.astype(float), minlength=n_region)
        with np.errstate(invalid='ignore', divide='ignore'):
            out['I'][t] = np.where(region_n_edges > 0, num / region_n_edges, np.nan)
        lnum = np.bincount(elayer, weights=hetero.astype(float), minlength=n_layers)
        with np.errstate(invalid='ignore', divide='ignore'):
            out['layer_I'][t] = np.where(layer_n_edges > 0, lnum / layer_n_edges, np.nan)

        # 동종 간선만 남긴 그래프의 연결성분 (성분 하나 = 같은 층·같은 원소)
        hi, hj, hoff = ei[homo], ej[homo], eoff[homo]
        adj = coo_matrix((ones[homo], (hi, hj)), shape=(n_nodes, n_nodes))
        n_comp, labels = connected_components(adj, directed=False)

        comp_size = np.bincount(labels, minlength=n_comp).astype(float)
        comp_layer = np.zeros(n_comp, dtype=int)
        comp_layer[labels] = node_layer
        comp_elem = np.zeros(n_comp, dtype=int)
        comp_elem[labels] = (~st).astype(int)          # 0 = Ir, 1 = Ru

        np.maximum.at(out['layer_max'][t], (comp_layer, comp_elem), comp_size)
        np.add.at(out['layer_ncl'][t], (comp_layer, comp_elem), 1)
        np.add.at(out['layer_n_a'][t], (comp_layer, comp_elem), comp_size)

        # percolation: unwrap 오프셋을 매긴 뒤 모든 동종 간선을 검사
        src = np.concatenate([hi, hj])
        dst = np.concatenate([hj, hi])
        off = np.concatenate([hoff, -hoff])
        indptr, dst_s, off_s = _csr_from_edges(n_nodes, src, dst, off)
        u = _unwrap_offsets(n_nodes, labels, indptr, dst_s, off_s)
        wind = _percolation(u, hi, hj, hoff)
        if len(hi):
            np.logical_or.at(out['layer_perc'][t],
                             (elayer[homo], (~st[hi]).astype(int)), wind)

        # region 수준
        g = layer_r[comp_layer] * 2 + comp_elem                 # region × 원소 그룹
        n_group = n_region * 2
        grp_n = np.bincount(g, weights=comp_size, minlength=n_group)          # N_a,R
        grp_cnt = np.bincount(g, minlength=n_group).astype(float)             # n_cluster
        with np.errstate(divide='ignore', invalid='ignore'):
            grp_slns = np.bincount(g, weights=comp_size * np.log(comp_size),
                                   minlength=n_group)
        out['n_a'][t] = grp_n.reshape(n_region, 2)
        out['n_cluster'][t] = grp_cnt.reshape(n_region, 2).astype(int)

        # D_a = Σ_l max_q / Σ_l N_a,l  (층별 최대의 합)
        max_sum = np.zeros((n_region, 2))
        np.add.at(max_sum, (layer_r, slice(None)), out['layer_max'][t])
        with np.errstate(invalid='ignore', divide='ignore'):
            out['D'][t] = np.where(grp_n.reshape(n_region, 2) > 0,
                                   max_sum / grp_n.reshape(n_region, 2), np.nan)

        # H_a = [ln N − (1/N) Σ_q s_q ln s_q] / ln n_cluster
        with np.errstate(divide='ignore', invalid='ignore'):
            ent = np.log(grp_n) - grp_slns / grp_n
            h = ent / np.log(grp_cnt)
        h = np.where(grp_cnt == 1, 0.0, h)              # n_cluster = 1 → 0
        h = np.where(grp_n == 0, np.nan, h)             # N_a,R = 0 → NA
        out['H'][t] = h.reshape(n_region, 2)

        # P: region 안 어느 층이라도 관통하면 1 (층에 걸쳐 OR)
        np.logical_or.at(out['P'][t], (layer_r, slice(None), slice(None)),
                         out['layer_perc'][t])

    out['layer_n_edges'] = layer_n_edges
    out['region_n_edges'] = region_n_edges
    out['layer_region_index'] = layer_r
    return out
