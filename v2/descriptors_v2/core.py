"""
v1 pair 통계 C_lat · D_dom (cutoff 그래프). v2 산출물에서는 v1 대비 회귀 확인용으로만 계산한다.

입력은 한 region의 cation 부분집합이다. region 분할은 regions.py가 담당한다.
"""
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def get_cation_pairs_within_cutoff(cation_positions, cell, pbc, cutoff):
    """주기 경계 안에서 cutoff 이내 cation pair (i<j) 인덱스 배열 (P,2)를 반환한다.

    최소이미지는 ase.neighborlist(격자벡터 기준)로 잡는다 — 삼사정계 (111) 셀에서도 정확하다.
    자기 이미지(i==j)는 빼고, 같은 (i,j)가 여러 이미지로 잡혀도 한 번만 센다.
    """
    positions = np.asarray(cation_positions, dtype=float)
    if len(positions) < 2:
        return np.empty((0, 2), dtype=int)

    # 그래프만 만들므로 원소 종류 대신 더미 원자번호를 쓴다
    atoms = Atoms(numbers=np.ones(len(positions), dtype=int),
                  positions=positions, cell=cell, pbc=pbc)
    i, j = neighbor_list('ij', atoms, float(cutoff))

    upper = i < j
    if not upper.any():
        return np.empty((0, 2), dtype=int)
    return np.unique(np.stack([i[upper], j[upper]], axis=1), axis=0)


def compute_clat(cation_symbols, pairs):
    """v1 lateral clustering index C_lat (개념서 식 24–26, 이성분계)."""
    species_a = cation_symbols[pairs[:, 0]]
    species_b = cation_symbols[pairs[:, 1]]
    n_pairs_total = len(pairs)
    if n_pairs_total == 0:
        return np.nan

    n_same = np.sum(species_a == species_b)
    f_same = n_same / n_pairs_total

    unique, counts = np.unique(cation_symbols, return_counts=True)
    frac = dict(zip(unique, counts / len(cation_symbols)))
    x = list(frac.values())[0]
    f_same_rand = x**2 + (1 - x)**2
    if f_same_rand >= 1.0:
        return np.nan

    return (f_same - f_same_rand) / (1 - f_same_rand)


def compute_ddom(cation_symbols, pairs):
    """v1 domain-size score D_dom (개념서 식 27–28)."""
    species_list = np.unique(cation_symbols)

    d_per_species = {}
    for sp in species_list:
        sp_mask = cation_symbols == sp
        sp_indices = np.where(sp_mask)[0]
        if len(sp_indices) == 0:
            continue

        same_sp_pairs = pairs[
            (cation_symbols[pairs[:, 0]] == sp) & (cation_symbols[pairs[:, 1]] == sp)
        ] if len(pairs) else pairs

        local_index = {global_idx: local_idx for local_idx, global_idx in enumerate(sp_indices)}
        n_sp = len(sp_indices)

        if len(same_sp_pairs) == 0:
            d_per_species[sp] = 1.0 / n_sp
            continue

        rows = [local_index[i] for i in same_sp_pairs[:, 0]]
        cols = [local_index[i] for i in same_sp_pairs[:, 1]]
        adjacency = coo_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(n_sp, n_sp)
        )
        n_components, labels = connected_components(adjacency, directed=False)
        _, component_sizes = np.unique(labels, return_counts=True)
        d_per_species[sp] = component_sizes.max() / n_sp

    if not d_per_species:
        return np.nan, {}
    return max(d_per_species.values()), d_per_species
