"""
깊이별 조성 — x_region · x_layer · S_surf(avg) · S_sub(avg) · A_tb · G*

    x_R         = (1/N_R) Σ_{i∈R} s_i            R ∈ {TS, BS, TSS, BSS, bulk}
    S_surf(avg) = x_{TS∪BS} − x_bulk,   S_sub(avg) = x_{TSS∪BSS} − x_bulk
    A_tb        = x_TS − x_BS
    G*          = Σ_l |x_{l+1} − x_l| / (K − 1)
"""
import numpy as np


def compute_depth_frames(s_frames, layer_id, regions5):
    """프레임 묶음의 깊이별 조성.

    s_frames : (F, N) occupation (1 = Ir, 0 = Ru)
    layer_id : (N,) 층 라벨 · regions5 : region 이름 -> (N,) bool 마스크 (둘 다 초기 구조에서 고정)
    반환     : 열 이름 -> (F,) 배열. x_layer는 (F, K), n_*은 정수. 빈 region의 x는 NaN
    """
    s = np.asarray(s_frames, dtype=float)
    if s.ndim == 1:
        s = s[None, :]
    layer_id = np.asarray(layer_id, dtype=int)

    def frac_ir(mask):
        n = int(mask.sum())
        return s[:, mask].sum(axis=1) / n if n else np.full(len(s), np.nan)

    n_layers = int(layer_id.max()) + 1
    if (np.bincount(layer_id, minlength=n_layers) == 0).any():
        raise ValueError("원자가 없는 층이 있다.")
    x_layer = np.stack([frac_ir(layer_id == l) for l in range(n_layers)], axis=1)

    x = {name: frac_ir(mask) for name, mask in regions5.items()}
    x_surf = frac_ir(regions5['TS'] | regions5['BS'])
    x_sub = frac_ir(regions5['TSS'] | regions5['BSS'])
    A_tb = x['TS'] - x['BS']

    out = {f'x_{name}': x[name] for name in ('TS', 'BS', 'TSS', 'BSS', 'bulk')}
    out.update({f'n_{name}': int(regions5[name].sum()) for name in ('TS', 'BS', 'TSS', 'BSS', 'bulk')})
    out.update(
        S_surf_top=x['TS'] - x['bulk'], S_surf_bottom=x['BS'] - x['bulk'],
        S_surf_avg=x_surf - x['bulk'],
        S_sub_top=x['TSS'] - x['bulk'], S_sub_bottom=x['BSS'] - x['bulk'],
        S_sub_avg=x_sub - x['bulk'],
        A_tb=A_tb, A_tb_abs=np.abs(A_tb),
        G_star=np.abs(np.diff(x_layer, axis=1)).sum(axis=1) / (n_layers - 1),
        x_layer=x_layer,
        n_atomic_layers=n_layers,
    )
    return out
