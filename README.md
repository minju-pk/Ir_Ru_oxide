# Ir–Ru 혼합 산화물 slab — 원자 배열 descriptor (spec v2)

Ir–Ru 혼합 산화물 slab 30구조(Monte Carlo swap, 800 K)의 Ir/Ru 배열을
협업 연구실의 descriptor 정의(2026-08-27, spec v2)에 따라 정량화한 코드와 결과입니다.

| 범주 | descriptor | 상태 |
|---|---|---|
| 깊이별 조성 | x_region(TS/BS/TSS/BSS/bulk) · x_layer · S_surf(avg) · S_sub(avg) · A_tb · G* | 완료 |
| 단거리 ordering | α_in(region × shell) · α_out(layer-pair × shell) | 완료 |
| Domain morphology | D_Ir · D_Ru · n_cluster · H · I_R · P (k=4 면내 그래프) | 완료 |
| 국소 기하 | structure-level SOAP (relaxation 전/후) | 예정 |

## 결과 — 그림과 그림 데이터

- 그림: `v2/plot/figures_v2_report/` (17장, PNG·PDF)
- 그림에 찍힌 모든 점의 값: `v2/plot/figure_data_v2_report/`
  — 평균·SD와 함께 **valid_fraction**(값이 정의된 비율)과 **single_atom_share**
  (소수 원자 1개라 α=0·D=1·H=0으로 고정된 비율)를 함께 실었습니다. 열 설명은 그 폴더의 README.
- 집계: MC 601–1100 스텝, 점유 스텝 가중 평균. 오차막대 = 스텝 간 SD(평균의 오차 아님)

## 구조

```
v2/
├── descriptors_v2/          계산 모듈 (regions · scalar · sro · domain)
├── run_descriptors_v2.py    30케이스 배치 실행
├── measure_shells.py · measure_domain_graph.py   껍질 거리 · k=4 그래프 실측
├── verify_sro.py · verify_domain.py              검증
├── descriptors_v2_domain_6/run_params.json      최종 산출물의 실행 설정 (CSV 본체는 미포함)
└── plot/                    그림 · 그림 데이터 · 생성 스크립트
```

## 데이터 (저장소 미포함)

MC 궤적(약 14 GB)은 올리지 않았습니다. 재실행하려면 다음 위치에 둡니다:
`Ir_Ru_oxide/correct/MC_final_30cases_20260806/mc_runs/`

중간 산출물 CSV(최대 370 MB)도 미포함 — `run_descriptors_v2.py`로 다시 만들 수 있습니다(약 200초).

## 사용 방법

### 1. 설치

```bash
pip install -r requirements.txt     # Python 3.12에서 확인
```

### 2. MC 결과 전체를 한 번에 계산

입력은 케이스마다 폴더 하나입니다.

```
Ir_Ru_oxide/correct/MC_final_30cases_20260806/mc_runs/
└── ir_ru_sqs_x0p25_111_S0_Orich_dO+32_N_1040_normalz/   폴더명에서 조성·facet·O 조건을 읽음
    ├── surface.xyz                                      초기 slab — 층 라벨 · 이웃 목록을 여기서 한 번 정함
    └── run_*/
        ├── accepted_states.xyz                          수락된 MC 프레임
        └── results_every_step.json                      스텝 기록 — 프레임별 점유 스텝 구간
```

```bash
cd v2
python run_descriptors_v2.py        # descriptor 계산 → descriptors_v2_domain_6/ (약 200초)
python verify_sro.py                # α 검증 6건
python verify_domain.py             # domain 검증 8건
python plot/plot_v2_report.py       # 그림
python plot/export_figure_data.py   # 그림 데이터 표
```

출력 폴더에 결과가 이미 있으면 덮어쓰지 않고 멈춥니다.
폴더명 형식이 다르면 그 케이스는 건너뜁니다(`run_descriptors_v2.py`의 `CASE_RE`).
전제: 원소는 Ir/Ru, vacuum 축이 하나인 slab, 원자층 4개 이상(위·아래에 표면 + 차표면).

### 3. 출력 파일 (`descriptors_v2_domain_6/`)

| 파일 | 한 행 | 주요 칼럼 |
|---|---|---|
| `per_frame.csv` | 프레임 1개 | 깊이별 조성 `x_{TS,BS,TSS,BSS}` `S_surf_avg` `S_sub_avg` `A_tb` `G_star` · domain `D_{Ir,Ru}_{R}` `ncl_` `H_` `Pa_` `Pb_` `Pany_` `I_{R}` · 점유 스텝 `step_from` `step_to` |
| `layer_profile.csv` | (프레임, 층) | `x` = 그 층의 Ir 비율 (`layer` 0 = 좌표가 가장 작은 층) |
| `sro_long.csv` | (프레임, 껍질) | `quantity` = `alpha_in` / `alpha_in_region` / `alpha_out` · `m` · `shell_r`(Å) · `alpha` · `valid` · `in_spec` |
| `shell_pairs.csv` | (케이스, 껍질) | 껍질 거리 · pair 수 (프레임 무관) |
| `domain_layer.csv` | (프레임, 층, 원소) | `max_cluster` `n_cluster` `P_a` `P_b` |
| `domain_graph.csv` | (케이스, 층) | k=4 그래프의 간선 수 · degree (프레임 무관) |

`per_frame.csv`의 `C_lat_*` `D_dom_*` `S_depth` `G_layer` 등은 이전 정의(v1)와의 비교용이며 그림에는 쓰지 않습니다.

**평균 낼 때**: MC에서 거부된 스텝 동안에는 직전 구조가 유지되므로 프레임을 같은 무게로 평균하면 안 됩니다.
점유 스텝 수로 가중합니다.

```python
lo, hi = 601, 1100                                   # 평형 구간
w = (np.minimum(pf.step_to, hi) - np.maximum(pf.step_from, lo) + 1).clip(lower=0)
```

**주의**: `alpha`는 한 원소로만 된 층에서 정의되지 않아 NaN입니다(`valid == False`) — 행을 지우지 않으므로
`valid.mean()`이 유효 비율입니다. `sro_long.csv`는 370 MB라 `usecols`로 열을 골라 읽고,
`facet`은 `dtype={'facet': str}`로 읽어야 `'111'` 비교가 됩니다.

## 검증 요약

| 범주 | 손계산 | 외부 대조 | 그 밖에 |
|---|---|---|---|
| 깊이별 조성 | (100)·(111) 각 1프레임, 전 항목 일치 | MC 산출물에 포함된 region·층 조성 표와 일치, top/bottom 방향 30/30 | 항등식 9종, 21,820 프레임 전부 |
| α_in / α_out | (100)·(111) 6항목 일치 | MC 산출물에 포함된 SRO 표(`analysis_*/sro_*_table.csv`)와 이종 pair 비율 30/30 일치 (최대오차 1e-16) | Warren–Cowley 항등식 · 인공 배열 부호 |
| Domain | D·H·I_R 손계산 일치, P는 다른 알고리즘(복제 셀)으로 대조 | 대응하는 표가 없음 → 독립 재구현(간선 집합 재구성)으로 대체 | 항등식 · 결측 규약 |

## 스펙과 다르게 / 추가로 계산한 것

- (111) 면내 2번째 껍질(6.41 Å)이 스펙 상한 6.0 Å 밖 → 상한 6.6 Å으로 변경
- (111)의 최근접 결합(3.15 Å)이 두 층 건너(Δlayer=2) → α_out을 Δlayer 1–3까지 계산.
  스펙 그대로의 값은 `sro_long.csv`의 `in_spec == True`
- (110)은 k=4의 4번째 이웃이 4중 동률 → 동률을 모두 포함(이웃 6개)

## 참고

- descriptor 정의의 출처는 협업 연구실의 정리 문서(2026-08-27)이며 저장소에 포함하지 않았습니다.
- 환경: Python 3.12.13 · numpy 2.5.1 · pandas 3.0.5 · ase 3.29.0 · scipy 1.18.0 · matplotlib 3.11.1 · SciencePlots 2.2.2
