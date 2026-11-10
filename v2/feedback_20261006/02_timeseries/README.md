# 시간 변화 — 대표 descriptor의 MC step에 따른 변화

피드백 "한 구조가 시간에 따라 어떻게 달라지는지 보일 것 (601–1100 평균이 구조를 대표하지 않을 수 있다)"에 대한 그림입니다.
대표 descriptor 8개와 energy를 100 step 블록 평균으로 그렸습니다.

## 실행

```bash
cd v2/feedback_20261006/02_timeseries
python build_timeseries.py   # ../../results/ → block_means.csv · block_se.csv
python plot_timeseries.py    # figures/F1_facet/ (facet별 3패널)
python plot_F1_grid.py       # figures/F1_grid/  (3×3 판)
```

결과 CSV가 이미 있으면 덮어쓰지 않고 멈춥니다. 그림은 PNG · PDF가 모두 있으면 건너뜁니다.

## 파일

| 파일 | 한 행 | 내용 |
|---|---|---|
| `block_means.csv` | 구조 × 계열 × 블록 | 블록 평균 `mean` · `sd` · `valid_fraction` |
| `block_se.csv` | 구조 × 계열 | 601–1100 평균 `m_601` · `block_se` · `n_blocks` · `n_changes` · `no_change` — 03 지도의 점과 오차막대 |
| `figures/F1_facet/fig_F1_<계열>` | — | facet별 3패널 (PNG · PDF, 9개씩) |
| `figures/F1_grid/fig_F1grid_<계열>` | — | 행 = facet, 열 = O 조건 3×3 판 (PNG · PDF, 9개씩) |

계열: `S_surf_avg` · `S_sub_avg` · `A_tb`(부호 있음) · `G_star` · `a_in_surf_m{1,2}`(α_in 표면) · `a_in_bulk_m{1,2}`(α_in 내부) · `energy_meV_atom`.
CSV에는 표면 각각의 `a_in_TS_m*` · `a_in_BS_m*`도 있습니다(그림에는 없음).

## 집계 규칙

- 블록 0 = 1–100(adaptive), 1–10 = 101–200 … 1001–1100. 블록과 겹치는 점유 step 수로 가중합니다(그림 v2와 같은 가중)
- α가 정의되지 않는 step(층이 한 원소뿐)은 평균에서 빼고 `valid_fraction`에 남깁니다
- `block_se` = 601–1100의 블록 평균 5개의 SD(ddof = 1) / √5. 블록 사이 상관이 남을 수 있어 **하한**으로 읽습니다.
  `n_changes`(601–1100 동안 값이 바뀐 횟수)가 적으면 믿기 어렵고, 0이면 `no_change` — 이때 `block_se` = 0은 "오차 없음"이 아니라 "잴 정보 없음"

## 그림 읽는 법

- 점 = 100 step 블록 평균(블록 가운데 step). 회색 바탕 = adaptive 구간 1–100, 점선 = 601(평균 구간 시작)
- 색 = O 조건, 기호 = x_Ir, 파선 = (111) stoich의 두 번째 slab(S2)
- 회색 채움 기호 = α가 정의된 step이 절반 미만인 블록. "undefined" = α가 아예 정의되지 않는 구조 수
- energy = 구조마다 마지막 블록 평균을 뺀 값 — 끝에서 0으로 모이는 것은 이 정의 때문입니다
- α 내부는 α 표면과 같은 y축 범위로 그렸습니다
