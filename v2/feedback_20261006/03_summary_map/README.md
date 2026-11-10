# 요약 지도 — 구조 1개 = 점 1개

피드백 "그림 하나에 여러 정보가 들어가게 (구조 하나 = 점 하나)"에 대한 그림입니다.

## 실행

```bash
cd v2/feedback_20261006/03_summary_map
python plot_summary_pairmap.py   # 그림 C
python plot_summary_map.py       # 그림 B1
```

입력은 `../02_timeseries/`의 `block_se.csv` · `block_means.csv`입니다. 그림은 PNG · PDF가 모두 있으면 멈춥니다.

## 그림

| 파일 | 내용 |
|---|---|
| `figures/fig_C_pairmap` | 행 = facet, 열 = ① S_surf(avg) × S_sub(avg) ② α_in 표면 m = 1 × m = 2 ③ α_in 내부 m = 1 × m = 2 |
| `figures/fig_B1` | S_surf(avg) × S_sub(avg) + 꼬리 (MC 101–200 평균 → 601–1100 평균) |

## 그림 읽는 법

- 점 = MC 601–1100 점유 가중 평균, 오차막대 = block SE(평균의 오차, 하한 — `../02_timeseries/README.md`). 마커보다 작으면 가려집니다
- 색 = O 조건, 기호 = x_Ir, 속빈 기호 = (111) stoich의 두 번째 slab(S2)
- 그림 C의 α 두 열은 같은 축 범위입니다 → 내부가 원점 근처에 모인 것을 표면과 같은 눈금에서 비교할 수 있습니다
- "undefined" = α_in 표면이 정의되지 않는 구조 (x_Ir 0.25 (111) 3구조 — 표면층이 거의 Ru로만 채워짐)
- 꼬리의 시작을 101–200으로 둔 이유: 1–100은 adaptive 구간입니다
- 대표 8개 중 G* · A_tb는 지도에 없습니다 — 짝을 지어 그려도 패턴이 읽히지 않았고, (111)의 G*는 2/3가 bulk 층 사이 차이라 facet마다 뜻이 다릅니다
