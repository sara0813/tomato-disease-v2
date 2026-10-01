# 경량 CNN 기반 토마토 잎 병해 분류 및 강건성 평가

> 모델은 더 작게 · 분류 문제는 더 어렵게 · 성능은 안정적으로

실제 촬영 환경의 어려운 이미지에서도 쓸 수 있는 **작고 빠른** 토마토 병해 분류 모델을 직접 설계하고,
대형 모델과 비교해 **정확도 · 효율성 · 강건성 · 일반화**의 균형을 검증하는 프로젝트. PyTorch 기반.

## 진행 상태 (2026-10-01 기준)

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | 경량 모델 설계 + 학습 (Tiny CNN A/B/C) | ✅ 완료 |
| 2 | 내부 성능 평가 | ✅ 완료 |
| 3 | 효율성 평가 | ✅ 완료 |
| 4 | Corruption 강건성 평가 | ✅ 완료 (seed 42/123/2026 반복) |
| 5 | 외부 데이터 평가 | ✅ 완료 |
| 5.5 | Seed 재현성 (42/123/2026) | ✅ 완료 |
| 5.6 | 해상도 × 리사이즈 보간법 튜닝 (tiny_cnn_c) | ✅ 완료 |
| 6 | 최종 모델 선정 | ✅ 완료 — **tiny_cnn_c, 96px, bicubic 리사이즈** |
| 7 | 웹 시스템 적용 | ✅ 프로토타입 완료 (`app/streamlit_app.py`, 모델 선택형) |
| 8 | Domain Adaptation (AdaBN/TENT/SHOT) | ✅ 완료 (seed42 단일) — 마스킹+SHOT 조합이 최종 권장안 (아래 참고) |

진행 과정과 판단 근거를 상세히 정리한 슬라이드: `docs/CNN_토마토_병해분류_V2_진행보고.pptx`
(2026-10-02 기준 28장 — 최종 모델 선정·seed 재현성·Domain Adaptation 최종 결과 반영)

## 연구 질문

| | 질문 |
|---|---|
| RQ1 | 대형 사전학습 모델보다 훨씬 작은 CNN으로 유사한 내부 분류 성능을 얻을 수 있는가? |
| RQ2 | 작은 모델이 조명 변화, 블러, 노이즈 및 부분 가림에도 안정적으로 분류할 수 있는가? |
| RQ3 | 모델 크기와 정확도 사이에서 가장 효율적인 구조는 무엇인가? |
| RQ4 | PlantVillage로 학습한 경량 모델이 Taiwan · Bangladesh · PlantDoc 데이터에도 일반화되는가? |

## 1차 실험(V1)에서 확인한 문제

내부 테스트 정확도는 최대 86.21%였지만 외부 데이터에서는 **9.46 ~ 30.89%로 급락**했다.
다수 모델이 외부 이미지를 Late Blight로 편향 예측했다. 즉 핵심 문제는 학습 부족이 아니라
**실제 환경 일반화 실패(domain shift)** 다. Taiwan에서는 가장 단순한 Baseline CNN이 1위였다.

| 모델 | 내부 Accuracy | Taiwan | Bangladesh BBox |
|---|---|---|---|
| EfficientNetB0 | 86.21% | 25.16% | 18.09% |
| MobileNetV2 | 84.42% | 29.30% | 12.95% |
| Baseline CNN | 83.32% | **30.89%** | 9.46% |
| EfficientNetB0 + CW | 82.30% | 23.25% | 12.67% |
| DenseNet121 | 80.58% | 28.98% | 11.02% |

이 4개 모델(+ MobileNetV2)은 **V2에서 재학습하지 않는다** — CPU로 7개 모델 전부 재학습하면 약 25시간이
걸리고, V1이 이미 이 질문에 답을 냈기 때문이다. V2는 위 표의 수치를 인용 기준선(`config.CITED_REFERENCE_MODELS`,
`results/summary/v1_cited_reference.json`)으로 쓰고, **직접 설계한 Tiny CNN 3종에만 학습·평가를 집중**한다.

## V2 결과 요약

### 내부 성능 (`results/internal/`)

| 모델 | 파라미터 | 내부 test Accuracy (seed42) | seed 42/123/2026 평균 ± 표준편차 |
|---|---|---|---|
| tiny_cnn_a (최소형) | 94,762 | 96.59% | 96.36% ± 0.46 |
| **tiny_cnn_b (중간형)** | 242,474 | **98.42%** | **97.64% ± 0.64** |
| tiny_cnn_c (Depthwise) | **31,498** | 96.26% | 96.46% ± **0.25** (가장 안정적) |

V1 최고 기록(EfficientNetB0 86.21%)보다 세 모델 모두 높다. `tiny_cnn_b`가 가장 정확하지만
seed 간 편차도 가장 크고(±0.64), `tiny_cnn_c`는 `tiny_cnn_b`의 13% 크기로도 평균 1.18%p
이내 차이면서 재현성(seed 안정성)은 셋 중 가장 좋다 — Depthwise Separable Conv의 설계
의도가 실측으로 확인됨(RQ1, RQ3).

### 효율성 (`results/efficiency/`)

| 모델 | FLOPs | 파일 크기 | CPU 추론(batch=1) |
|---|---|---|---|
| tiny_cnn_a | 330.3M | 0.37MB | 3.88ms |
| tiny_cnn_b | 405.8M | 0.94MB | 7.08ms |
| **tiny_cnn_c** | **74.4M** | **0.13MB** | 6.20ms |

### Corruption 강건성 (`results/corruption/`) — 조건별 평균 성능 저하율

| 조건 | 평균 저하율 | 비고 |
|---|---|---|
| reflection | 7.21% | 가장 잘 버팀 |
| occlusion | 13.34% | |
| shadow | 14.63% | |
| noise | 34.32% | |
| blur | 48.46% | |
| brightness | 60.41% | 가장 취약 (학습에 밝기 증강 안 씀) |

모델별 평균 저하율(seed42): tiny_cnn_b 25.91% (최선) · tiny_cnn_c 30.35% · tiny_cnn_a 32.92%.
seed 42/123/2026 3개 평균으로는 tiny_cnn_b 26.37% (최선) · tiny_cnn_c 29.74% · tiny_cnn_a 31.28%로
순위는 그대로 유지된다.

### 외부 데이터 일반화 (`results/external/`) — RQ4

| 모델 | 내부 | Taiwan (3/10 클래스) | Bangladesh (6/10 클래스) | PlantDoc (8/10 클래스) |
|---|---|---|---|---|
| tiny_cnn_a | 96.59% | 26.75% | 17.62% | 23.70% |
| tiny_cnn_b | 98.42% | 29.30% | 18.26% | 23.55% |
| tiny_cnn_c | 96.26% | 25.16% | 15.44% | 20.03% |

**경량화해도 domain shift 자체는 해결되지 않는다.** V1의 대형 모델과 비슷한 범위로 급락 — 모델
크기의 문제가 아니라 실제 환경 일반화 자체가 남은 과제라는 프로젝트의 핵심 문제의식이 재확인됨.
특히 PlantDoc은 클래스 커버리지가 Taiwan(3개)보다 훨씬 넓은데도(8개) 정확도가 비슷한 ~20%대에
머문다 — 클래스가 안 겹쳐서 낮은 게 아니라 **진짜 도메인 시프트(실제 촬영 조건) 문제**라는 뜻.

## 최종 모델 선정: tiny_cnn_c, 96px, bicubic 리사이즈

**1단계 — 구조 선택 (A/B/C).** tiny_cnn_b가 정확도·corruption 강건성에서 계속 1위지만 격차가
크지 않다(정확도 +1.18%p, corruption 저하율 +3.4%p). 반면 tiny_cnn_c는 파라미터 1/8, FLOPs 1/5.5면서
seed 재현성은 셋 중 가장 좋다. 이 프로젝트의 목표가 경량화인 만큼, 작은 성능 손실로 8배 작은 모델을
얻는 tiny_cnn_c를 최종 구조로 선택했다.

**2단계 — 해상도 × 보간법 튜닝.** tiny_cnn_c를 64/96/128/224px × bilinear/bicubic/lanczos/box(≈OpenCV area)
13개 조합(224px는 학습 1회 5시간이 걸려 bilinear만 측정)으로 비교했다 — 관련 그래프:
`results/figures/resolution_interp_matrix_tiny_cnn_c.png`(내부 정확도), `external_interp_matrix_tiny_cnn_c.png`
(외부 평균), `training_time_matrix_tiny_cnn_c.png`(학습 시간).

| 설정 | 내부 정확도 | 외부 평균 | 학습 시간 |
|---|---|---|---|
| 128px bilinear (기존 기준) | 96.26% | 20.21% | 68.5분 |
| 128px box (내부 정확도 최고) | **96.96%** | 21.89% | 113.3분 (가장 느림) |
| **96px bicubic (최종 선택)** | 96.59% | **23.02%** | 75.9분 |
| 128px bicubic | 96.66% | 23.31% (전체 1위) | 95.7분 |

128px box가 내부 정확도는 가장 높지만 외부 일반화는 중간 수준이고 학습도 가장 오래 걸려 제외했다.
(box는 torchvision/PIL에 OpenCV의 area 보간이 따로 없어서 같은 개념으로 대응시킨 필터다 — `src/dataset.py`의 `INTERPOLATION_MODES` 참고.)
96px bicubic은 128px bicubic(13개 조합 중 외부 1위)과 내부·외부 정확도 모두 0.1~0.3%p밖에 차이
안 나면서 학습 시간은 21% 더 짧다 — 해상도가 낮아 추론 연산량도 비례해 줄어들므로(96²/128²≈0.56배)
경량화 목표에 가장 부합하는 균형점으로 최종 선택했다. 참고로 이 작은 모델은 CPU + `num_workers=0`
환경에서 이미지 디코딩 비용이 conv 연산 비용을 압도해서, 96px과 128px의 학습 시간이 리사이즈
방법에 따라서는 거의 같게 나오기도 한다(bilinear 기준 68.1분 vs 68.5분) — 해상도를 낮춘다고 학습
시간이 항상 비례해서 줄지는 않는다는 것도 함께 확인된 점이다.

### 그룹 분할 전/후 비교 (leakage 수정 효과 실측)

seed42를 그룹 단위 분할(위 [그룹 단위 분할](#그룹-단위-분할-leakage-방지) 참고)로 재학습해서, 수정
전(`*_predup`, 구 split) 대비 실제로 수치가 얼마나 바뀌었는지 확인했다.

| 모델 | 내부 Acc (전→후) | Corruption 평균 저하율 (전→후) | Taiwan (전→후) | Bangladesh (전→후) | PlantDoc (전→후) |
|---|---|---|---|---|---|
| tiny_cnn_a | 94.99%→96.59% (+1.60%p) | 34.11%→32.92% (−1.19%p) | 20.38%→26.75% | 15.26%→17.62% | 20.03%→23.70% |
| tiny_cnn_b | 97.66%→98.42% (+0.76%p) | 27.83%→25.91% (−1.91%p) | 28.03%→29.30% | 17.89%→18.26% | 20.87%→23.55% |
| tiny_cnn_c | 97.29%→96.26% (−1.03%p) | 30.52%→30.35% (−0.17%p) | 27.39%→25.16% | 15.17%→15.44% | 20.87%→20.03% |

내부/corruption은 세 모델 모두 ±2%p 이내(A·B는 소폭 상승, C는 소폭 하락)로, 애초에 걸러낸 중복이
전체의 0.15%(28장/18,160장)뿐이라 **모델 순위나 결론(RQ1~RQ4)에 영향을 줄 정도의 차이가 아님**을
확인했다. 외부 데이터는 A·B는 3개 외부셋 모두 상승, C는 혼조(Taiwan·PlantDoc 소폭 하락, Bangladesh
소폭 상승)인데, 외부셋은 PlantVillage 내부 분할과 무관한 완전히 별도의 데이터라 이 변동은 그룹 분할
자체의 효과가 아니라 재학습에 따른 원래 랜덤 변동 범위로 해석하는 게 맞다. 구 결과(`seed42_predup/`)는
비교 근거로 당분간 남겨두고, 확인이 끝나면 삭제할 예정이다.

## Domain Adaptation (AdaBN / TENT / SHOT) — 최종 정리

외부 일반화(RQ4)가 여전히 낮은 문제를 완화하기 위해, 최종 모델(tiny_cnn_c, 96px, bicubic)에
source-free·target-label 미사용 조건으로 AdaBN·TENT·SHOT을 순서대로 적용해봤다. 자세한 실험 설계와
교수님 확인이 필요한 가정은 `docs/토마토_병해분류_Domain_Adaptation_실행계획_v2.md` 참고, 코드는
`src/adapt/domain_adaptation.py`. **아래 결과는 전부 seed42 단일 실행 기준이다 — 3-seed 재현성 검증은
아직 못 했고 향후 과제로 남겨둔다.**

### 1차 시도: AdaBN·TENT (마스킹 없이) → negative transfer

| 데이터셋 | 클래스 겹침 | Source-only | AdaBN | TENT |
|---|---|---|---|---|
| Taiwan | 3/10 | 27.39% | 7.64% | 8.28% |
| Bangladesh | 6/10 | 21.80% | 10.17% | 8.63% |
| PlantDoc | 8/10 | 19.89% | 13.12% | 11.14% |

**둘 다 기대와 반대로 source-only보다 성능이 떨어졌다.** 예측 분포를 분석해보니 원인은 명확하다 —
AdaBN/TENT 모두 PlantVillage 10개 클래스 전체 기준으로 BN 통계/예측을 재조정하는데, 외부 데이터는
일부 클래스만 존재해서(partial label-space) 존재하지 않는 클래스 쪽으로 예측이 쏠리는
**negative transfer**가 발생했다(근거: `results/figures/negative_transfer_evidence.png` — "target에
없는 클래스로 예측된 비율"이 AdaBN/TENT 적용 후 Taiwan 기준 21.7%→83.1%/78.0%로 급등). 게다가
**클래스 겹침이 적을수록 저하 폭이 더 크다**(Taiwan > Bangladesh > PlantDoc) — 설계 문서가 PADA
논문을 인용해 이론적으로 경고했던 partial-domain-adaptation 문제가 실측으로 정확히 재현된 것이다.
구현 정확성은 BN 통계/파라미터 변화량을 직접 검증해 확인했다(AdaBN은 conv weight 불변·BN 통계만
변화, TENT는 conv weight 불변·BN affine만 변화).

### 해결책: Class-restriction 마스킹

`class_info.py`에 설계 단계부터 있던 클래스 매핑표(target ground truth가 아니라 어떤 클래스가
존재할 수 있는지에 대한 사전 정보)로, target에 없는 클래스의 로짓에 `-inf`를 더해 예측 후보에서
제외한다(추가 학습 없음, zero-cost). 모든 데이터셋에서 마스킹 단독만으로도 source-only보다 개선됐다.

| 데이터셋 | Source-only | +마스킹만 | AdaBN+마스킹 | TENT+마스킹(mask-aware) |
|---|---|---|---|---|
| Taiwan | 27.39% | 32.48% | **38.85%** | 29.94% |
| Bangladesh | 21.80% | **22.34%** | 15.71% | 13.71% |
| PlantDoc | 19.89% | **20.31%** | 14.81% | 12.83% |

TENT는 entropy를 마스킹된 클래스 안에서만 최소화하도록 고쳐도(mask-aware) 처음엔 "한 클래스로만
100% 확신" 하는 degenerate solution으로 collapse했다(`prediction_distribution.csv`로 확인) — batch
다양성을 최대화하는 정규화 항(SHOT의 information-maximization 축소판)을 추가해 collapse는 고쳤지만,
그래도 마스킹만 한 것보다 전부 나빠서 최종안에서는 제외했다. AdaBN을 momentum 블렌딩(0.05~0.3)으로
완전 교체 대신 약하게 섞어도 Bangladesh·PlantDoc은 마스킹-only를 넘지 못했다 — BN 통계를 조금이라도
건드리는 것 자체가 해로웠다는 뜻이다. Taiwan만 완전 교체(AdaBN)가 확실히 더 좋았다.

### 최종 승자: SHOT (pseudo-labeling + information maximization)

Liang et al., *Do We Really Need to Access the Source Data? Source Hypothesis Transfer for
Unsupervised Domain Adaptation*, ICML 2020. source classifier(`head.fc`)는 고정하고 feature
extractor(`features`)만 미세조정한다 — 처음엔 분류기 softmax를 가중치 삼은 weighted k-means로
클래스별 중심을 구해 pseudo-label을 할당(마스킹된 클래스 안에서만), 그 다음 pseudo-label
cross-entropy + information-maximization(entropy 최소화 + batch 다양성 최대화) loss로 학습한다.

| 데이터셋 | Source+마스킹 | AdaBN+마스킹 | TENT+마스킹 | **SHOT+마스킹** |
|---|---|---|---|---|
| Taiwan | 32.48% | **38.85%** | 29.94% | 38.54% |
| Bangladesh | 22.34% | 15.71% | 13.71% | **27.97%** |
| PlantDoc | **20.31%** | 14.81% | 12.83% | 18.05% (macro F1 **0.173**, 전체 1위) |

AdaBN/TENT와 달리 **SHOT은 3개 데이터셋 전부에서 collapse 없이 안정적으로 잘 됐다** — Taiwan은
AdaBN과 거의 동률, Bangladesh는 전체 방법 통틀어 최고, PlantDoc은 정확도는 비슷해도 macro F1이
가장 높다. 데이터셋마다 다른 방법을 골라야 했던 AdaBN/TENT와 달리 튜닝 없이 두루 통하는 유일한
방법이라, **class-restriction 마스킹은 항상 적용 + 적응 방법은 SHOT을 기본으로 사용**하는 것을
최종 권장 파이프라인으로 정했다.

**남은 과제:** 지금은 seed42 단일 실행 결과다. `tiny_cnn_c`를 96px·bicubic 기준으로 seed123/2026에서
추가 학습한 뒤 이 비교 전체를 3-seed로 재검증할 계획이다(seed123 학습은 진행 중).

## 데이터셋

| 데이터 | 역할 | 규모 |
|---|---|---|
| PlantVillage Tomato | 학습 · 내부 평가 | 18,160장 / 10개 클래스 (70/15/15 그룹 단위 분할: train 12,706 / val 2,727 / test 2,727) |
| Taiwan Tomato | 외부 환경 평가 (학습 안 함) | 314장, 겹치는 3개 클래스만 사용 |
| Bangladesh Tomato Leaf | 외부 환경 + BBox 평가 (학습 안 함) | 1,101장, YOLO bbox + 40% 여백 + 최소 96px 크롭 |
| PlantDoc | 외부 환경 평가 (학습 안 함) | 709장, 8개 클래스 대응 (Google/Bing 이미지 검색 기반, 원본 746장 중 진단표·비교 콜라주 등 35장 제외) |
| Corruption 변형셋 | 4단계 강건성 평가 전용 | test셋 × 6조건 × 3단계 = 49,086장 |

품질 검증: 이미지 무결성(20,284장 전수, 콘텐츠 문제 0건) · 클래스 불균형 기록(보정 없이 자연 분포 사용,
14.36배) · train/test leakage 검사(perceptual hash, 근접중복 0.62~0.88%는 재분할 불필요 수준이었지만
완전 동일 이미지가 소수 발견되어(test-train 3장, val-train 4장) 그룹 단위 분할로 개선 후 재검증 결과
완전 동일 0건 — 아래 [그룹 단위 분할](#그룹-단위-분할-leakage-방지) 참고) · PlantDoc은 콘택트시트
수작업 검수로 잎 사진이 아닌 이미지(진단표/비교 콜라주/삽화) 35장을 걸러냄.

## 실험 로드맵

| 단계 | 내용 | 코드 | 산출물 |
|---|---|---|---|
| 1 | 경량 모델 설계 (Tiny CNN 3종) | `src/models/tiny_cnn.py`, `src/train/train_model.py` | `models/*/seed<seed>.pt`, `results/internal/*/seed<seed>/training_log.*` |
| 2 | 기본 성능 평가 | `src/evaluate/evaluate_internal.py` | `results/internal/` |
| 3 | 효율성 평가 | `src/evaluate/measure_efficiency.py` | `results/efficiency/` |
| 4 | Corruption 평가 | `src/evaluate/evaluate_corruption.py` | `results/corruption/` |
| 5 | 외부 데이터 평가 | `src/evaluate/evaluate_external.py` | `results/external/` |
| 6 | 최종 모델 선정 — tiny_cnn_c/96px/bicubic | `src/train/train_model.py --img-size --interp` | `results/figures/*_interp_matrix_tiny_cnn_c.png` |
| 7 | 시스템 적용 | `app/streamlit_app.py` | 웹 프로토타입 (모델 선택형) |
| 8 | Domain Adaptation (AdaBN/TENT/SHOT) | `src/adapt/domain_adaptation.py` | `results/external/*/*_adabn*/`, `*_tent*/`, `*_shot_masked/` |

**Corruption은 학습 증강이 아니라 평가용 변형이다.** 밝기 · 그림자 · 반사 · 블러 · 노이즈 · 가림
6개 조건 × 약/중/강 3단계로 **테스트 이미지만** 변형해 모델별 성능 저하율을 비교한다. 학습 데이터에는
전혀 섞이지 않는다(`data/processed/plantvillage/train/` vs `data/corrupted/`, 완전히 분리된 경로).

최종 모델은 최고 정확도 하나로 고르지 않는다.
① 내부 성능 ② 외부 일반화 ③ corruption 저하율 ④ 파라미터 수 ⑤ 모델 크기 ⑥ CPU 추론시간을 종합한다.

### 반복실험 (재현성) — 완료

`config.SEEDS = [42, 123, 2026]` 3개 seed로 tiny_cnn_a/b/c를 전부 재학습하고 내부·corruption·외부
평가까지 완료했다. 결과가 특정 seed의 우연이 아님을 확인했다 — 위 내부 성능·corruption 표의
"평균 ± 표준편차" 열 참고. `train_model.py` / `evaluate_*.py`는 모두 `--seed` 인자를 받고,
`config.model_path()`와 `*_result_dir()` 헬퍼가 seed별로 경로를 분리한다
(`models/<model>/seed<seed>.pt`, `results/<단계>/<model>/seed<seed>/`)
— 다른 seed로 다시 실행해도 기존 seed의 가중치·로그는 덮어쓰이지 않는다.

`src/summary/aggregate_seeds.py`가 `config.SEEDS` 중 완료된 seed를 모아 모델별 평균 Accuracy ·
표준편차 · 평균 Macro F1 · 최고/최저 성능을 계산해 `results/summary/seed_aggregate.{csv,md}`에
저장한다. **tiny_cnn_c가 세 모델 중 seed 재현성이 가장 좋다**(표준편차 0.25%p, tiny_cnn_b는 0.64%p)
— 최종 모델 선정에서 이 안정성도 함께 고려했다(아래 참고).

학습은 중간에 끊겨도 안전하다 — `train_model.py`가 매 epoch마다 체크포인트(모델/옵티마이저/rng 상태)를
저장하고, 같은 run을 다시 실행하면 자동으로 그 지점부터 이어서 학습한다(`--no-resume`으로 끄고 처음부터
새로 시작 가능). 실제로 세션이 끊긴 상황에서 정확히 이어받아 끝까지 완주하는 것까지 검증했다.

### 그룹 단위 분할 (leakage 방지)

`check_leakage.py` 진단에서 PlantVillage train/test 사이에 perceptual hash(dHash)가 **완전히 동일한**
이미지가 소수 발견됐다(test-train 3장, val-train 4장 — 근접 중복까지 포함하면 0.62~0.88%). 같은 사진이
train과 test에 나뉘어 들어가면 내부 테스트 정확도가 "새 이미지를 맞히는 능력"이 아니라 "본 이미지의
쌍둥이를 맞히는 능력"으로 부풀려질 수 있다.

**완료:** `split_plantvillage.py`를 파일 단위 무작위 분할에서 **그룹 단위 분할**로 개선하고, 실제로
데이터를 재생성 + seed42로 재학습까지 마쳤다.
1. `build_hash_groups()` — 클래스별로 dHash가 동일한 파일들을 그룹으로 묶는다 (중복 없으면 그룹 크기 1).
2. `split_groups()` — 그룹을 섞은 뒤, 목표 비율(70/15/15) 대비 가장 덜 채워진 split에 그룹째로 배정한다.
   같은 그룹은 항상 같은 split에만 들어가므로 완전 동일 이미지가 train/val/test로 나뉘는 일이 없어진다.
   (그룹을 통째로 옮기다 보니 클래스별 실제 비율이 70/15/15에서 아주 약간 벗어날 수 있는데, 그 오차는
   `results/_common/plantvillage_split_stats.json`에 클래스별로 기록된다.)
3. 중복 dHash 계산 로직은 `check_leakage.py`와 함께 쓰도록 `src/utils/phash.py`로 공통화했다.
4. `data/processed/plantvillage`와 `data/corrupted`를 새 split 기준으로 재생성하고, tiny_cnn_a/b/c를
   seed42로 재학습 + 전체 평가(internal/efficiency/corruption/external)까지 재실행했다. 구 결과는
   `seed42_predup/`에 남겨뒀고, [위 비교표](#그룹-분할-전후-비교-leakage-수정-효과-실측)에서 전/후
   차이가 크지 않음을 확인했다(±2%p 이내).
5. `check_leakage.py` 재검증 결과: 완전 동일 이미지 0건(기존 test-train 3건, val-train 4건 → 0).

**향후 계획:** seed 123, 2026 반복실험도 이 새 split(그룹 단위)을 그대로 사용한다. 구 결과
(`seed42_predup/`)는 비교 근거로 당분간 남겨두고, 정리가 끝나면 삭제할 예정이다.

## 폴더 구조

```
tomato_V2/
├── data/
│   ├── raw/              # 원본 (plantvillage, taiwan, bangladesh) — 절대 수정하지 않음
│   ├── processed/        # PlantVillage train/val/test 분할
│   ├── external/         # 외부 평가셋 (taiwan, bangladesh_bbox)
│   └── corrupted/        # corruption 변형 테스트셋 (49,212장)
├── src/
│   ├── config.py         # 경로·하이퍼파라미터·모델 그룹(TINY_MODELS/CITED_REFERENCE_MODELS) 전역 설정
│   ├── class_info.py     # 클래스 정의(한국어 설명 포함) + 외부 데이터 클래스 매핑
│   ├── dataset.py        # PyTorch Dataset/DataLoader (ImageFolder + 빈 클래스 허용용 커스텀 로더)
│   ├── data_prep/        # 분할 · 외부 데이터 변환 · 무결성/불균형/leakage 점검
│   ├── models/           # tiny_cnn(A/B/C, nn.Module) · reference(인용용) · 레지스트리
│   ├── corruption/       # 변형 정의(6조건×3단계) · 변형셋 생성
│   ├── train/            # train_model.py — 모델 이름 인자 하나로 통일, --img-size/--interp 튜닝, 체크포인트 재개
│   ├── evaluate/         # 내부 · 효율 · corruption · 외부 평가
│   ├── adapt/            # domain_adaptation.py — AdaBN/TENT/SHOT + class-restriction 마스킹 (source-free, target label 미사용)
│   ├── summary/          # 종합 비교표 (V2 실측 + V1 인용, 판단 없음)
│   ├── visualize/        # 결과 그래프 (results/figures/*.png)
│   └── utils/            # 시드 · 입출력
├── models/               # 학습된 .pt 가중치 (git 제외) — <model>/seed<seed>.pt
├── results/              # _common · internal · efficiency · corruption · external · summary · figures
│                         #   (efficiency·summary·figures 제외하고는 <model>/seed<seed>/ 로 분리 저장)
├── app/                  # Streamlit 웹 프로토타입 (모델 선택형, 정상/비정상 색상 표시)
├── notebooks/
└── docs/                 # 프로젝트 브리프 PDF · V2 진행 보고 PPTX · Domain Adaptation 실행계획 MD
```

## 실행 순서

```bash
# 가상환경 (Python 3.11)
py -3.11 -m venv .venv
.venv/Scripts/pip install -r requirements.txt

# 데이터 준비 (원본은 건드리지 않고 새 폴더에 생성)
# split_plantvillage.py는 dHash가 동일한 이미지를 그룹으로 묶어 같은 split에만 배정한다
python src/data_prep/split_plantvillage.py
python src/data_prep/prepare_taiwan.py
python src/data_prep/prepare_bangladesh_bbox.py
python src/data_prep/prepare_plantdoc.py
python src/data_prep/verify_image_integrity.py
python src/data_prep/check_dataset.py
python src/data_prep/check_leakage.py
python src/corruption/make_corrupted_sets.py

# 학습 — V2에서 실제로 학습하는 건 이 3개뿐 (config.TINY_MODELS)
# --seed 생략 시 config.SEED(42). 다른 seed로 반복실험하려면 --seed 123 등으로 지정
# (seed별로 별도 경로에 저장되므로 기존 결과를 덮어쓰지 않는다)
python src/train/train_model.py --model tiny_cnn_a
python src/train/train_model.py --model tiny_cnn_b
python src/train/train_model.py --model tiny_cnn_c

# 해상도 × 보간법 튜닝 (최종 모델 tiny_cnn_c/96px/bicubic을 고른 실험)
# --img-size/--interp 생략 시 기본(128px, bilinear). run_name에 자동으로 접미사가 붙어 분리 저장된다.
python src/train/train_model.py --model tiny_cnn_c --img-size 96 --interp bicubic
python src/evaluate/evaluate_internal.py --model tiny_cnn_c --img-size 96 --interp bicubic
python src/evaluate/evaluate_external.py --model tiny_cnn_c --img-size 96 --interp bicubic

# 평가 — internal/corruption/external은 --seed 지정 가능(생략 시 config.SEED).
# measure_efficiency는 seed와 무관(구조로만 결정되는 파라미터 수·FLOPs·크기 측정)이라 --seed 없음.
python src/evaluate/evaluate_internal.py
python src/evaluate/measure_efficiency.py
python src/evaluate/evaluate_corruption.py
python src/evaluate/evaluate_external.py

# Domain Adaptation (AdaBN/TENT/SHOT) — 최종 모델에 적용, target label 미사용
python src/adapt/domain_adaptation.py --model tiny_cnn_c --img-size 96 --interp bicubic --dataset all --method all --masked
# SHOT만 (마스킹이 방법 자체에 내장돼 있어 --masked 없어도 항상 적용됨)
python src/adapt/domain_adaptation.py --model tiny_cnn_c --img-size 96 --interp bicubic --dataset all --method shot

# 종합 + 시각화
python src/summary/make_summary.py
python src/summary/aggregate_seeds.py   # 완료된 seed만 모아 평균·표준편차·최고/최저 집계
python src/visualize/plot_results.py
python src/visualize/plot_class_distribution.py

# 웹 프로토타입
streamlit run app/streamlit_app.py
```

Windows에서 한글 출력이 깨지면 `PYTHONUTF8=1`을 앞에 붙여 실행한다.

## 실행 환경

로컬 CPU 또는 일반 Colab. 고성능 GPU를 전제하지 않는 크기를 목표로 한다 — Tiny CNN 3종은 CPU
기준 128px 학습 시 모델당 1~2.5시간 내외(tiny_cnn_c 68분 ~ tiny_cnn_b 153분, early stopping 포함),
추론은 이미지당 5ms 미만이다. 해상도를 올리면 학습 시간이 크게 늘 수 있다(224px는 약 5시간/회) —
자세한 해상도별 실측은 `results/figures/training_time_matrix_tiny_cnn_c.png` 참고.
