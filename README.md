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
| 8 | Domain Adaptation (AdaBN/TENT/SHOT/EATA/UPA/R-GV/AaD) | ✅ 완료 (3-seed 검증 완료) — 마스킹+SHOT(epoch=2) 조합이 최종 권장안(APS·데이터셋별 극단 튜닝·AaD 모두 noise/붕괴/무개선으로 기각), 데이터셋별 심층 진단 포함 (아래 참고) |

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

| 데이터셋 | Source-only | +마스킹만 | AdaBN+마스킹 |
|---|---|---|---|
| Taiwan | 27.39% | 32.48% | **38.85%** |
| Bangladesh | 21.80% | **22.34%** | 15.71% |
| PlantDoc | 19.89% | **20.31%** | 14.81% |

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
cross-entropy + information-maximization(entropy 최소화 + batch 다양성 최대화) loss로 학습한다
(`--shot-epochs`, `--shot-cls-weight`로 조정 가능, 아래 두 하이퍼파라미터 설명 참고). 아래
"UPA의 APS" 절에서 설명할 신뢰도 기반 pseudo-label 재할당(`--shot-aps --shot-aps-gamma 1.0`)도
시도했으나, **3-seed 검증 결과 seed42에서만 우연히 좋아 보였을 뿐 평균적으로는 개선이 아니었다**
(아래 "3-seed 검증 결과" 절 참고) — 그래서 APS는 기본값에 포함하지 않고 선택적 옵션으로만 남긴다.

아래 표는 seed42 단일 실행 기준이다(괄호의 F1은 seed42 기준). **3-seed 평균±표준편차는 바로 아래
"3-seed 검증 결과" 절을 참고** — 표준편차가 커서(특히 Taiwan·Bangladesh ±5~6%p) 이 표의 개별
수치를 "정확한 성능"으로 과신하면 안 된다.

| 데이터셋 | Source+마스킹 | AdaBN+마스킹 | **SHOT+마스킹(epoch=2, 최종)** | SHOT+APS+마스킹 | EATA+마스킹 |
|---|---|---|---|---|---|
| Taiwan | 32.48% | 38.85% | **42.04%** | 44.59%(F1 0.426) | 35.35% |
| Bangladesh | 22.34% | 15.71% | **30.79%** | 30.52%(거의 동일) | 20.89% |
| PlantDoc | 20.31% | 14.81% | **19.46%(F1 0.182)** | 19.04%(F1 0.180) | 12.27% |

AdaBN/TENT와 달리 **SHOT은 3개 데이터셋 전부에서 collapse 없이 안정적으로 잘 됐다** — Taiwan·
Bangladesh 둘 다 AdaBN을 앞서고, PlantDoc은 정확도도 가장 높고 macro F1도 전체 1위다. 데이터셋마다
다른 방법을 골라야 했던 AdaBN/TENT와 달리 튜닝 없이 두루 통하는 유일한 방법이라, **class-restriction
마스킹은 항상 적용 + 적응 방법은 SHOT(epoch=2)만 기본으로 사용**하는 것을 최종 권장 파이프라인으로
정했다.

**epoch=2로 고정한 이유 (hyperparameter 선택의 공정성)**: SHOT은 epoch 수를 늘릴수록 더 많이
학습하는데, `results/figures/shot_epoch_sensitivity.png`에서 보듯 **Taiwan(클래스 3개)은 epoch을
늘릴수록 계속 좋아지는 반면(epoch 5까지 시험 시 57%대) Bangladesh(6개)·PlantDoc(8개)은 epoch 2~3
근처에서 정점을 찍고 이후 내려간다** — SHOT이 자기가 만든 pseudo-label을 계속 더 믿고 학습하면서
생기는 과적합인데, 클래스가 많고 불균형할수록 pseudo-label 오류가 누적되기 쉽기 때문으로 보인다.
**데이터셋마다 가장 높은 epoch을 따로 골라 보고하면 target accuracy를 보고 몰래 튜닝한 셈이라
label-free라는 전제와 모순된다** — 그래서 "SHOT은 원래 TENT보다 더 학습이 필요하지만 가벼운
적응이라는 프로젝트 기조상 최소한만 늘린다"는 사전 기준으로 3개 데이터셋 공통 epoch=2를 고정값으로
썼다. epoch별 전체 수치는 `results/external/*/tiny_cnn_c_res96_bicubic_shot_masked_ep{1,3}_cw0.3/`에
남겨뒀다.

### 심층 진단: SHOT은 정확히 어디서 성공·실패하는가

epoch 단위 비교만으로는 "왜" 성공/실패하는지 알 수 없어서, target label을 **적응에는 전혀 안 쓰고
진단에만** 쓰는 세 가지 분석을 추가했다(`src/adapt/shot_stage_analysis.py`,
`src/adapt/shot_oracle_upperbound.py`).

**① pseudo-label 품질 추적**: epoch마다 pseudo-label 정확도와 그걸로 학습한 뒤 모델 정확도를
같이 기록했다. Taiwan은 pseudo-label 정확도가 epoch이 지날수록 계속 올라가고(34%→53%) 모델
정확도도 같이 올라간다 — "좋은 pseudo-label → 좋은 적응"의 선순환. Bangladesh는 epoch 2~3 근처에서
둘 다 같이 정체된다. **PlantDoc은 pseudo-label 정확도가 처음부터 끝까지 19% 근처에서 평평하다** —
더 학습해도 pseudo-label 품질 자체가 안 올라간다.

**② 초기 클래스별 precision/recall**: source softmax와 SHOT의 첫 pseudo-label을 클래스별로 비교했다.
- **Taiwan**: "healthy"가 source·클러스터링 둘 다에서 precision·recall **0%** — 314장 중 Late_blight로
  308장이 쏠리는 degenerate 상태이고, healthy 진짜 이미지 대부분이 거기 흡수됐다. SHOT을 거쳐도
  healthy 예측은 1장→10장으로 찔끔 늘 뿐 여전히 사각지대다.
- **Bangladesh**: SHOT 클러스터링이 "교환"을 일으킨다 — 원래 한 번도 예측 안 되던 Leaf_Mold(recall
  0%→25%)·Target_Spot(0%→43%)을 찾아내는 대신, 원래 잘 맞추던 healthy(recall 12%→20%는 오히려
  나아짐이지만)·Late_blight(recall 60%→22%)가 깎인다. 정확히는 "Late_blight 쏠림이 Target_Spot
  쏠림으로 옮겨간 것"에 더 가깝다(Target_Spot 예측 개수 0→211장, 진짜는 47장뿐이라 대부분 틀림).
- **PlantDoc**: source 단계에서 이미 Bacterial_spot·Leaf_Mold·healthy 3개 클래스가 precision·recall
  **0%**(완전한 사각지대)다. SHOT 클러스터링을 거치면 이 3개는 0%에서는 벗어나지만, 원래 잘하던
  Early_blight·Late_blight·Septoria의 recall이 전부 거의 반토막난다 — 득실이 상쇄돼 전체 정확도가
  epoch을 늘려도 거의 안 움직이는 이유다.

**③ Oracle 상한 실험** (`shot_oracle_upperbound.py`, 진단 전용 — 학습/평가 이미지는 stratified
70/30으로 분리): pseudo-label 대신 **실제 target 정답**으로 SHOT을 fine-tuning하면 어디까지
가능한지 측정했다.

| 데이터셋 | Source(eval split) | **Oracle(정답으로 학습)** |
|---|---|---|
| Taiwan | 33.68% | **70.53%** (+36.9%p) |
| Bangladesh | 22.96% | **58.61%** (+35.7%p) |
| PlantDoc | 23.47% | **23.47%** (변화 없음) |

Taiwan·Bangladesh는 정답만 있으면 정확도가 2배 이상 뛴다 — **pseudo-label 품질이 핵심 병목**이라는
뜻이다. **PlantDoc은 정답을 줘도 전혀 개선되지 않았다** — train 정확도는 19%→27%로 분명히 오르는데
(학습 자체는 정상, gradient·loss 확인 완료) eval 정확도는 그대로라 **train 데이터에만 과적합되고
새 이미지로 전혀 일반화가 안 된다는 뜻**이다. 이건 pseudo-label 문제가 아니라 **tiny_cnn_c의 feature
extractor가 PlantDoc의 시각적 다양성을 구분할 표현력 자체가 부족하다**는 신호로 해석한다.

### 추가 시도: UPA의 APS — Taiwan만 개선, CACL은 실패

**UPA**(Chen et al., *Uncertainty-Aware Pseudo-Label Filtering for Source-Free Unsupervised Domain
Adaptation*, Neurocomputing 2024, 공식 코드 [`chenxi52/UPA`](https://github.com/chenxi52/UPA))의
두 구성요소를 분리해서 시도했다.

**APS (Adaptive Pseudo-label Selection)**: 각 샘플의 K-최근접 이웃(K=4, 코사인 유사도)이 투표해
pseudo-label을 다듬고(2회 반복), 클래스별 confidence 상위 gamma%만 cross-entropy에 반영한다.
논문 기본값(gamma=0.6, 40% 필터링)은 오히려 SHOT보다 나빴다 — Taiwan에서 gamma를 0.6→1.0으로
스윕해보니 **필터링을 아예 끄고 KNN 라벨 재할당만 쓴 gamma=1.0이 최고**였다(42.04%→44.59%). 대규모
벤치마크용 필터링 강도가 300~1,100장 규모의 우리 데이터엔 너무 공격적이었던 것 — EATA의 lr
재조정과 같은 패턴이다. Bangladesh/PlantDoc은 APS를 켜도 거의 중립이었다.

**CACL (Class-Aware Contrastive Learning)**: APS의 confident set 안에서 같은 pseudo-label끼리
feature를 가깝게, 배치 내 다른 샘플과는 멀게 당기는 contrastive loss(temperature=0.1). Taiwan은
learning rate를 100배 낮춰도(1e-5) APS 단독(44.59%)보다 계속 나빴고(33~38%대), Bangladesh·
PlantDoc은 중립이었다. **고정된 classifier(`head.fc`)와 contrastive loss로 움직이는 feature space가
구조적으로 안 맞는 것으로 보인다** — CACL은 최종 파이프라인에서 제외했다.

**Bangladesh의 trade-off를 줄여보려는 추가 시도**(class 역빈도 가중치, APS+class-balance 조합,
epoch 3/5, cls_weight=1.0)도 전부 기존 SHOT(30.79%)을 못 넘었다 — Target_Spot recall은
51%→57%로 개선됐지만 그 대가로 Bacterial_spot이 14%→4%로 나빠져 macro F1은 오히려 소폭
하락했다. **Bangladesh는 pseudo-label을 미세조정하는 접근으로는 한계에 도달한 것으로 보인다.**

### 추가 시도: R-GV (CVPR 2025, 포스터 기반 근사 구현) — SHOT+APS를 못 넘음

Zhu et al., *Revisiting Source-Free Domain Adaptation: Insights into Representativeness,
Generalization, and Variety*, CVPR 2025. PlantDoc의 "표현력 부족" 문제를 겨냥해 시도했다.
**이 논문은 공식 코드가 없고 본문 PDF도 CVF/ResearchGate 둘 다 접근이 막혀 있어서, 공식 CVPR
포스터에 공개된 수준(① confidence+coverage 기반 progressive sample selection ② target 내부
semantic alignment ③ 여러 라운드 반복)만으로 재구성한 근사 구현이다 — 정확한 논문 수식과는
다를 수 있다.**

semantic alignment(confident/unconfident 그룹 간 배치 단위 feature 평균 거리 최소화)를 켜면
Taiwan·Bangladesh가 마스킹-only보다도 나빠졌다 — CACL과 똑같은 패턴(고정된 classifier와 움직이는
feature space의 구조적 불일치)이다. alignment를 끄고 progressive selection만 남긴 최선의 설정으로도:

| 데이터셋 | SHOT+APS(최종) | R-GV(alignment 제외, 근사) |
|---|---|---|
| Taiwan | **44.59%** | 36.94% |
| Bangladesh | **30.79%** | 30.61%(거의 동일) |
| PlantDoc | **19.46%** | 17.07%(더 나쁨) |

어디서도 SHOT+APS를 넘지 못했다. PlantDoc에서조차 개선이 없었다는 점은 오라클 실험 결과(정답을
줘도 안 변함)와 일관된다 — **표현력 자체가 부족한 상황에서는 pseudo-label을 다듬는 방향이든,
feature를 재정렬하는 방향이든, 우리가 시도한 방식으로는 돌파가 안 됐다.**

### 검토만 하고 구현은 보류: UCon-SFDA

Xu et al., *Revisiting Source-Free Domain Adaptation: A New Perspective via Uncertainty Control*
(UCon-SFDA), ICLR 2025, 공식 코드 [`xugezheng/UCon_SFDA`](https://github.com/xugezheng/UCon_SFDA).
논문 본문(4장)까지 확인했는데, distributionally-robust-optimization 기반 contrastive loss(식 6~7,
k-NN 양성쌍 + augmentation 기반 "가짜 음성쌍" dispersion control) 위에 partial-label 지도까지
더한 구조로, **CACL과 근본적으로 같은 계열(contrastive loss)**이다. CACL이 고정된 classifier와
구조적으로 안 맞아 실패했던 걸 감안하면 UCon-SFDA도 같은 문제를 겪을 가능성이 높고, 하이퍼파라미터도
훨씬 많아(λ_CL, λ_DC, λ_PL, τ, K_PL) 구현·재조정에 드는 시간 대비 성공 가능성이 낮다고 판단해
**구현하지 않고 이 판단 근거만 기록**으로 남긴다.

### 추가 시도: EATA (TENT의 최신 후속작) — SHOT을 넘지 못함

AdaBN·TENT·SHOT 다음으로 더 최근·유명한 방법도 시도해봤다. **EATA**(Niu et al., *Efficient
Test-Time Model Adaptation without Forgetting*, ICML 2022, 공식 코드
[`mr-eggplant/EATA`](https://github.com/mr-eggplant/EATA))는 TENT와 같은 지점(BN affine만 학습)에서
출발하되 두 가지를 더한다 — ① entropy가 임계값(E0 = 0.4·ln(클래스 수))보다 높은(불확실한) 샘플과
직전 배치 평균과 코사인 유사도가 높은(중복된) 샘플은 적응에서 제외하고, ② source 데이터(PlantVillage
test셋)로 추정한 Fisher information을 가중치 삼아 `(현재 파라미터 - 원본 파라미터)²` 항을 손실에
더해 source 지식을 과도하게 잊지 않게 막는다(`src/adapt/domain_adaptation.py`의 `apply_eata`).

**구현하면서 발견한 점**: 논문 기본 설정 그대로(`lr=1e-3`, TENT/SHOT과 동일)로는 거의 적응이 안
일어났다 — entropy·다양성 필터와 가중치 스킴이 ImageNet 1000클래스·대량 continual 스트림 기준으로
설계돼 있어서, 우리처럼 클래스가 3~8개뿐이고 target 샘플이 수백~천 장뿐인 설정에서는 유효 gradient가
너무 작아진다. `lr=0.1`(100배)로 올려야 실제로 움직이는 걸 실측으로 확인하고 그 값을 썼다.

| 데이터셋 | Source+마스킹 | SHOT+마스킹(최종) | EATA+마스킹 |
|---|---|---|---|
| Taiwan | 32.48% | **42.04%** | 35.35% |
| Bangladesh | 22.34% | **30.79%** | 20.89% |
| PlantDoc | **20.31%** | 19.46% | 12.27% |

**SHOT을 어디서도 못 넘었다.** Taiwan은 마스킹-only·TENT는 앞서지만 AdaBN·SHOT엔 못 미치고,
Bangladesh·PlantDoc은 마스킹-only보다도 낮다(PlantDoc은 전체 방법 중 최저치, 다만 8개 클래스에
고르게 퍼져 있어 collapse는 아니다). SHOT이 최종 권장 방법 자리를 유지한다 — "유명하고 TENT 직계
후속작이라도 설계 당시 가정한 스케일(대형 모델·수천 클래스급 continual TTA)이 다르면 소형 모델·
소수 클래스 문제엔 그대로 옮겨오기 어렵다"는 것도 하나의 결론으로 남긴다.

### 추가 검증: 모델을 더 줄이면 SHOT 과적합이 줄어들까 (tiny_cnn_c_lite)

교수님 피드백: "SHOT이 epoch 2만 넘어도 과적합되는 건 모델 용량이 커서 그런 것 아니냐, 채널을 더
줄여봐라(32→64→128→128 → 32→32→64→64)." 이 가설을 검증하려고 `tiny_cnn_c`와 같은 4블록
Depthwise Separable 구조에 채널만 절반 수준으로 줄인 **tiny_cnn_c_lite**(파라미터 10,218개,
tiny_cnn_c의 31,498개 대비 약 1/3)를 추가했다. 보간법도 피드백대로 bicubic 대신 기본값인
bilinear로 96px에서 학습했다.

| | tiny_cnn_c (31.5K) | tiny_cnn_c_lite (10.2K) |
|---|---|---|
| 내부 정확도 | 96.59% | 95.34% |
| Taiwan SHOT(epoch=2) | **42.04%** | 37.90% |
| Bangladesh SHOT(epoch=2) | **30.79%** | 21.07% |
| PlantDoc SHOT(epoch=2) | 19.46%(F1 0.182) | **22.57%(F1 0.202)** |

**결과는 가설과 절반만 맞았다.** epoch별 추이(`results/figures/shot_epoch_sensitivity_tiny_cnn_c_lite.png`)를
보면:
- **Bangladesh**는 가설대로 과적합이 완화됐다 — tiny_cnn_c는 epoch 2~3에서 꺾였는데, lite 모델은
  epoch 5까지도 계속 좋아진다(16.4%→21.1%→23.3%→26.3%).
- **PlantDoc은 반대로 더 빨리 과적합됐다** — lite 모델은 epoch 1에서 이미 정점(23.7%)을 찍고
  epoch 2부터 바로 내려간다(22.6%→21.0%→18.2%). 모델 용량이 작아지니 pseudo-label 노이즈에
  더 빨리 수렴해버리는 것으로 보인다.
- Taiwan·Bangladesh는 작은 모델이 전반적으로 더 나쁘고(내부 정확도도 1.25%p 낮음), PlantDoc만
  더 좋다 — "모델을 줄이면 전반적으로 낫다"는 결론은 내릴 수 없었다.

**해석**: 모델 용량 축소가 SHOT의 pseudo-label 과적합을 줄이는 효과는 데이터셋의 클래스 수·불균형
정도에 따라 방향이 달라, 일관된 해결책이 아니다. 과적합 억제는 모델 크기보다 `shot_epoch_sensitivity`
분석에서처럼 **epoch 수를 고정하는 방식**으로 다루는 게 더 안정적이라는 기존 결론을 재확인한 셈이다.

### 3-seed 검증 결과: APS의 개선은 noise 범위 안이었다

`tiny_cnn_c`(96px·bicubic)를 seed123/2026에서 추가 학습하고(seed42와 합쳐 3-seed), 위 DA 비교
(Source/AdaBN/SHOT/SHOT+APS)를 전부 재평가했다.

| 방법 | Taiwan | Bangladesh | PlantDoc |
|---|---|---|---|
| Source+마스킹 | 32.27% ±0.79 | 16.50% ±4.58 | 20.17% ±1.79 |
| AdaBN+마스킹 | 35.24% ±2.57 | 15.14% ±2.44 | 16.83% ±1.80 |
| SHOT+마스킹(epoch=2) | **37.58% ±5.86** | **24.28% ±5.62** | **19.75% ±1.40** |
| SHOT+APS+마스킹 | 36.41% ±6.29 | 24.07% ±4.88 | 18.95% ±0.69 |

(seed 42/123/2026 세 번 실행의 평균±표준편차, %p. 내부 테스트 정확도도 96.59/95.93/96.52%로
seed마다 흔들려 ±0.30%p 변동이 있다.)

seed42 단일 실행에서는 APS가 Taiwan을 42.04%→44.59%로 뚜렷하게 개선하는 것처럼 보였지만, **3-seed로
보면 SHOT+APS가 SHOT보다 모든 데이터셋에서 평균이 오히려 살짝 낮다**(Taiwan -1.17%p, Bangladesh
-0.21%p, PlantDoc -0.80%p) — 그 차이는 표준편차(5~6%p)보다 작아 noise 범위를 벗어나지 못한다.
**seed42가 APS에 우연히 유리했던 것일 뿐, APS가 통계적으로 유의미한 개선이라는 근거는 없다.**
그래서 **최종 권장 파이프라인을 "마스킹 항상 적용 + SHOT(epoch=2)"로 되돌린다** — APS는 "Taiwan
같이 클래스가 적은 데이터셋에서는 추가해볼 만한 옵션"으로 남기되, 기본값으로 켜지 않는다. SHOT
자체도 seed에 따라 표준편차가 5~6%p로 커서(특히 Taiwan·Bangladesh), 단일 seed 결과를 그대로
"최종 성능"으로 보고하면 과대 해석이 될 수 있다는 점도 이번 검증에서 같이 확인했다.

### 데이터셋별 하이퍼파라미터 극단 튜닝 시도 — 또 한 번의 seed 과적합

지도교수님이 "데이터셋마다 따로 튜닝해도 된다"고 확인해 주셔서, epoch=2 공통 고정값 제약을 풀고
Taiwan·Bangladesh 각각에 맞는 하이퍼파라미터를 seed42 기준으로 최대한 밀어붙여 봤다. **결론만
먼저 말하면 — 이 역시 APS와 같은 seed 과적합 함정에 걸렸고, Bangladesh는 오히려 기존보다
나빠졌다.** 이 절은 그 과정과 실패를 투명하게 남긴다.

**Taiwan**: epoch·cls_weight를 더 키울수록 seed42 정확도가 계속 올라, epoch=7·cls_weight=1.0에서
**63.06%**(F1 0.592)까지 나왔다(epoch=10은 56%대로 다시 떨어져 7이 사실상 정점). 60% 목표를
넘겼다고 생각했지만, 이 설정을 seed123/2026에 그대로 적용하자 33.12%/42.99%로 크게 떨어졌다 —
**3-seed 평균 46.39% ±12.46**. 평균 자체는 epoch=2 기준(37.58%)보다 높지만, 표준편차가 너무 커서
"60%를 reproducible하게 달성했다"고는 말할 수 없다.

**Bangladesh**: 다음을 새로 구현/시도했다 — ① `_balanced_assign`(클래스별 pseudo-label 할당
개수에 상한을 둬 한 클래스로의 쏠림을 억제, `--shot-balanced-cap`) ② 특정 클래스를 pseudo-labeling에서
완전히 제외(`--shot-exclude-classes`) ③ seed42/123/2026 세 source 모델의 softmax를 평균한
앙상블 pseudo-labeling. Oracle 진단을 클래스별로 다시 뜯어보니 원인이 명확해졌다 — **Target_Spot
(recall 21%)·Bacterial_spot(recall 7%)은 실제 정답을 줘도 모델이 잘 못 맞추는, 작고(14~28장)
시각적으로 겹치는 클래스였다.** 반면 Late_blight(96%)·Leaf_Mold(58%)·healthy(57%)·Early_blight(64%)는
오라클에서 잘 분리돼, 전체 샘플의 87%를 차지하는 이 4개만 잘 맞혀도 오라클 수준(58.6%)에 근접할
수 있다는 계산이 나왔다. 그런데 ①~③ 전부 명확한 돌파구가 되지 못했다 — exclude는 오히려
악화시켰고(Target_Spot 자리를 없애도 그 샘플들이 Late_blight로 안 가고 다른 틀린 클래스로 흩어짐),
앙상블도 중립적이었다. 결국 `class_balance + balanced_cap(1.0)` 조합에 epoch·cls_weight를
그리드서치해 **epoch=4·cls_weight=2.0에서 seed42 34.42%**(F1 0.302)를 찾았는데, 이걸 3-seed로
검증하자 seed123=10.54%, seed2026=13.81%로 **완전히 붕괴했다 — 3-seed 평균 19.59% ±10.57,
기존 SHOT(epoch=2) 기준(24.28%)보다도 더 나쁘다.**

**교훈**: "데이터셋별로 튜닝해도 된다"는 허락은 공정성(여러 데이터셋에 같은 epoch을 억지로
강제하지 않아도 된다) 문제를 풀어준 것이지, "1개 seed만 보고 하이퍼파라미터를 끝까지 밀어도
된다"는 뜻이 아니다. cls_weight를 2.0까지 올리는 것처럼 하이퍼파라미터를 공격적으로 키울수록
seed42의 특정 pseudo-label 초기 상태에 과적합할 위험도 같이 커지고, 이번처럼 Bangladesh에서는
그 과적합이 다른 seed에서 "개선"이 아니라 "붕괴"로 나타났다. **모든 하이퍼파라미터 탐색은
반드시 3-seed로 재검증한 뒤에만 "성능 향상"이라고 보고해야 한다**는 원칙을 이번에도 재확인했다
— 그래서 두 설정 다 최종 파이프라인에 반영하지 않고, `마스킹+SHOT(epoch=2)`(3-seed 검증된
Taiwan 37.58%±5.86·Bangladesh 24.28%±5.62·PlantDoc 19.75%±1.40)를 최종으로 유지한다.
`_balanced_assign`/`--shot-exclude-classes`는 코드에는 남겨 두되(재현·추가 실험용), 기본
파이프라인에는 포함하지 않는다.

**세 데이터셋 모두 60% 이상이라는 목표는, 이 프로젝트 규모(경량 CNN, target 300~1,100장,
label-free)에서는 현실적이지 않다는 게 최종 결론이다** — Taiwan은 오라클 상한(70.5%)이 높아
seed-robust한 방법을 더 찾으면 가능성이 있고, Bangladesh는 오라클 상한(58.6%)과의 격차가 여전히
크며 그중 상당 부분이 Target_Spot/Bacterial_spot의 구조적 난이도(정답을 줘도 어려움)에서 온다,
PlantDoc은 오라클 상한 자체가 23.5%라 애초에 불가능하다.

### 추가 시도: AaD (NeurIPS 2022) — pseudo-label 없이도 SHOT을 못 넘음

Yang et al., *Attracting and Dispersing: A Simple Approach for Source-Free Domain Adaptation*,
NeurIPS 2022, 공식 코드 [`Albert0147/AaD_SFDA`](https://github.com/Albert0147/AaD_SFDA). 지금까지
시도한 SHOT/EATA/R-GV는 전부 "이 샘플이 어떤 클래스냐"를 먼저 정하는 클러스터링/pseudo-label
단계가 있고, Bangladesh에서 반복 관찰된 "한 클래스로 쏠림"(Target_Spot 등)이 그 단계에서 생겼다.
AaD는 클러스터링이 전혀 없다 — feature 공간에서 k-최근접 이웃끼리는 softmax 예측이 가깝도록
당기고(attract), 나머지 전체 샘플과는 멀어지도록(disperse) 만드는 output-space 제약만으로
적응한다. 공식 구현은 momentum memory bank를 쓰지만, 데이터셋이 작아(300~1,100장) 매 epoch
시작 시 전체를 다시 계산하는 것으로 단순화했다(EATA/R-GV와 같은 "근사 구현" 패턴).

| 데이터셋 | SHOT(epoch=2, 최종) | AaD(최선 하이퍼파라미터) |
|---|---|---|
| Taiwan | 42.04%(F1 0.389) | 42.36%(F1 0.422, disperse_weight=3.0) |
| Bangladesh | **30.79%**(F1 0.261) | 20.16%(F1 0.194, disperse_weight=2.0) |
| PlantDoc | **19.46%**(F1 0.182) | 19.18%(F1 0.141, 기본값) |

클러스터링이 없어서 기대했던 "쏠림 방지" 효과는 확인되지 않았다 — Bangladesh는 오히려 SHOT보다
10%p 이상 낮다(disperse 항이 세질수록 소수 클래스 쪽 신호가 더 묻히는 것으로 보인다). Taiwan은
seed42 기준 SHOT과 거의 동치(소수점 차이)지만, 이 프로젝트에서 이미 두 번(APS, 데이터셋별 극단
튜닝) 확인했듯 **단일 seed에서의 "거의 동치"는 3-seed로 보면 어느 쪽으로든 뒤집힐 수 있어서**,
굳이 3-seed 검증에 시간을 더 쓰기보다 여기서 멈춘다. AdaBN/TENT/EATA/UPA/R-GV/AaD까지 — 최신·
유명 방법을 포함해 시도한 모든 대안이 결국 SHOT(epoch=2)을 넘지 못했고, 그 각각의 실패 이유가
전부 다르다는 점(negative transfer, contrastive-loss 구조 불일치, alignment 손실의 역효과,
disperse 항의 소수 클래스 희석)이 오히려 SHOT이 이 프로젝트 규모에 가장 안정적으로 맞는 방법
이라는 결론을 거듭 뒷받침한다.

### 연구 방향 재정리와 가설 검증

지금까지의 진단(심층 진단 절)과 추가 시도(UPA/R-GV/EATA/UCon-SFDA 절) 결과를 바탕으로, 이 프로젝트의
Domain Adaptation 부분은 "여러 방법을 똑같이 돌려서 비교"하는 단계를 넘어 **데이터셋마다 다른
실패 원인에 맞춰 다른 개선 전략을 검증**하는 단계로 재정의한다.

> **연구 주제**: 경량 CNN 기반 토마토 병해 분류에서 Pseudo-label 신뢰성과 특징 표현 개선을 통한
> Source-Free Domain Adaptation 성능 향상

- **가설 1**: 신뢰도 기반 pseudo-label 선택(UPA의 APS)은 클래스별 성능 손실을 줄이면서 적응
  성능을 개선할 수 있다.
  → **seed42 단일 실행에서는 확인되는 것처럼 보였으나(42.04%→44.59%), 3-seed 검증에서 기각**
  — 위 "3-seed 검증 결과" 절에서 보듯 SHOT+APS의 3-seed 평균은 SHOT 단독보다 모든 데이터셋에서
  오히려 약간 낮고, 그 차이는 seed 간 표준편차보다 작다. Bangladesh는 애초에 seed42에서도
  기각이었다 — APS·class-balance·둘의 조합 전부 기존 SHOT(30.79%)을 못 넘었고, Late_blight↔
  Target_Spot "교환" 패턴도 줄이지 못했다. **단일 seed에서 좋아 보였던 개선이 다중 seed 검증을
  통과하지 못한 사례**로, 이 프로젝트 규모(300~1,100장)에서는 seed 1개만으로 하이퍼파라미터나
  구성요소를 확정하면 안 된다는 교훈을 남겼다.
- **가설 2**: 초기 feature 분포가 충분히 구분되지 않는 PlantDoc에서는 pseudo-label 정제보다 feature
  표현·정렬 개선이 더 중요하다.
  → **진단(오라클 실험)은 확인됨**: 정답을 줘도 PlantDoc은 전혀 개선되지 않아, 병목이 pseudo-label이
  아니라 feature 표현력이라는 진단 자체는 맞았다. **그런데 그 처방(R-GV의 semantic alignment, CACL의
  contrastive loss)은 둘 다 실패**했다 — 원인이 진단대로 "표현력 부족"이라 해도, 우리가 시도한
  "feature를 직접 재정렬/contrastive로 당기는" 방식은 고정된 source classifier와 구조적으로
  맞지 않았다. 진단은 맞고 처방은 틀렸다는, 구분해서 봐야 하는 결과다.

이 프로젝트 규모(경량 CNN, 수백~천 장 target, 파라미터 1~3만 개)에서 반복적으로 확인된 메타
패턴: **AdaBN 완전 교체, EATA의 entropy 필터, APS의 필터링, SHOT의 적응 epoch까지 — 대규모
벤치마크(ImageNet급 continual TTA, Office-Home 등)를 기준으로 설계된 하이퍼파라미터/구성요소는
거의 전부 재조정이 필요했고, 일부(완전 교체형 통계 변경, contrastive loss 기반 feature 정렬)는
재조정으로도 못 고치고 아예 빼야 했다.** 모델과 데이터가 작을수록 "최신 방법일수록 좋다"보다
"그 방법이 가정한 스케일이 내 상황과 맞는가"를 먼저 따져야 한다는 것이, 이번 Domain Adaptation
전체 과정에서 얻은 가장 일반적인 교훈이다.

### 한계점: 왜 PlantDoc은 더 못 올리는가 — 그리고 왜 더 시도하지 않았는가

오라클 상한 실험으로 PlantDoc의 병목이 pseudo-label이 아니라 **feature extractor의 표현력
자체**라는 것까지는 확인했다(정답을 줘도 23.47%가 한계). 이 병목을 근본적으로 풀 수 있는
방법은 이론적으로 하나뿐이다 — **source(PlantVillage) 학습 자체를 바꿔서, 처음부터 실제 촬영
환경(조명·배경·구도 변화)에 더 강건한 feature를 학습시키는 것**(예: 밝기·색상·블러 증강 추가,
Corruption 강건성 절에서 이미 "밝기 변화에 가장 취약하다"고 확인된 약점과 직접 연결된다).

**이 프로젝트에서는 학습 데이터·학습 파이프라인을 바꾸는 것이 범위 밖이라, 시도하지 않고
한계점으로 남긴다.** 지금까지 시도한 모든 방법(AdaBN/TENT/SHOT/EATA/UPA/R-GV/AaD + 하이퍼파라미터
탐색)은 전부 "이미 고정된 feature extractor 위에서 추론/적응 단계만 바꾸는" 접근이었고, 이
범위 안에서는 SHOT(epoch=2)이 가장 안정적인 방법이라는 결론에 3-seed 검증까지 거쳐 도달했다.
**PlantDoc의 23%대, Bangladesh의 오라클-SFDA 격차(58.6% vs 24~31%)는 "적응 기법을 더 찾아서"
풀 수 있는 문제가 아니라, 근본적으로는 source 학습 데이터·증강 전략의 한계**라는 게 이번
프로젝트의 최종 진단이다. 후속 연구에서 다룰 만한 방향으로 명시해 둔다.

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
| 8 | Domain Adaptation (AdaBN/TENT/SHOT/EATA/UPA/R-GV/AaD) | `src/adapt/domain_adaptation.py`, `shot_stage_analysis.py`, `shot_oracle_upperbound.py` | `results/external/*/*_{adabn,tent,shot,eata,rgv,aad}*masked*/` |

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
│   ├── adapt/            # domain_adaptation.py(AdaBN/TENT/SHOT/EATA/UPA/R-GV/AaD + 마스킹) + shot_stage_analysis.py/shot_oracle_upperbound.py(진단 전용)
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

# Domain Adaptation (AdaBN/TENT/SHOT/EATA/R-GV) — 최종 모델에 적용, target label 미사용
python src/adapt/domain_adaptation.py --model tiny_cnn_c --img-size 96 --interp bicubic --dataset all --method all --masked
# 최종 권장 구성: SHOT(epoch=2) — 마스킹이 방법 자체에 내장돼 있어 --masked 없어도 항상 적용됨
# (APS는 3-seed 검증에서 noise 범위로 기각돼 기본값에서 제외; 옵션으로 쓰려면 --shot-aps --shot-aps-gamma 1.0 추가)
python src/adapt/domain_adaptation.py --model tiny_cnn_c --img-size 96 --interp bicubic --dataset all --method shot

# SHOT 심층 진단 — pseudo-label 품질 추적 + 클래스별 precision/recall (target label은 진단에만 사용)
python src/adapt/shot_stage_analysis.py --dataset all --epochs 5
# SHOT oracle 상한 실험 — pseudo-label 대신 실제 정답으로 학습(진단 전용, 정상 SFDA 결과 아님)
python src/adapt/shot_oracle_upperbound.py --dataset all

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
