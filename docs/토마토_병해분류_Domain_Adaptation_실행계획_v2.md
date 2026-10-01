# CNN 기반 토마토 병해 분류 — Domain Adaptation 실행 계획 (v2)

작성 기준: 2026-10-01
전제: 이 문서는 기존 `토마토_병해분류_Domain_Adaptation_실험설계_및_레퍼런스.pdf`의 후속 실행판이다.
교수님 확인을 아직 못 받은 상태에서, 지금 가진 정보만으로 실행 가능한 계획을 세운다.
교수님 피드백을 받으면 "3. 진행 가정" 섹션을 업데이트하고 필요한 부분만 재실행한다.

## 0. 교수님 로드맵 (3단계) 인용

> ③ Domain Adaptation을 통한 일반화 성능 개선
> - 선정된 모델·해상도·Resize 방법을 기준 모델로 고정
> - 다양한 Domain Adaptation 방법론 및 오픈소스 조사·적용
> - 외부 데이터셋 성능 및 추가 리소스 비교 → 효율적인 방법 선정
> - 선정 방법을 튜닝하여 External Dataset 성능 향상

이 문서의 구성은 이 4단계를 그대로 따른다.

## 1. 기준 모델 고정 (1단계 — 완료)

| 항목 | 값 |
|---|---|
| 모델 | tiny_cnn_c |
| 해상도 | **96px** |
| Resize 방법 | bicubic |
| 내부 정확도 | 96.59% |
| 외부 평균 정확도 | 23.02% (13개 해상도×보간법 조합 중 2위, 1위인 128px bicubic 23.31%와 0.3%p 이내) |
| 학습 시간 | 76분 (128px bicubic 96분 대비 21% 절약) |
| 파라미터 / FLOPs | 31.5K / 74M (tiny_cnn_b 대비 1/8, 1/5.5) |

선정 근거: **경량화가 프로젝트 목표**이므로, 내부 정확도·외부 일반화가 128px bicubic(1위)과 거의 동급(0.1~0.3%p 차이)이면서 해상도가 낮아 학습·추론 연산량이 더 적은 96px을 최종으로 선택했다. 128px box(torchvision/PIL에 OpenCV area 보간이 없어 같은 개념으로 대응시킨 필터)가 내부 정확도만 보면 더 높았지만(96.96%) 외부 일반화는 오히려 중간 수준이었고 학습 시간도 제일 길어 제외했다. (상세 비교는 `results/figures/resolution_interp_matrix_tiny_cnn_c.png`, `external_interp_matrix_tiny_cnn_c.png`, `training_time_matrix_tiny_cnn_c.png` 참고).

이 모델의 외부 데이터셋 source-only 성능(Accuracy, Macro F1, class-wise F1/Recall, confusion matrix)은 이미 `results/external/*/tiny_cnn_c_res96_bicubic/seed42/`에 저장되어 있다 — **다시 측정할 필요 없음**.

## 2. 진행 가정 (교수님 확인 전 임시 결정)

교수님께 아직 확인 못 받은 핵심 질문: "외부 데이터셋을 train시키지 말라"는 지시가 (a) target label을 이용한 지도학습만 금지하는 것인지, (b) target 이미지로 파라미터를 업데이트하는 것 자체(TENT 등)까지 금지하는 것인지.

**지금 세우는 가정** (교수님 답변 오면 즉시 수정):
- Label은 어떤 단계에서도 target 쪽 학습에 사용하지 않는다 (이 부분은 두 해석 모두에서 확정 금지).
- Target 이미지 자체를 이용한 파라미터 업데이트(TENT의 BN affine 조정 등)는 교수님 로드맵의 "다양한 DA 방법론 및 오픈소스 조사·적용"이라는 문구에 근거해 **일단 시도**한다.
- 단, **AdaBN 결과만으로도 독립적인 결론을 낼 수 있도록 실험을 설계**한다 — 나중에 (b)로 해석이 확정되어 TENT/SHOT을 못 쓰게 되어도, AdaBN 실험과 그 분석은 그대로 보고서에 남는 구조로 간다.
- SHOT은 이번 1차 실행 범위에서 제외한다 (3번 항목 참고).

## 3. 데이터셋 우선순위 — 1개만 먼저

3개 외부 데이터셋 전체를 한 번에 돌리지 않고, **Bangladesh 하나를 먼저 끝까지 진행**한 뒤 같은 코드로 나머지에 확장 적용한다.

| 데이터셋 | 전체 장수 | PlantVillage 10개 중 존재 클래스 | 클래스별 장수 범위 | source-only 정확도 |
|---|---|---|---|---|
| Taiwan | 314 | 3개 | 98~110 (균형적) | 27.39% |
| **Bangladesh** | **1101** | 6개 | 47~395 (불균형 큼) | 21.80% |
| PlantDoc | 709 | 8개 | 51~146 (균형적) | 19.89% |

**Bangladesh를 1순위로 선택한 이유**:
1. 전체 샘플 수가 가장 많아(1101장) AdaBN의 BN 통계 재계산, TENT의 배치 단위 적응이 가장 안정적으로 동작할 가능성이 높다.
2. source-only 정확도가 낮은 편(21.80%)이라 DA 적용 효과를 보여줄 여지가 크다.
3. 클래스 6개는 Taiwan(3개, 너무 단순)과 PlantDoc(8개, closed-set에 가까움)의 중간 난이도라 partial-set 문제를 보여주기에 적당하다.

**주의**: Bangladesh는 클래스별 불균형이 크다(Target_Spot 47장 vs healthy 395장, 약 8.4배 차이). Accuracy만 보면 안 되고 class-wise F1/Recall을 반드시 같이 봐야 한다 — 원본 PDF의 지적과 동일.

Taiwan/PlantDoc은 Bangladesh용으로 만든 코드를 그대로 재사용해 시간 남을 때 확장한다 (섹션 6 참고).

## 4. DA 방법론 조사·적용 (2단계)

| 방법 | 핵심 아이디어 | Target label | Target 이미지로 파라미터 업데이트 | 이번 1차 범위 |
|---|---|---|---|---|
| Source-only | 적용 없음 (기준) | 사용 안 함 | 안 함 | ✅ 완료 |
| AdaBN | BN running mean/var를 target 통계로 교체 | 사용 안 함 | 안 함 (통계만 교체) | ✅ 진행 |
| TENT | Entropy 최소화로 BN affine(γ,β)만 업데이트 | 사용 안 함 | 함 | ✅ 진행 (가정 하에) |
| SHOT | Feature extractor 전체 + pseudo-labeling | 사용 안 함 | 함 (훨씬 큰 폭) | ⏸ 보류 (스트레치 목표) |

SHOT을 이번에 보류하는 이유: 구현 난이도가 AdaBN/TENT보다 훨씬 높고(pseudo-labeling + information maximization loss 직접 구현 필요), Bangladesh의 최소 클래스가 47장뿐이라 클러스터링 기반 pseudo-label이 불안정해질 위험이 있다. AdaBN/TENT 결과를 먼저 보고, 개선 폭이 충분치 않을 때만 스트레치로 시도한다.

### 4-1. AdaBN 구현 방식
1. `tiny_cnn_c_res96_bicubic`의 학습된 가중치를 불러온다.
2. `model.train()` 모드로 전환하되 **역전파는 하지 않는다** (`torch.no_grad()` 유지) — forward pass만 여러 번 통과시켜 4개 BN 레이어의 `running_mean`/`running_var`만 Bangladesh 데이터 쪽으로 갱신한다.
3. 갱신이 끝나면 `model.eval()`로 전환해 라벨이 있는 동일 Bangladesh 데이터로 최종 평가(Accuracy/Macro F1/class-wise/confusion matrix)한다.
4. Reference: Li et al., *Revisiting Batch Normalization for Practical Domain Adaptation* (2016).

### 4-2. TENT 구현 방식
1. 같은 소스 모델에서 **새로 시작** (AdaBN 결과에 이어서 하지 않음 — 원본 PDF의 "비교 원칙"을 그대로 따름).
2. BN 레이어의 `weight`(γ), `bias`(β)만 `requires_grad=True`로 열고 나머지 파라미터는 전부 고정.
3. Bangladesh 배치에 대해 예측 entropy를 손실로 두고 몇 step(논문 기본값 1 epoch 상당) 최적화.
4. 평가는 AdaBN과 동일한 지표로.
5. Reference: Wang et al., *Tent: Fully Test-Time Adaptation by Entropy Minimization*, ICLR 2021. 공식 코드: `DequanWang/tent` (GitHub) — 구조 참고용으로 확인.

## 5. 비교 및 방법 선정 (3단계)

| Method | Accuracy | Macro F1 | Class-wise F1 (최저/최고) | 추가 연산 시간 | 비고 |
|---|---|---|---|---|---|
| Source-only | 21.80% | (측정 완료) | | 기준 | 이미 완료 |
| AdaBN | | | | 매우 낮음 (forward만) | |
| TENT | | | | 낮음 (BN param만 학습) | |

이 표를 다 채운 뒤, **정확도 개선 폭 대비 추가 비용**이 가장 효율적인 방법을 4단계(튜닝) 대상으로 선정한다. AdaBN이 이미 충분한 개선을 보이면 TENT까지 안 가도 스토리가 완성된다 — 원본 PDF의 취지("실패도 다음 근거가 된다")를 그대로 유지.

## 6. 선정 방법 튜닝 및 확장 (4단계)

1. 3단계에서 고른 방법의 hyperparameter를 조정한다 (TENT라면 step 수/learning rate, AdaBN이라면 momentum 등).
2. Bangladesh에서 검증된 코드를 **Taiwan, PlantDoc에 그대로 재실행**한다 (같은 함수에 dataset만 바꿔서 호출 — 별도 구현 불필요하게 설계).
3. 3개 데이터셋 전체에 대한 최종 비교표와 그래프를 만든다 (해상도×보간법 히트맵과 같은 스타일로, "DA 방법 × 데이터셋" 히트맵 추천).

## 7. 리스크 / 한계 노트

- **교수님 확인 미해결**: 섹션 2의 가정이 틀리면 TENT(및 향후 SHOT) 결과를 보고서에서 제외해야 할 수 있음. AdaBN은 어느 해석에서도 안전.
- **Bangladesh 클래스 불균형**: Target_Spot(47장)처럼 샘플이 적은 클래스는 DA 이후에도 신뢰도 낮은 지표가 나올 수 있음 — class-wise 지표를 반드시 같이 보고.
- **Taiwan은 3클래스뿐**이라 partial-set 문제를 다루기엔 좋지만, 방법 비교(3단계)의 "일반적 효과" 결론을 내리기엔 표본이 좁음 — 1차 결론은 Bangladesh 기준으로, Taiwan은 참고로만 사용.

## 8. 참고문헌 (기존 PDF에서 유지 + 확인)

- Li et al., *Revisiting Batch Normalization for Practical Domain Adaptation* (arXiv, 2016) — AdaBN
- Wang et al., *Tent: Fully Test-Time Adaptation by Entropy Minimization*, ICLR 2021 — TENT (공식 코드: DequanWang/tent)
- Liang, Hu, Feng, *Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation*, ICML 2020 — SHOT (보류)
- Cao et al., *Partial Adversarial Domain Adaptation*, ECCV 2018 — PADA (이론적 근거용)
- Cao et al., *Learning to Transfer Examples for Partial Domain Adaptation*, CVPR 2019 — ETN (이론적 근거용)
- Wu et al., *From Laboratory to Field: Unsupervised Domain Adaptation for Plant Disease Recognition in the Wild*, Plant Phenomics, 2023 — 실제 존재 확인함(웹 검색), MSUN 제안, **PlantDoc에서 56.06% 달성**. 우리 목표(약 60%)가 최신 전용 DA 기법과 비슷한 수준의 현실적 목표임을 보여주는 근거로 인용 가능.
  - https://spj.science.org/doi/10.34133/plantphenomics.0038
