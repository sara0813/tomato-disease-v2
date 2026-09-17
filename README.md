# 경량 CNN 기반 토마토 잎 병해 분류 및 강건성 평가

> 모델은 더 작게 · 분류 문제는 더 어렵게 · 성능은 안정적으로

실제 촬영 환경의 어려운 이미지에서도 쓸 수 있는 **작고 빠른** 토마토 병해 분류 모델을 직접 설계하고,
대형 모델과 비교해 **정확도 · 효율성 · 강건성 · 일반화**의 균형을 검증하는 프로젝트. PyTorch 기반.

## 진행 상태 (2026-09-17 기준)

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | 경량 모델 설계 + 학습 (Tiny CNN A/B/C) | ✅ 완료 |
| 2 | 내부 성능 평가 | ✅ 완료 |
| 3 | 효율성 평가 | ✅ 완료 |
| 4 | Corruption 강건성 평가 | ✅ 완료 |
| 5 | 외부 데이터 평가 | ✅ 완료 |
| 6 | 최종 모델 선정 | ⏸️ 보류 (정확도·크기·강건성·일반화 중 우선순위 판단 필요) |
| 7 | 웹 시스템 적용 | ✅ 프로토타입 완료 (`app/streamlit_app.py`, 모델 선택형) |

진행 과정과 판단 근거를 정리한 슬라이드: `docs/CNN_토마토_병해분류_V2_진행보고.pptx`

## 연구 질문

| | 질문 |
|---|---|
| RQ1 | 대형 사전학습 모델보다 훨씬 작은 CNN으로 유사한 내부 분류 성능을 얻을 수 있는가? |
| RQ2 | 작은 모델이 조명 변화, 블러, 노이즈 및 부분 가림에도 안정적으로 분류할 수 있는가? |
| RQ3 | 모델 크기와 정확도 사이에서 가장 효율적인 구조는 무엇인가? |
| RQ4 | PlantVillage로 학습한 경량 모델이 Taiwan · Bangladesh 데이터에도 일반화되는가? |

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

| 모델 | 파라미터 | 내부 test Accuracy | Macro F1 |
|---|---|---|---|
| tiny_cnn_a (최소형) | 94,762 | 94.99% | 0.9396 |
| **tiny_cnn_b (중간형)** | 242,474 | **97.66%** | 0.9718 |
| tiny_cnn_c (Depthwise) | **31,498** | 97.29% | 0.9684 |

V1 최고 기록(EfficientNetB0 86.21%)보다 세 모델 모두 높다. `tiny_cnn_c`는 `tiny_cnn_b`의 13% 크기로
정확도 차이 0.4%p — Depthwise Separable Conv의 설계 의도가 실측으로 확인됨(RQ1, RQ3).

### 효율성 (`results/efficiency/`)

| 모델 | FLOPs | 파일 크기 | CPU 추론(batch=1) |
|---|---|---|---|
| tiny_cnn_a | 330.3M | 0.37MB | 4.98ms |
| tiny_cnn_b | 405.8M | 0.94MB | 4.91ms |
| **tiny_cnn_c** | **74.4M** | **0.13MB** | 4.99ms |

### Corruption 강건성 (`results/corruption/`) — 조건별 평균 성능 저하율

| 조건 | 평균 저하율 | 비고 |
|---|---|---|
| reflection | 5.90% | 가장 잘 버팀 |
| occlusion | 12.27% | |
| shadow | 16.84% | |
| noise | 37.36% | |
| blur | 47.90% | |
| brightness | 64.64% | 가장 취약 (학습에 밝기 증강 안 씀) |

모델별 평균 저하율: tiny_cnn_b 27.83% (최선) · tiny_cnn_c 30.52% · tiny_cnn_a 34.11%.

### 외부 데이터 일반화 (`results/external/`) — RQ4

| 모델 | 내부 | Taiwan (3/10 클래스) | Bangladesh (6/10 클래스) | PlantDoc (8/10 클래스) |
|---|---|---|---|---|
| tiny_cnn_a | 94.99% | 20.38% | 15.26% | 20.03% |
| tiny_cnn_b | 97.66% | 28.03% | 17.89% | 20.87% |
| tiny_cnn_c | 97.29% | 27.39% | 15.17% | 20.87% |

**경량화해도 domain shift 자체는 해결되지 않는다.** V1의 대형 모델과 비슷한 범위로 급락 — 모델
크기의 문제가 아니라 실제 환경 일반화 자체가 남은 과제라는 프로젝트의 핵심 문제의식이 재확인됨.
특히 PlantDoc은 클래스 커버리지가 Taiwan(3개)보다 훨씬 넓은데도(8개) 정확도가 비슷한 ~20%대에
머문다 — 클래스가 안 겹쳐서 낮은 게 아니라 **진짜 도메인 시프트(실제 촬영 조건) 문제**라는 뜻.

## 데이터셋

| 데이터 | 역할 | 규모 |
|---|---|---|
| PlantVillage Tomato | 학습 · 내부 평가 | 18,160장 / 10개 클래스 (70/15/15 층화분할: train 12,707 / val 2,719 / test 2,734) |
| Taiwan Tomato | 외부 환경 평가 (학습 안 함) | 314장, 겹치는 3개 클래스만 사용 |
| Bangladesh Tomato Leaf | 외부 환경 + BBox 평가 (학습 안 함) | 1,101장, YOLO bbox + 40% 여백 + 최소 96px 크롭 |
| PlantDoc | 외부 환경 평가 (학습 안 함) | 709장, 8개 클래스 대응 (Google/Bing 이미지 검색 기반, 원본 746장 중 진단표·비교 콜라주 등 35장 제외) |
| Corruption 변형셋 | 4단계 강건성 평가 전용 | test셋 × 6조건 × 3단계 = 49,212장 |

품질 검증: 이미지 무결성(20,284장 전수, 콘텐츠 문제 0건) · 클래스 불균형 기록(보정 없이 자연 분포 사용,
14.36배) · train/test leakage 검사(perceptual hash, 0.62~0.88% — 재분할 불필요로 판단) · PlantDoc은
콘택트시트 수작업 검수로 잎 사진이 아닌 이미지(진단표/비교 콜라주/삽화) 35장을 걸러냄.

## 실험 로드맵

| 단계 | 내용 | 코드 | 산출물 |
|---|---|---|---|
| 1 | 경량 모델 설계 (Tiny CNN 3종) | `src/models/tiny_cnn.py`, `src/train/train_model.py` | `models/*.pt`, `results/internal/*/training_log.*` |
| 2 | 기본 성능 평가 | `src/evaluate/evaluate_internal.py` | `results/internal/` |
| 3 | 효율성 평가 | `src/evaluate/measure_efficiency.py` | `results/efficiency/` |
| 4 | Corruption 평가 | `src/evaluate/evaluate_corruption.py` | `results/corruption/` |
| 5 | 외부 데이터 평가 | `src/evaluate/evaluate_external.py` | `results/external/` |
| 6 | 최종 모델 선정 (보류) | `src/summary/make_summary.py` | `results/summary/model_comparison.*` (비교표만, 판단 없음) |
| 7 | 시스템 적용 | `app/streamlit_app.py` | 웹 프로토타입 (모델 선택형) |

**Corruption은 학습 증강이 아니라 평가용 변형이다.** 밝기 · 그림자 · 반사 · 블러 · 노이즈 · 가림
6개 조건 × 약/중/강 3단계로 **테스트 이미지만** 변형해 모델별 성능 저하율을 비교한다. 학습 데이터에는
전혀 섞이지 않는다(`data/processed/plantvillage/train/` vs `data/corrupted/`, 완전히 분리된 경로).

최종 모델은 최고 정확도 하나로 고르지 않는다.
① 내부 성능 ② 외부 일반화 ③ corruption 저하율 ④ 파라미터 수 ⑤ 모델 크기 ⑥ CPU 추론시간을 종합한다.

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
│   ├── train/            # train_model.py — 모델 이름 인자 하나로 통일, early stopping, 즉시 체크포인트 저장
│   ├── evaluate/         # 내부 · 효율 · corruption · 외부 평가
│   ├── summary/          # 종합 비교표 (V2 실측 + V1 인용, 판단 없음)
│   ├── visualize/        # 결과 그래프 (results/figures/*.png)
│   └── utils/            # 시드 · 입출력
├── models/               # 학습된 .pt 가중치 (git 제외)
├── results/              # _common · internal · efficiency · corruption · external · summary · figures
├── app/                  # Streamlit 웹 프로토타입 (모델 선택형, 정상/비정상 색상 표시)
├── notebooks/
└── docs/                 # 프로젝트 브리프 PDF + V2 진행 보고 PPTX
```

## 실행 순서

```bash
# 가상환경 (Python 3.11)
py -3.11 -m venv .venv
.venv/Scripts/pip install -r requirements.txt

# 데이터 준비 (원본은 건드리지 않고 새 폴더에 생성)
python src/data_prep/split_plantvillage.py
python src/data_prep/prepare_taiwan.py
python src/data_prep/prepare_bangladesh_bbox.py
python src/data_prep/prepare_plantdoc.py
python src/data_prep/verify_image_integrity.py
python src/data_prep/check_dataset.py
python src/data_prep/check_leakage.py
python src/corruption/make_corrupted_sets.py

# 학습 — V2에서 실제로 학습하는 건 이 3개뿐 (config.TINY_MODELS)
python src/train/train_model.py --model tiny_cnn_a
python src/train/train_model.py --model tiny_cnn_b
python src/train/train_model.py --model tiny_cnn_c

# 평가
python src/evaluate/evaluate_internal.py
python src/evaluate/measure_efficiency.py
python src/evaluate/evaluate_corruption.py
python src/evaluate/evaluate_external.py

# 종합 + 시각화
python src/summary/make_summary.py
python src/visualize/plot_results.py
python src/visualize/plot_class_distribution.py

# 웹 프로토타입
streamlit run app/streamlit_app.py
```

Windows에서 한글 출력이 깨지면 `PYTHONUTF8=1`을 앞에 붙여 실행한다.

## 실행 환경

로컬 CPU 또는 일반 Colab. 고성능 GPU를 전제하지 않는 크기를 목표로 한다 — Tiny CNN 3종은 CPU
기준 모델당 학습 1.5~2시간 내외(early stopping 포함), 추론은 이미지당 5ms 미만이다.
