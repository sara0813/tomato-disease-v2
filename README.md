# 경량 CNN 기반 토마토 잎 병해 분류 및 강건성 평가

> 모델은 더 작게 · 분류 문제는 더 어렵게 · 성능은 안정적으로

실제 촬영 환경의 어려운 이미지에서도 쓸 수 있는 **작고 빠른** 토마토 병해 분류 모델을 직접 설계하고,
대형 모델과 비교해 **정확도 · 효율성 · 강건성**의 균형을 검증하는 프로젝트.

## 연구 질문

| | 질문 |
|---|---|
| RQ1 | 대형 사전학습 모델보다 훨씬 작은 CNN으로 유사한 내부 분류 성능을 얻을 수 있는가? |
| RQ2 | 작은 모델이 조명 변화, 블러, 노이즈 및 부분 가림에도 안정적으로 분류할 수 있는가? |
| RQ3 | 모델 크기와 정확도 사이에서 가장 효율적인 구조는 무엇인가? |
| RQ4 | PlantVillage로 학습한 경량 모델이 Taiwan · Bangladesh 데이터에도 일반화되는가? |

## 1차 실험에서 확인한 문제

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

## 데이터셋

| 데이터 | 역할 | 규모 |
|---|---|---|
| PlantVillage Tomato | 학습 · 내부 평가 | 18,160장 / 10개 클래스 (train 12,707 / val 2,719 / test 2,734) |
| Taiwan Tomato | 외부 환경 평가 | 겹치는 3개 클래스만 사용 |
| Bangladesh Tomato Leaf | 외부 환경 + BBox 평가 | YOLO bbox 크롭 후 사용 |

## 실험 로드맵

| 단계 | 내용 | 코드 | 산출물 |
|---|---|---|---|
| 1 | 경량 모델 설계 (Tiny CNN 3종) | `src/models/tiny_cnn.py` | 구조도 · 파라미터 수 |
| 2 | 기본 성능 평가 | `src/evaluate/evaluate_internal.py` | `results/internal/` |
| 3 | 효율성 평가 | `src/evaluate/measure_efficiency.py` | `results/efficiency/` |
| 4 | Corruption 평가 | `src/evaluate/evaluate_corruption.py` | `results/corruption/` |
| 5 | 외부 데이터 평가 | `src/evaluate/evaluate_external.py` | `results/external/` |
| 6 | 최종 모델 선정 | `src/summary/make_summary.py` | `results/summary/` |
| 7 | 시스템 적용 | `app/streamlit_app.py` | 웹 프로토타입 |

**Corruption은 학습 증강이 아니라 평가용 변형이다.** 밝기 · 그림자 · 반사 · 블러 · 노이즈 · 가림
6개 조건 × 약/중/강 3단계로 테스트 이미지만 변형해 모델별 성능 저하율을 비교한다.

최종 모델은 최고 정확도 하나로 고르지 않는다.
① 내부 성능 ② 외부 일반화 ③ corruption 저하율 ④ 파라미터 수 ⑤ 모델 크기 ⑥ CPU 추론시간을 종합한다.

## 폴더 구조

```
tomato_V2/
├── data/
│   ├── raw/              # 원본 (plantvillage, taiwan, bangladesh) — 수정하지 않음
│   ├── processed/        # PlantVillage train/val/test 분할
│   ├── external/         # 외부 평가셋 (taiwan, bangladesh_bbox)
│   └── corrupted/        # corruption 변형 테스트셋
├── src/
│   ├── config.py         # 경로·하이퍼파라미터 전역 설정
│   ├── class_info.py     # 클래스 정의 + 외부 데이터 클래스 매핑
│   ├── dataset.py        # tf.data 로더
│   ├── data_prep/        # 분할 · 외부 데이터 변환 · 점검
│   ├── models/           # tiny_cnn(A/B/C) · reference · 레지스트리
│   ├── corruption/       # 변형 정의 · 변형셋 생성
│   ├── train/            # 학습 (모델 이름 인자 하나로 통일)
│   ├── evaluate/         # 내부 · 효율 · corruption · 외부 평가
│   ├── summary/          # 종합 비교표 · 최종 선정
│   ├── visualize/        # 그래프
│   └── utils/            # 시드 · 입출력
├── models/               # 학습된 .keras 가중치
├── results/              # _common · internal · efficiency · corruption · external · summary · figures
├── app/                  # Streamlit 웹 프로토타입
├── notebooks/
└── docs/                 # 프로젝트 정리 문서
```

## 실행 순서

```bash
pip install -r requirements.txt

# 데이터 준비
python src/data_prep/split_plantvillage.py
python src/data_prep/prepare_taiwan.py
python src/data_prep/prepare_bangladesh_bbox.py
python src/corruption/make_corrupted_sets.py

# 학습 (모델별)
python src/train/train_model.py --model tiny_cnn_a

# 평가
python src/evaluate/evaluate_internal.py
python src/evaluate/measure_efficiency.py
python src/evaluate/evaluate_corruption.py
python src/evaluate/evaluate_external.py

# 종합
python src/summary/make_summary.py
python src/visualize/plot_results.py
```

## 실행 환경

로컬 CPU 또는 일반 Colab. 고성능 GPU를 전제하지 않는 크기를 목표로 한다.
