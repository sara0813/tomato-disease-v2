"""4단계: 강건성 평가.

data/corrupted/<type>/level<n>/ 전체에 대해 모델별 정확도를 측정하고,
원본 테스트셋 대비 성능 저하율을 계산한다.

    저하율(%) = (원본 Accuracy - 변형 Accuracy) / 원본 Accuracy * 100

산출물 → results/corruption/<model>/corruption_metrics.csv
         (컬럼: model, corruption_type, level, accuracy, macro_f1, drop_rate)
"""

# TODO: 구현
