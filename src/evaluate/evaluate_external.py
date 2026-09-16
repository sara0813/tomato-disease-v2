"""5단계: 외부 데이터 일반화 평가 (Taiwan / Bangladesh BBox).

외부 데이터에는 PlantVillage 10개 클래스 중 일부만 존재하므로,
실제로 존재하는 클래스에 한정해 지표를 계산한다.

Domain shift 진단을 위해 예측 분포도 함께 남긴다.
(1차 실험에서 다수 모델이 외부 이미지를 Late Blight로 편향 예측했다.)

산출물 → results/external/<dataset>/<model>/
    metrics.json
    classification_report.csv
    confusion_matrix.csv
    prediction_distribution.csv
"""

# TODO: 구현
