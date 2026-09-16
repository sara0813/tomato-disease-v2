"""2단계: PlantVillage 원본 테스트셋 성능 평가 (PyTorch).

config.model_path(name)에서 state_dict를 불러와 model.eval() + torch.no_grad()로
추론한다.

산출물 → results/internal/<model>/
    metrics.json          Accuracy, Macro F1, Weighted F1
    classification_report.csv
    confusion_matrix.csv
"""

# TODO: 구현
