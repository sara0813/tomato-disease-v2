"""학습 스크립트 (모델 이름을 인자로 받는 단일 진입점, PyTorch).

사용 예:
    python src/train/train_model.py --model tiny_cnn_a
    python src/train/train_model.py --model mobilenetv2 --epochs 15

V1처럼 모델마다 train_*.py를 따로 두지 않고 한 파일로 통일한다.
models.build_model(name)으로 구조만 가져온 뒤, 여기서 optimizer/loss/loop을
모든 모델에 동일하게 적용해 학습 설정 일관성을 유지한다.

  loss       nn.CrossEntropyLoss
  optimizer  Adam
  device     CPU 기준 (torch.cuda.is_available()이면 자동으로 사용)

학습 곡선과 학습 시간은 results/internal/<model>/ 에 저장하고,
가중치는 config.model_path(model_name) (.pt, state_dict)에 저장한다.
"""

# TODO: 구현
