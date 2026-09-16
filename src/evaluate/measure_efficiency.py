"""3단계: 효율성 측정 (PyTorch).

모델별로 파라미터 수(sum(p.numel() for p in model.parameters())),
state_dict 파일 크기(MB), FLOPs, CPU 추론시간(ms/image)을 측정한다.
추론시간은 워밍업 후 여러 번 반복해 중앙값을 쓴다(로컬 CPU 기준, torch.no_grad()).

산출물 → results/efficiency/efficiency.csv
"""

# TODO: 구현
