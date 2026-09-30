"""Domain Adaptation (AdaBN / TENT) — source-free, target label 미사용.

기준 모델(tiny_cnn_c, 96px, bicubic)의 학습된 가중치에서 시작해, 외부(target) 데이터의
"이미지만" 이용해 두 가지 방식으로 적응시킨 뒤 외부 평가를 다시 돌린다.
AdaBN과 TENT는 항상 "원본 source-only 가중치"에서 독립적으로 시작한다
(AdaBN 결과에 TENT를 이어붙이지 않음 — docs/토마토_병해분류_Domain_Adaptation_실행계획_v2.md 4장 참고).

AdaBN (Li et al., 2016)
    BatchNorm의 running_mean/var를 초기화한 뒤, momentum=None(누적 평균) 상태로 target
    데이터를 한 번 통과시켜 BN 통계를 target 도메인 것으로 완전히 교체한다. 역전파 없음.

TENT (Wang et al., ICLR 2021)
    BN의 affine parameter(weight/bias)만 학습 가능하게 열고 나머지는 전부 고정한 뒤,
    라벨 없이 예측 entropy를 최소화하도록 몇 epoch 최적화한다. BN은 running 통계 대신
    항상 현재 배치 통계를 쓰도록(track_running_stats=False) 바꾼다(공식 구현과 동일).

산출물 → results/external/<dataset>/<base_run_name>_<method>/seed<seed>/
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
from torch import nn

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import EXTERNAL_DIRS, SEED, model_path  # noqa: E402
from dataset import INTERPOLATION_MODES, make_external_dataloader  # noqa: E402
from evaluate.evaluate_external import evaluate_model_on_external  # noqa: E402
from models import MODEL_BUILDERS, input_shape_for  # noqa: E402
from utils.seed import set_seed  # noqa: E402

METHODS = ("adabn", "tent")


def load_base_model(model_name: str, run_name: str, seed: int, img_size: tuple[int, int], device) -> nn.Module:
    weights_path = model_path(run_name, seed)
    if not weights_path.exists():
        raise FileNotFoundError(f"학습된 가중치가 없습니다: {weights_path}")
    model = MODEL_BUILDERS[model_name](input_shape=(3, *img_size)).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model


def apply_adabn(model: nn.Module, target_loader, device) -> nn.Module:
    """BN running 통계를 초기화하고 target 데이터 한 번 통과로 완전히 재계산한다 (역전파 없음)."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # None -> 누적 평균(cumulative moving average), 논문의 "정확한 통계 교체"에 해당

    model.train()
    with torch.no_grad():
        for images, _ in target_loader:
            images = images.to(device)
            model(images)
    model.eval()
    return model


def _entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=1)
    log_probs = logits.log_softmax(dim=1)
    return -(probs * log_probs).sum(dim=1).mean()


def _prepare_tent(model: nn.Module) -> nn.Module:
    """BN affine parameter만 학습 가능하게 열고, BN이 항상 배치 통계를 쓰도록 바꾼다.
    나머지 파라미터는 전부 requires_grad=False로 고정 (공식 TENT 구현과 동일한 설정)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def apply_tent(model: nn.Module, target_loader, device, epochs: int = 1, lr: float = 1e-3) -> nn.Module:
    model = _prepare_tent(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)

    model.train()
    for _ in range(epochs):
        for images, _ in target_loader:
            images = images.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = _entropy_loss(logits)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def run_one(
    method: str,
    model_name: str,
    base_run_name: str,
    dataset_key: str,
    data_dir: Path,
    img_size: tuple[int, int],
    interp: str,
    seed: int,
    device,
    batch_size: int = 32,
    tent_epochs: int = 1,
    tent_lr: float = 1e-3,
) -> dict:
    set_seed(seed)
    model = load_base_model(model_name, base_run_name, seed, img_size, device)
    loader = make_external_dataloader(data_dir, img_size, batch_size, interpolation=INTERPOLATION_MODES[interp])

    t0 = time.time()
    if method == "adabn":
        model = apply_adabn(model, loader, device)
    elif method == "tent":
        model = apply_tent(model, loader, device, epochs=tent_epochs, lr=tent_lr)
    else:
        raise ValueError(f"알 수 없는 method: {method}")
    adapt_time = time.time() - t0

    run_name = f"{base_run_name}_{method}"
    metrics = evaluate_model_on_external(
        model, model_name, run_name, dataset_key, data_dir, img_size, interp, seed, device, batch_size
    )
    metrics["adapt_time_sec"] = adapt_time
    print(f"[{run_name}] {dataset_key} adapt_time={adapt_time:.1f}s")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Domain Adaptation (AdaBN/TENT) 외부 평가")
    parser.add_argument("--model", default="tiny_cnn_c")
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--interp", choices=list(INTERPOLATION_MODES.keys()), default="bicubic")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--dataset", choices=list(EXTERNAL_DIRS.keys()) + ["all"], default="bangladesh_bbox",
        help="기본값: 실행계획 v2에서 1순위로 정한 bangladesh_bbox",
    )
    parser.add_argument("--method", choices=list(METHODS) + ["all"], default="all")
    parser.add_argument("--tent-epochs", type=int, default=1)
    parser.add_argument("--tent-lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = (args.img_size, args.img_size)
    base_run_name = args.model if args.img_size == input_shape_for(args.model)[1] and args.interp == "bilinear" else (
        f"{args.model}_res{args.img_size}" if args.interp == "bilinear" else f"{args.model}_res{args.img_size}_{args.interp}"
    )

    datasets = list(EXTERNAL_DIRS.items()) if args.dataset == "all" else [(args.dataset, EXTERNAL_DIRS[args.dataset])]
    methods = list(METHODS) if args.method == "all" else [args.method]

    results = []
    for dataset_key, data_dir in datasets:
        for method in methods:
            results.append(
                run_one(
                    method, args.model, base_run_name, dataset_key, data_dir, img_size, args.interp, args.seed,
                    device, tent_epochs=args.tent_epochs, tent_lr=args.tent_lr,
                )
            )

    print("\n=== 요약 ===")
    for r in results:
        print(f"{r['dataset']:16s} {r['run_name']:32s} acc={r['accuracy']:.4f}  macro_f1={r['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
