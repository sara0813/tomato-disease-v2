"""3단계: 효율성 측정 (PyTorch).

모델별로 파라미터 수, state_dict 파일 크기(MB), FLOPs(Conv2d/Linear 기준),
CPU 추론시간(ms/image, batch=1)을 측정한다.
추론시간은 워밍업 후 여러 번 반복해 중앙값을 쓴다(로컬 CPU 기준, torch.no_grad()).

산출물 → results/efficiency/efficiency.csv
"""

import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import EFFICIENCY_RESULT_DIR, TINY_MODELS, model_path  # noqa: E402
from models import build_model, input_shape_for  # noqa: E402


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_flops(model: nn.Module, input_shape: tuple[int, int, int]) -> int:
    """Conv2d/Linear의 FLOPs만 센다 (BN/ReLU/Pooling은 무시 — 전체 대비 미미함).
    Conv2d는 groups(depthwise 포함)를 반영한다. 곱셈+덧셈을 2 FLOPs로 카운트."""
    flops = 0

    def hook(module, _inp, output):
        nonlocal flops
        if isinstance(module, nn.Conv2d):
            out_h, out_w = output.shape[2], output.shape[3]
            kh, kw = module.kernel_size
            flops += (
                2
                * (module.in_channels // module.groups)
                * kh
                * kw
                * module.out_channels
                * out_h
                * out_w
            )
        elif isinstance(module, nn.Linear):
            flops += 2 * module.in_features * module.out_features

    handles = [
        m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))
    ]

    model.eval()
    with torch.no_grad():
        model(torch.randn(1, *input_shape))

    for h in handles:
        h.remove()

    return flops


def measure_inference_ms(model: nn.Module, input_shape: tuple[int, int, int], n_warmup=10, n_measure=50) -> float:
    """batch=1 기준 CPU 추론시간(ms)의 중앙값."""
    model.eval()
    dummy = torch.randn(1, *input_shape)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)

        times = []
        for _ in range(n_measure):
            t0 = time.perf_counter()
            model(dummy)
            times.append(time.perf_counter() - t0)

    times.sort()
    return times[len(times) // 2] * 1000


def main() -> None:
    rows = []
    for name in TINY_MODELS:
        shape = input_shape_for(name)
        model = build_model(name)

        weights_path = model_path(name)
        file_size_mb = weights_path.stat().st_size / (1024 * 1024) if weights_path.exists() else None
        if weights_path.exists():
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))

        n_params = count_params(model)
        flops = count_flops(model, shape)
        inf_ms = measure_inference_ms(model, shape)

        rows.append(
            {
                "model": name,
                "input_size": f"{shape[1]}x{shape[2]}x{shape[0]}",
                "params": n_params,
                "flops": flops,
                "flops_mflops": round(flops / 1e6, 3),
                "file_size_mb": round(file_size_mb, 4) if file_size_mb is not None else None,
                "cpu_inference_ms": round(inf_ms, 3),
            }
        )
        print(
            f"[{name}] params={n_params:,}  FLOPs={flops/1e6:.2f}M  "
            f"size={file_size_mb:.3f}MB  inference={inf_ms:.2f}ms/image"
        )

    df = pd.DataFrame(rows)
    EFFICIENCY_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EFFICIENCY_RESULT_DIR / "efficiency.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
