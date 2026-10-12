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
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from class_info import CLASS_NAMES, EXTERNAL_PRESENT_CLASSES  # noqa: E402
from config import EXTERNAL_DIRS, SEED, TEST_DIR, model_path  # noqa: E402
from dataset import INTERPOLATION_MODES, make_dataloader, make_external_dataloader  # noqa: E402
from evaluate.evaluate_external import evaluate_model_on_external  # noqa: E402
from models import MODEL_BUILDERS, input_shape_for  # noqa: E402
from utils.seed import set_seed  # noqa: E402

METHODS = ("source", "adabn", "tent", "shot", "eata")


def class_mask_for(dataset_key: str, device) -> torch.Tensor:
    """target에 없다고 설계 단계부터 알려진(class_info.py의 매핑표 기준, ground truth를 본 게 아님)
    클래스의 로짓에 -inf를 더해 예측 후보에서 제외하는 마스크. TENT에 주면 entropy 손실도
    이미 존재하는 클래스들 사이에서만 계산된다(mask-aware adaptation)."""
    allowed = {CLASS_NAMES.index(c) for c in EXTERNAL_PRESENT_CLASSES[dataset_key]}
    mask = torch.zeros(len(CLASS_NAMES), device=device)
    for i in range(len(CLASS_NAMES)):
        if i not in allowed:
            mask[i] = float("-inf")
    return mask


def load_base_model(model_name: str, run_name: str, seed: int, img_size: tuple[int, int], device) -> nn.Module:
    weights_path = model_path(run_name, seed)
    if not weights_path.exists():
        raise FileNotFoundError(f"학습된 가중치가 없습니다: {weights_path}")
    model = MODEL_BUILDERS[model_name](input_shape=(3, *img_size)).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model


def apply_adabn(model: nn.Module, target_loader, device, blend_momentum: float | None = None) -> nn.Module:
    """BN running 통계를 target 쪽으로 옮긴다.

    blend_momentum=None(기본): running_mean/var를 초기화한 뒤 target 데이터 한 번 통과로
    "완전히" 재계산한다 (논문 원안, 역전파 없음) — Bangladesh/PlantDoc처럼 클래스가 많고
    불균형한 데이터에서는 이 완전 교체 자체가 negative transfer를 더 키우는 것으로 진단됨.

    blend_momentum=m (0<m<1): running_mean/var를 초기화하지 않고 source 가중치의 통계를
    그대로 시작점으로 둔 채, 표준 EMA 업데이트(new = (1-m)*old + m*batch)로 target 배치 통계를
    "조금씩만" 섞는다. m이 작을수록 source 쪽에 더 가깝게 남아 완전 교체의 부작용을 줄인다는
    가설을 검증하기 위한 옵션."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if blend_momentum is None:
                m.reset_running_stats()
                m.momentum = None  # None -> 누적 평균(cumulative moving average), 완전 교체
            else:
                m.momentum = blend_momentum  # source 통계를 초기값으로 둔 채 조금씩만 블렌딩

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


def _entropy_of(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)


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


def apply_tent(
    model: nn.Module,
    target_loader,
    device,
    epochs: int = 1,
    lr: float = 1e-3,
    class_mask=None,
    div_weight: float = 0.0,
) -> nn.Module:
    """class_mask를 주면 entropy를 target에 있다고 알려진 클래스들 사이에서만 계산한다
    (mask-aware TENT) — 없는 클래스 쪽으로 확신을 몰아주는 negative transfer를 적응 단계에서부터
    막는다는 게 핵심. None이면 기존처럼 10개 클래스 전체에 대해 entropy를 최소화한다.

    div_weight>0이면 배치 평균 예측 분포의 entropy를 "최대화"하는 diversity 항을 함께 최적화한다
    (SHOT의 information-maximization을 축소 적용). mask-aware TENT를 실측해보니 entropy
    최소화만으로는 "아무 클래스나 하나를 100% 확신"하는 degenerate solution으로 전부 collapse됐는데
    (prediction_distribution.csv로 확인, 3개 데이터셋 전부 한 클래스로만 예측), 이 항이 그 collapse를
    막기 위한 것 — 개별 샘플은 confident하게, 배치 전체로는 여러 클래스에 퍼지게 강제한다."""
    model = _prepare_tent(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)

    model.train()
    for _ in range(epochs):
        for images, _ in target_loader:
            images = images.to(device)
            optimizer.zero_grad()
            logits = model(images)
            if class_mask is not None:
                logits = logits + class_mask
            probs = logits.softmax(dim=1)
            loss = _entropy_of(probs).mean()
            if div_weight > 0:
                mean_probs = probs.mean(dim=0)
                loss = loss - div_weight * _entropy_of(mean_probs)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def _shot_pseudo_labels(model, loader, device, allowed_idx, class_mask, num_classes, return_feats: bool = False):
    """SHOT 논문의 weighted k-means pseudo-labeling을 축소 적용한다. target 존재 클래스
    제한(allowed_idx)은 class_info.py 매핑표(설계 단계부터 알려진 정보)에서 온 것이지
    target ground truth를 보고 정한 게 아니다. 1차: 분류기 softmax를 가중치 삼아 클래스별
    중심(centroid)을 구하고 코사인 거리로 재할당 → 2차: 그 결과(one-hot)로 중심을 다시 구해
    한 번 더 재할당한다(논문의 2-step 절차)."""
    model.eval()
    feats_list, probs_list = [], []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            feats = model.head.pool(model.features(images)).flatten(1)
            logits = model.head.fc(feats) + class_mask
            probs_list.append(logits.softmax(dim=1))
            feats_list.append(feats)
    feats = torch.cat(feats_list)
    probs = torch.cat(probs_list)
    disallowed = [i for i in range(num_classes) if i not in allowed_idx]

    def assign(weight: torch.Tensor) -> torch.Tensor:
        centroids = (weight.T @ feats) / (weight.sum(dim=0).unsqueeze(1) + 1e-8)
        feats_n = F.normalize(feats, dim=1)
        centroids_n = F.normalize(centroids, dim=1)
        sims = feats_n @ centroids_n.T
        if disallowed:
            sims[:, disallowed] = -1e9
        return sims.argmax(dim=1)

    pseudo = assign(probs)
    pseudo = assign(F.one_hot(pseudo, num_classes).float())
    if return_feats:
        return pseudo.cpu(), feats.cpu()
    return pseudo.cpu()


def _aps_refine(
    feats: torch.Tensor,
    pseudo_labels: torch.Tensor,
    allowed_idx,
    num_classes: int,
    k: int = 4,
    iters: int = 2,
    gamma: float = 0.6,
):
    """UPA(Chen et al., Uncertainty-Aware Pseudo-Label Filtering for SFUDA, Neurocomputing 2024,
    github.com/chenxi52/UPA)의 APS(Adaptive Pseudo-label Selection)만 분리 적용한다(CACL은 제외).
    논문 식 4~6: 각 샘플의 K-최근접 이웃(코사인 유사도)이 현재 임시 라벨에 투표해 클래스 posterior를
    추정하고(식4), argmax로 라벨을 갱신하며(식5), 이를 iters번 반복한다. 마지막에 "이웃 중 최종 라벨과
    일치하는 비율(유사도 가중)"을 confidence로 삼아(식6), 클래스별로 confidence 상위 gamma%만
    "신뢰할 수 있는 샘플"로 표시한다 — SHOT의 pseudo-label을 그대로 믿지 않고, 이웃 합의가 약한
    샘플은 fine-tuning의 pseudo-label loss에서 제외하자는 아이디어."""
    n = feats.size(0)
    k = min(k, n - 1)
    feats_n = F.normalize(feats, dim=1)
    sims = feats_n @ feats_n.T
    sims.fill_diagonal_(-2.0)  # 자기 자신 제외
    topk_sims, topk_idx = sims.topk(k, dim=1)

    disallowed = [i for i in range(num_classes) if i not in allowed_idx]
    temp_labels = pseudo_labels.clone()
    for _ in range(iters):
        neighbor_labels = temp_labels[topk_idx]
        onehot = F.one_hot(neighbor_labels, num_classes).float()
        posterior = (topk_sims.unsqueeze(-1) * onehot).sum(dim=1) / k  # 식 4
        if disallowed:
            posterior[:, disallowed] = -1e9
        temp_labels = posterior.argmax(dim=1)  # 식 5

    neighbor_labels = temp_labels[topk_idx]
    agree = (neighbor_labels == temp_labels.unsqueeze(1)).float()
    confidence = (agree * topk_sims).sum(dim=1) / k  # 식 6

    confident_mask = torch.zeros(n, dtype=torch.bool)
    for c in allowed_idx:
        class_idx = (temp_labels == c).nonzero(as_tuple=True)[0]
        if len(class_idx) == 0:
            continue
        n_keep = max(1, int(round(len(class_idx) * gamma)))
        top_local = confidence[class_idx].topk(min(n_keep, len(class_idx))).indices
        confident_mask[class_idx[top_local]] = True

    return temp_labels, confident_mask


def apply_shot(
    model: nn.Module,
    target_loader,
    device,
    dataset_key: str,
    epochs: int = 1,
    lr: float = 1e-3,
    cls_loss_weight: float = 0.3,
    div_weight: float = 1.0,
    aps: bool = False,
    aps_k: int = 4,
    aps_iters: int = 2,
    aps_gamma: float = 0.6,
    class_balance: bool = False,
) -> nn.Module:
    """SHOT (Liang et al., ICML 2020) 축소 적용. source classifier(head.fc)는 고정하고
    feature extractor(features)만 업데이트한다 — TENT(BN affine만 조금 흔드는 가장 약한 적응)보다
    훨씬 큰 폭으로 적응하는 방법. loss = pseudo-label cross-entropy(가중치 cls_loss_weight)
    + information-maximization(entropy 최소화 + batch 다양성 최대화, apply_tent의 collapse 방지
    항과 동일한 수식). class_mask는 pseudo-labeling과 손실 계산 양쪽에 항상 적용한다 —
    partial label-space 문제를 이미 알고 시작하는 게 이 축소판의 핵심 전제라서 끄는 옵션은 없다.

    aps=True면 UPA의 APS(Adaptive Pseudo-label Selection, _aps_refine 참고)로 pseudo-label을
    KNN 이웃 합의 기준으로 다시 다듬고, 클래스별 confidence 상위 aps_gamma%만 pseudo-label
    cross-entropy 항에 반영한다(IM loss는 그대로 전체 배치에 적용) — SHOT의 클러스터링 결과를
    무조건 믿지 않고, 이웃끼리 의견이 갈리는 애매한 샘플은 fine-tuning 신호에서 빼는 것.

    class_balance=True면 pseudo-label cross-entropy를 그 epoch의 pseudo-label 클래스 분포의
    역빈도로 가중한다 — Bangladesh처럼 pseudo-label 자체가 불균형하면(예: Leaf_Mold가 많고
    Target_Spot이 적으면) 다수 클래스가 gradient를 지배해 소수 클래스가 더 밀리는데, 이걸
    완화하기 위한 것. APS와 독립적으로 켤 수 있다."""
    allowed_idx = sorted({CLASS_NAMES.index(c) for c in EXTERNAL_PRESENT_CLASSES[dataset_key]})
    class_mask = class_mask_for(dataset_key, device)
    num_classes = len(CLASS_NAMES)

    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.features.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(model.features.parameters(), lr=lr)

    for _ in range(epochs):
        pseudo_labels, pl_feats = _shot_pseudo_labels(
            model, target_loader, device, allowed_idx, class_mask, num_classes, return_feats=True
        )
        if aps:
            pseudo_labels, confident_mask = _aps_refine(
                pl_feats, pseudo_labels, allowed_idx, num_classes, k=aps_k, iters=aps_iters, gamma=aps_gamma,
            )
        else:
            confident_mask = torch.ones_like(pseudo_labels, dtype=torch.bool)

        if class_balance:
            counts = torch.bincount(
                pseudo_labels[confident_mask], minlength=num_classes
            ).float().clamp(min=1)
            ce_weight = torch.zeros(num_classes)
            ce_weight[allowed_idx] = 1.0 / counts[allowed_idx]
            ce_weight[allowed_idx] *= len(allowed_idx) / ce_weight[allowed_idx].sum()  # 평균 1이 되게 정규화
            ce_weight = ce_weight.to(device)
        else:
            ce_weight = None

        model.train()
        idx = 0
        for images, _ in target_loader:
            images = images.to(device)
            bsz = images.size(0)
            batch_pseudo = pseudo_labels[idx: idx + bsz].to(device)
            batch_confident = confident_mask[idx: idx + bsz].to(device)
            idx += bsz

            optimizer.zero_grad()
            feats = model.head.pool(model.features(images)).flatten(1)
            logits = model.head.fc(feats) + class_mask
            probs = logits.softmax(dim=1)

            im_loss = _entropy_of(probs).mean() - div_weight * _entropy_of(probs.mean(dim=0))
            if batch_confident.any():
                cls_loss = F.cross_entropy(
                    logits[batch_confident], batch_pseudo[batch_confident], weight=ce_weight,
                )
            else:
                cls_loss = images.new_zeros(())
            loss = im_loss + cls_loss_weight * cls_loss
            loss.backward()
            optimizer.step()

    model.eval()
    return model


_FISHER_CACHE: dict[tuple[str, str, int, tuple[int, int]], tuple[dict, dict]] = {}


def _compute_fisher(model_name: str, run_name: str, seed: int, img_size: tuple[int, int], device, n_samples: int = 2000):
    """EATA의 anti-forgetting 정규화에 쓸 Fisher information을 source(PlantVillage) test셋으로 추정한다.
    target label은 안 쓰지만, source 쪽 라벨 있는 데이터(원래 학습에도 쓰인 PlantVillage test셋)를
    쓰는 것이라 "target-label 미사용" 원칙과는 무관하다. 모델/seed/해상도 조합별로 한 번만 계산해 캐시한다."""
    cache_key = (model_name, run_name, seed, img_size)
    if cache_key in _FISHER_CACHE:
        return _FISHER_CACHE[cache_key]

    fisher_model = load_base_model(model_name, run_name, seed, img_size, device)
    fisher_model = _prepare_tent(fisher_model)  # Fisher도 TENT/EATA가 실제로 여는 BN affine 파라미터 기준으로 계산
    trainable = {n: p for n, p in fisher_model.named_parameters() if p.requires_grad}
    anchor = {n: p.detach().clone() for n, p in trainable.items()}

    loader = make_dataloader(TEST_DIR, img_size, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    fisher = {n: torch.zeros_like(p) for n, p in trainable.items()}
    seen = 0
    fisher_model.train()  # EATA 원 구현과 동일하게 BN은 배치 통계로 forward (track_running_stats=False 상태 유지)
    for images, labels in loader:
        if seen >= n_samples:
            break
        images, labels = images.to(device), labels.to(device)
        fisher_model.zero_grad()
        logits = fisher_model(images)
        loss = criterion(logits, labels)
        loss.backward()
        for n, p in trainable.items():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2 * images.size(0)
        seen += images.size(0)
    for n in fisher:
        fisher[n] /= max(seen, 1)

    _FISHER_CACHE[cache_key] = (fisher, anchor)
    return fisher, anchor


def apply_eata(
    model: nn.Module,
    target_loader,
    device,
    fisher: dict,
    anchor: dict,
    lr: float = 1e-3,
    e0_factor: float = 0.4,
    d0: float = 0.05,
    fisher_alpha: float = 1.0,
    class_mask=None,
) -> nn.Module:
    """EATA (Niu et al., ICML 2022) — 공식 구현(github.com/mr-eggplant/EATA)의 핵심 두 아이디어를
    이 프로젝트 구조에 맞춰 재구현. TENT와 같은 지점(BN affine만 학습)에서 시작하되:

    1) Sample-efficient entropy minimization: entropy가 E0=e0_factor*ln(클래스 수)보다 높은
       (확신 없는) 샘플은 아예 적응에서 제외하고, 남은 샘플 중에서도 최근 배치 평균 예측과
       코사인 유사도가 d0보다 높은(=중복되는) 샘플은 가중치를 0으로 줘 적응 "예산"을 아낀다.
    2) Anti-forgetting 정규화: source test셋으로 추정한 Fisher information을 가중치 삼아
       (현재 파라미터 - 원본 source 파라미터)^2 항을 손실에 더해, target에 맞추다가 source
       지식을 과도하게 잊어버리는 것(=지금까지 관찰한 SHOT epoch 과적합과 같은 종류의 문제)을
       억제한다."""
    model = _prepare_tent(model)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam([p for _, p in trainable], lr=lr)

    num_classes = len(CLASS_NAMES) if class_mask is None else int((class_mask == 0).sum().item())
    e0 = e0_factor * math.log(max(num_classes, 2))
    moving_avg = None

    model.train()
    for images, _ in target_loader:
        images = images.to(device)
        logits = model(images)
        if class_mask is not None:
            logits = logits + class_mask
        probs = logits.softmax(dim=1)
        entropy = _entropy_of(probs)

        keep = entropy < e0
        if keep.sum() == 0:
            continue
        kept_probs = probs[keep]
        kept_entropy = entropy[keep]

        with torch.no_grad():
            batch_mean = kept_probs.mean(dim=0)
        if moving_avg is None:
            div_weight = torch.ones_like(kept_entropy)
        else:
            sims = F.cosine_similarity(kept_probs, moving_avg.unsqueeze(0).expand_as(kept_probs), dim=1)
            div_weight = (sims < d0).float()  # 이전 배치 평균과 너무 비슷한(중복) 샘플은 가중치 0
        moving_avg = batch_mean.detach() if moving_avg is None else 0.9 * moving_avg + 0.1 * batch_mean.detach()

        entropy_weight = 1.0 / torch.exp((kept_entropy - e0).clamp(max=20))
        sample_weight = entropy_weight * div_weight
        if sample_weight.sum() <= 1e-8:
            continue
        entropy_loss = (sample_weight * kept_entropy).sum() / sample_weight.sum()

        reg = images.new_zeros(())
        for n, p in trainable:
            if n in fisher:
                reg = reg + (fisher[n] * (p - anchor[n]) ** 2).sum()
        loss = entropy_loss + fisher_alpha * reg

        optimizer.zero_grad()
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
    masked: bool = False,
    div_weight: float = 0.0,
    adabn_blend_momentum: float | None = None,
    shot_epochs: int = 2,
    shot_lr: float = 1e-3,
    shot_cls_weight: float = 0.3,
    shot_aps: bool = False,
    shot_aps_k: int = 4,
    shot_aps_iters: int = 2,
    shot_aps_gamma: float = 0.6,
    shot_class_balance: bool = False,
    eata_lr: float = 0.1,
    eata_e0_factor: float = 0.4,
    eata_d0: float = 0.05,
    eata_fisher_alpha: float = 1.0,
) -> dict:
    """masked=True면 class_mask_for()로 target에 없는 클래스를 가린다.
    TENT는 적응 단계(entropy 손실)에도 마스크를 넣어 mask-aware하게 적응하고,
    AdaBN은 통계 재계산 자체가 loss-free라 적응 단계는 그대로 두고 최종 평가에만 마스크를 적용한다.
    SHOT·EATA는 마스킹이 방법 자체에 내장돼 있어 masked 인자와 무관하게 항상 적용된다.

    div_weight: TENT 전용, >0이면 diversity 정규화(collapse 방지)를 추가한다.
    adabn_blend_momentum: AdaBN 전용, 주면 완전 교체 대신 해당 momentum으로 source/target 통계를 블렌딩한다."""
    set_seed(seed)
    model = load_base_model(model_name, base_run_name, seed, img_size, device)
    loader = make_external_dataloader(data_dir, img_size, batch_size, interpolation=INTERPOLATION_MODES[interp])
    masked = masked or method in ("shot", "eata")
    class_mask = class_mask_for(dataset_key, device) if masked else None

    t0 = time.time()
    if method == "source":
        pass  # 적응 없음 — masked=True면 "마스킹만" 기준선, False면 순수 source-only
    elif method == "adabn":
        model = apply_adabn(model, loader, device, blend_momentum=adabn_blend_momentum)
    elif method == "tent":
        model = apply_tent(
            model, loader, device, epochs=tent_epochs, lr=tent_lr, class_mask=class_mask, div_weight=div_weight
        )
    elif method == "shot":
        model = apply_shot(
            model, loader, device, dataset_key, epochs=shot_epochs, lr=shot_lr, cls_loss_weight=shot_cls_weight,
            div_weight=div_weight if div_weight > 0 else 1.0,
            aps=shot_aps, aps_k=shot_aps_k, aps_iters=shot_aps_iters, aps_gamma=shot_aps_gamma,
            class_balance=shot_class_balance,
        )
    elif method == "eata":
        fisher, anchor = _compute_fisher(model_name, base_run_name, seed, img_size, device)
        model = apply_eata(
            model, loader, device, fisher, anchor, lr=eata_lr, e0_factor=eata_e0_factor, d0=eata_d0,
            fisher_alpha=eata_fisher_alpha, class_mask=class_mask,
        )
    else:
        raise ValueError(f"알 수 없는 method: {method}")
    adapt_time = time.time() - t0

    run_name = f"{base_run_name}_{method}"
    if masked:
        run_name += "_masked"
    if div_weight > 0:
        run_name += f"_div{div_weight:g}"
    if adabn_blend_momentum is not None:
        run_name += f"_blend{adabn_blend_momentum:g}"
    if method == "shot" and (shot_epochs != 2 or shot_cls_weight != 0.3):
        run_name += f"_ep{shot_epochs}_cw{shot_cls_weight:g}"
    if method == "shot" and shot_aps:
        run_name += f"_aps{shot_aps_gamma:g}"
    if method == "shot" and shot_class_balance:
        run_name += "_cb"
    if method == "eata" and (eata_e0_factor != 0.4 or eata_d0 != 0.05 or eata_fisher_alpha != 1.0):
        run_name += f"_e0{eata_e0_factor:g}_d0{eata_d0:g}_fa{eata_fisher_alpha:g}"
    metrics = evaluate_model_on_external(
        model, model_name, run_name, dataset_key, data_dir, img_size, interp, seed, device, batch_size,
        class_mask=class_mask,
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
    parser.add_argument(
        "--masked", action="store_true",
        help="class-restriction 마스킹 적용 (TENT는 entropy 손실에도 반영되는 mask-aware adaptation)",
    )
    parser.add_argument(
        "--div-weight", type=float, default=0.0,
        help="TENT 전용: diversity 정규화 가중치 (>0이면 collapse 방지용 batch diversity 항 추가)",
    )
    parser.add_argument(
        "--adabn-blend-momentum", type=float, default=None,
        help="AdaBN 전용: 주면 완전 교체 대신 이 momentum으로 source/target BN 통계를 블렌딩",
    )
    parser.add_argument(
        "--shot-epochs", type=int, default=2,
        help="2로 고정(모든 데이터셋 공통) — target accuracy를 보고 데이터셋별로 고르면 label-free 전제가 "
        "깨지므로, epoch 수에 따른 민감도(결과 커지면 Taiwan은 계속 개선되지만 Bangladesh/PlantDoc은 "
        "pseudo-label 과적합으로 하락)는 별도 분석으로만 보고하고 메인 비교에는 이 고정값을 쓴다",
    )
    parser.add_argument("--shot-lr", type=float, default=1e-3)
    parser.add_argument(
        "--shot-cls-weight", type=float, default=0.3,
        help="SHOT 전용: pseudo-label cross-entropy 항의 가중치 (나머지는 information-maximization)",
    )
    parser.add_argument(
        "--shot-aps", action="store_true",
        help="SHOT 전용: UPA(Chen et al., Neurocomputing 2024)의 APS(Adaptive Pseudo-label Selection)를 "
        "추가한다 — KNN 이웃 합의로 pseudo-label을 다듬고 클래스별 confidence 상위 shot-aps-gamma%만 "
        "pseudo-label loss에 반영한다 (CACL은 미적용, APS만 분리 검증)",
    )
    parser.add_argument("--shot-aps-k", type=int, default=4, help="APS 전용: KNN 이웃 수 (논문 기본값 4)")
    parser.add_argument("--shot-aps-iters", type=int, default=2, help="APS 전용: 라벨 재할당 반복 횟수 (논문 기본값 2)")
    parser.add_argument(
        "--shot-aps-gamma", type=float, default=0.6,
        help="APS 전용: 클래스별 confidence 상위 몇 %를 pseudo-label loss에 쓸지 (논문 기본값 0.6)",
    )
    parser.add_argument(
        "--shot-class-balance", action="store_true",
        help="SHOT 전용: pseudo-label cross-entropy를 그 epoch의 pseudo-label 클래스 분포 역빈도로 "
        "가중한다 (Bangladesh처럼 pseudo-label 자체가 불균형할 때 다수 클래스 쏠림 완화용, APS와 독립적)",
    )
    parser.add_argument("--eata-lr", type=float, default=0.1, help="TENT/SHOT 기본값(1e-3)보다 훨씬 큼 — entropy 가중치·필터링으로 유효 gradient가 작아져서 같은 lr로는 거의 적응이 안 됨 (실측 확인)")
    parser.add_argument(
        "--eata-e0-factor", type=float, default=0.4,
        help="EATA 전용: entropy 임계값 E0 = 이 값 * ln(클래스 수). 이보다 entropy 높은(불확실한) 샘플은 적응에서 제외",
    )
    parser.add_argument(
        "--eata-d0", type=float, default=0.05,
        help="EATA 전용: 직전 배치 평균 예측과 코사인 유사도가 이보다 크면(중복 샘플) 가중치 0",
    )
    parser.add_argument(
        "--eata-fisher-alpha", type=float, default=1.0,
        help="EATA 전용: anti-forgetting 정규화 가중치 (source test셋 Fisher information 기반, 크면 원본 가중치에서 덜 움직임)",
    )
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
                    device, tent_epochs=args.tent_epochs, tent_lr=args.tent_lr, masked=args.masked,
                    div_weight=args.div_weight, adabn_blend_momentum=args.adabn_blend_momentum,
                    shot_epochs=args.shot_epochs, shot_lr=args.shot_lr, shot_cls_weight=args.shot_cls_weight,
                    shot_aps=args.shot_aps, shot_aps_k=args.shot_aps_k, shot_aps_iters=args.shot_aps_iters,
                    shot_aps_gamma=args.shot_aps_gamma, shot_class_balance=args.shot_class_balance,
                    eata_lr=args.eata_lr, eata_e0_factor=args.eata_e0_factor, eata_d0=args.eata_d0,
                    eata_fisher_alpha=args.eata_fisher_alpha,
                )
            )

    print("\n=== 요약 ===")
    for r in results:
        print(f"{r['dataset']:16s} {r['run_name']:32s} acc={r['accuracy']:.4f}  macro_f1={r['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
