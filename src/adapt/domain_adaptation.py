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

METHODS = ("source", "adabn", "tent", "shot", "eata", "rgv", "aad", "cowa")


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


def _balanced_assign(sims: torch.Tensor, allowed_idx, disallowed, cap_ratio: float, max_iters: int = 50) -> torch.Tensor:
    """클래스별 할당 개수를 "공정 몫(n_samples / 클래스 수)의 cap_ratio배"로 상한을 둔 뒤, 그
    상한을 넘는 클래스에서 가장 낮은 유사도(확신이 약한) 샘플만 다음으로 유사도가 높은 허용
    클래스로 재배정한다(그리디). Bangladesh에서 관찰된 "한 클래스로 몰림"(예: Target_Spot
    예측 0장→211장) 같은 폭주를 억제하려는 목적 — target label은 전혀 안 쓰고, 샘플 수·허용
    클래스 수(마스킹과 동일하게 설계 단계부터 아는 정보)만으로 정한 상한이라 label-free 원칙과
    무관하다."""
    n = sims.size(0)
    cap = max(1, math.ceil(n / len(allowed_idx) * cap_ratio))
    work = sims.clone()
    if disallowed:
        work[:, disallowed] = -1e9
    labels = work.argmax(dim=1)

    for _ in range(max_iters):
        counts = torch.bincount(labels, minlength=sims.size(1))
        over_classes = [c for c in allowed_idx if counts[c] > cap]
        if not over_classes:
            break
        changed = False
        for c in over_classes:
            idx_cls = (labels == c).nonzero(as_tuple=True)[0]
            excess = int(counts[c].item()) - cap
            if excess <= 0:
                continue
            scores = work[idx_cls, c]
            kick_local = scores.topk(min(excess, idx_cls.numel()), largest=False).indices
            kick_idx = idx_cls[kick_local]
            work[kick_idx, c] = -1e9  # 이 클래스는 더 이상 후보가 아님
            labels[kick_idx] = work[kick_idx].argmax(dim=1)
            changed = True
        if not changed:
            break
    return labels


def _shot_pseudo_labels(
    model, loader, device, allowed_idx, class_mask, num_classes, return_feats: bool = False,
    balanced_cap_ratio: float | None = None,
):
    """SHOT 논문의 weighted k-means pseudo-labeling을 축소 적용한다. target 존재 클래스
    제한(allowed_idx)은 class_info.py 매핑표(설계 단계부터 알려진 정보)에서 온 것이지
    target ground truth를 보고 정한 게 아니다. 1차: 분류기 softmax를 가중치 삼아 클래스별
    중심(centroid)을 구하고 코사인 거리로 재할당 → 2차: 그 결과(one-hot)로 중심을 다시 구해
    한 번 더 재할당한다(논문의 2-step 절차). balanced_cap_ratio가 주어지면 각 단계의 재할당에
    _balanced_assign으로 클래스별 상한을 적용한다."""
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
        if balanced_cap_ratio is not None:
            return _balanced_assign(sims, allowed_idx, disallowed, balanced_cap_ratio)
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


def _contrastive_loss(
    feats: torch.Tensor, labels: torch.Tensor, confident_mask: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """UPA의 CACL(Class-Aware Contrastive Learning, 식 8). anchor는 confident 샘플(APS의
    confident_mask)만 쓰고, 양성쌍은 그 안에서 같은 pseudo-label을 가진 다른 샘플, 분모(음성쌍
    포함)는 배치 전체(자기 자신 제외) — SupCon과 같은 형태지만 anchor·양성쌍을 신뢰 가능한
    샘플로만 제한한 버전이다. feats는 별도 projection head 없이 L2-normalize만 적용한다."""
    z = F.normalize(feats, dim=1)
    sim = z @ z.T / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()  # 수치 안정화(softmax shift)
    sim.fill_diagonal_(-1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
    both_confident = confident_mask.unsqueeze(0) & confident_mask.unsqueeze(1)
    pos_mask = same_class & both_confident
    pos_mask.fill_diagonal_(False)

    pos_count = pos_mask.sum(dim=1)
    anchor_mask = confident_mask & (pos_count > 0)
    if not anchor_mask.any():
        return feats.new_zeros(())

    per_anchor_loss = -(pos_mask.float() * log_prob).sum(dim=1) / pos_count.clamp(min=1)
    return per_anchor_loss[anchor_mask].mean()


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
    cacl: bool = False,
    cacl_temperature: float = 0.1,
    balanced_cap: bool = False,
    balanced_cap_ratio: float = 1.5,
    exclude_classes: tuple[str, ...] = (),
    aad_weight: float = 0.0,
    aad_k: int = 4,
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
    완화하기 위한 것. APS와 독립적으로 켤 수 있다.

    cacl=True면 UPA의 CACL(Class-Aware Contrastive Learning, _contrastive_loss 참고)을 더한다 —
    APS의 confident_mask 안에서 같은 pseudo-label끼리 feature를 가깝게, 배치 내 다른 샘플과는
    멀게 당기는 contrastive loss. 논문에서는 APS 위에 얹어 쓰는 구성이라(positive pair가
    confident set 기준) aps=False면 전체 배치를 confident로 취급해 사실상 일반 SupCon이 된다.

    balanced_cap=True면 _balanced_assign으로 pseudo-label 클러스터링 단계에서 클래스별 할당
    개수에 상한(balanced_cap_ratio × 공정 몫)을 둔다 — Bangladesh에서 관찰된 "한 클래스로
    쏠림"(Late_blight 쏠림이 Target_Spot 쏠림으로 옮겨간 것) 같은 폭주를, loss 재가중(class_balance)이
    아니라 pseudo-label 생성 자체에서 막는 접근. class_balance와 독립적으로 켤 수 있다.

    exclude_classes: oracle 진단(실제 정답으로 학습해도 recall이 낮은 클래스가 있는지 확인)에서
    구조적으로 어려운 것으로 확인된 클래스를 pseudo-labeling·학습 손실 양쪽에서 완전히 제외한다
    (eval 시 최종 마스킹에는 영향 없음 — run_one()이 별도로 적용하는 class_mask는 그대로 전체
    허용 클래스를 쓴다). 이 클래스들이 "끌어당기는 자리"를 없애, 남은 클래스끼리는 더 깨끗하게
    분리되길 기대하는 접근 — Bangladesh의 Target_Spot/Bacterial_spot처럼 오라클 recall이 낮은
    소수 클래스를 빼고 나머지 다수 클래스(지금 사례는 전체의 87%)에 집중한다.

    aad_weight > 0이면 AaD(apply_aad 참고)의 attract/disperse 항을 SHOT의 기존 loss(pseudo-label
    CE + IM)에 보조 항으로 더한다 — AaD를 통째로 대체하는 게 아니라, SHOT이 주는 분류 신호는
    유지한 채 "feature 공간 이웃끼리는 예측도 비슷해야 한다"는 제약으로 pseudo-label 노이즈를
    다듬어보려는 하이브리드. SHOT 단독(쏠림 문제)과 AaD 단독(신호 약함) 각각의 약점을 서로
    보완할 수 있을지 확인하는 실험적 조합이다."""
    allowed_idx = sorted({CLASS_NAMES.index(c) for c in EXTERNAL_PRESENT_CLASSES[dataset_key]})
    class_mask = class_mask_for(dataset_key, device)
    num_classes = len(CLASS_NAMES)

    exclude_idx = {CLASS_NAMES.index(c) for c in exclude_classes}
    pseudo_allowed_idx = [i for i in allowed_idx if i not in exclude_idx]
    if exclude_idx:
        pseudo_class_mask = class_mask.clone()
        pseudo_class_mask[list(exclude_idx)] = float("-inf")
    else:
        pseudo_class_mask = class_mask

    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.features.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(model.features.parameters(), lr=lr)

    for _ in range(epochs):
        pseudo_labels, pl_feats = _shot_pseudo_labels(
            model, target_loader, device, pseudo_allowed_idx, pseudo_class_mask, num_classes, return_feats=True,
            balanced_cap_ratio=balanced_cap_ratio if balanced_cap else None,
        )
        if aps:
            pseudo_labels, confident_mask = _aps_refine(
                pl_feats, pseudo_labels, pseudo_allowed_idx, num_classes, k=aps_k, iters=aps_iters, gamma=aps_gamma,
            )
        else:
            confident_mask = torch.ones_like(pseudo_labels, dtype=torch.bool)

        if class_balance:
            counts = torch.bincount(
                pseudo_labels[confident_mask], minlength=num_classes
            ).float().clamp(min=1)
            ce_weight = torch.zeros(num_classes)
            ce_weight[pseudo_allowed_idx] = 1.0 / counts[pseudo_allowed_idx]
            ce_weight[pseudo_allowed_idx] *= len(pseudo_allowed_idx) / ce_weight[pseudo_allowed_idx].sum()  # 평균 1이 되게 정규화
            ce_weight = ce_weight.to(device)
        else:
            ce_weight = None

        if aad_weight > 0:
            with torch.no_grad():
                aad_probs_list = []
                for images, _ in target_loader:
                    images = images.to(device)
                    f = model.head.pool(model.features(images)).flatten(1)
                    lg = model.head.fc(f) + pseudo_class_mask
                    aad_probs_list.append(lg.softmax(dim=1))
                aad_bank_probs = torch.cat(aad_probs_list)
            n = pl_feats.size(0)
            aad_k_eff = min(aad_k, n - 1)
            aad_feats_n = F.normalize(pl_feats.to(device), dim=1)
            aad_sims = aad_feats_n @ aad_feats_n.T
            aad_sims.fill_diagonal_(-2.0)
            aad_topk_sim, aad_topk_idx = aad_sims.topk(aad_k_eff, dim=1)
            aad_topk_w = aad_topk_sim.softmax(dim=1)
            aad_neighbor_mask = torch.zeros(n, n, dtype=torch.bool, device=device)
            aad_neighbor_mask.scatter_(1, aad_topk_idx, torch.ones_like(aad_topk_idx, dtype=torch.bool))
            aad_non_neighbor_mask = ~aad_neighbor_mask
            aad_non_neighbor_mask.fill_diagonal_(False)
            aad_non_neighbor_counts = aad_non_neighbor_mask.sum(dim=1).clamp(min=1).float()

        model.train()
        idx = 0
        for images, _ in target_loader:
            images = images.to(device)
            bsz = images.size(0)
            batch_pseudo = pseudo_labels[idx: idx + bsz].to(device)
            batch_confident = confident_mask[idx: idx + bsz].to(device)
            batch_idx = torch.arange(idx, idx + bsz, device=device)
            idx += bsz

            optimizer.zero_grad()
            feats = model.head.pool(model.features(images)).flatten(1)
            logits = model.head.fc(feats) + pseudo_class_mask
            probs = logits.softmax(dim=1)

            im_loss = _entropy_of(probs).mean() - div_weight * _entropy_of(probs.mean(dim=0))
            if batch_confident.any():
                cls_loss = F.cross_entropy(
                    logits[batch_confident], batch_pseudo[batch_confident], weight=ce_weight,
                )
            else:
                cls_loss = images.new_zeros(())
            cl_loss = _contrastive_loss(feats, batch_pseudo, batch_confident, cacl_temperature) if cacl else 0.0
            loss = cl_loss + cls_loss_weight * cls_loss + im_loss  # UPA 식 10: L_CL + beta*L_CE + L_IM

            if aad_weight > 0:
                batch_topk_idx = aad_topk_idx[batch_idx]
                batch_topk_w = aad_topk_w[batch_idx]
                neighbor_probs = aad_bank_probs[batch_topk_idx]  # (bsz, k, C)
                dot_ij = (probs.unsqueeze(1) * neighbor_probs).sum(dim=2)
                attract = (batch_topk_w * dot_ij).sum(dim=1)
                non_neigh_mask_batch = aad_non_neighbor_mask[batch_idx].float()
                non_neigh_counts_batch = aad_non_neighbor_counts[batch_idx]
                disperse = (probs @ aad_bank_probs.T) * non_neigh_mask_batch
                disperse = disperse.sum(dim=1) / non_neigh_counts_batch
                loss = loss + aad_weight * (-attract + disperse).mean()

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


def apply_rgv(
    model: nn.Module,
    target_loader,
    device,
    dataset_key: str,
    rounds: int = 3,
    lr: float = 1e-3,
    cls_loss_weight: float = 0.3,
    align_weight: float = 0.3,
    div_weight: float = 1.0,
    start_ratio: float = 0.3,
    end_ratio: float = 0.9,
) -> nn.Module:
    """R-GV — Zhu et al., *Revisiting Source-Free Domain Adaptation: Insights into
    Representativeness, Generalization, and Variety*, CVPR 2025의 "영감을 받은(inspired)" 버전.

    **중요한 제약**: 이 논문은 공식 코드가 공개돼 있지 않고, 본문 PDF도 접근이 막혀 있어서
    (CVF/ResearchGate 403) 공식 CVPR 포스터에 나온 수준의 정보만으로 구현했다 — 정확한
    sample selection/semantic alignment 수식은 논문 본문에만 있어 확인 못 했다. 즉 이건
    "R-GV 논문의 충실한 재현"이 아니라 "포스터에 공개된 3단계 구조(Representativeness/
    Generalization/Variety)를 우리 식으로 구현한 근사판"이다 — 보고서에도 이렇게 명시해야 한다.

    포스터에서 확인한 구조:
    1) Sample Selection (대표성+다양성): 매 라운드 pseudo-label을 다시 구하고, confidence
       상위 ratio%를 신뢰 가능한 집합(D_t,l)으로 고른다. ratio는 start_ratio→end_ratio로
       라운드가 지날수록 커진다(Variety — 처음엔 확실한 것만, 점점 더 많은 샘플을 끌어들임).
       SHOT의 masking처럼 클래스별로 선별해서, 샘플 수가 적은 클래스도(Taiwan의 healthy처럼)
       최소 1개는 포함되게 한다 — 이게 "분포 커버리지"에 해당.
    2) Semantic Alignment (일반화): D_t,l과 D_t,u(나머지) 사이의 feature 평균 거리를
       최소화한다(배치 단위 MMD 축소판) — source 데이터 없이 target 내부에서만 정렬한다는
       점에서 완전히 source-free다.
    3) Progressive: 위 두 단계를 여러 라운드 반복하며 신뢰 집합을 키워간다.
    """
    allowed_idx = sorted({CLASS_NAMES.index(c) for c in EXTERNAL_PRESENT_CLASSES[dataset_key]})
    class_mask = class_mask_for(dataset_key, device)
    num_classes = len(CLASS_NAMES)

    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.features.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(model.features.parameters(), lr=lr)

    for r in range(rounds):
        ratio = start_ratio + (end_ratio - start_ratio) * (r / max(rounds - 1, 1))
        pseudo_labels, pl_feats = _shot_pseudo_labels(
            model, target_loader, device, allowed_idx, class_mask, num_classes, return_feats=True
        )
        with torch.no_grad():
            pl_logits = model.head.fc(pl_feats.to(device)) + class_mask
            pl_conf = pl_logits.softmax(dim=1).gather(1, pseudo_labels.to(device).unsqueeze(1)).squeeze(1).cpu()

        confident_mask = torch.zeros_like(pseudo_labels, dtype=torch.bool)
        for c in allowed_idx:
            class_idx = (pseudo_labels == c).nonzero(as_tuple=True)[0]
            if len(class_idx) == 0:
                continue
            n_keep = max(1, int(round(len(class_idx) * ratio)))  # 작은 클래스도 최소 1개는 포함(Variety)
            top_local = pl_conf[class_idx].topk(min(n_keep, len(class_idx))).indices
            confident_mask[class_idx[top_local]] = True

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
                cls_loss = F.cross_entropy(logits[batch_confident], batch_pseudo[batch_confident])
            else:
                cls_loss = images.new_zeros(())
            if batch_confident.any() and (~batch_confident).any():
                align_loss = (feats[batch_confident].mean(dim=0) - feats[~batch_confident].mean(dim=0)).pow(2).sum()
            else:
                align_loss = images.new_zeros(())

            loss = im_loss + cls_loss_weight * cls_loss + align_weight * align_loss
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def apply_aad(
    model: nn.Module,
    target_loader,
    device,
    dataset_key: str,
    epochs: int = 2,
    lr: float = 1e-3,
    k: int = 4,
    disperse_weight: float = 1.0,
) -> nn.Module:
    """AaD (Attracting and Dispersing) — Yang et al., *Attracting and Dispersing: A Simple
    Approach for Source-Free Domain Adaptation*, NeurIPS 2022, 공식 코드
    github.com/Albert0147/AaD_SFDA. SHOT/EATA/R-GV와 근본적으로 다른 접근 — pseudo-label을
    아예 만들지 않는다. feature 공간에서 k-최근접 이웃끼리는 softmax 예측이 서로 가깝도록
    당기고(attract), 나머지 전체 샘플과는 예측이 멀어지도록(disperse) 만드는 순수 output-space
    제약만으로 적응한다. "이 샘플이 어떤 클래스냐"를 확정하는 클러스터링 단계가 없어서, 지금까지
    Bangladesh에서 반복 관찰된 "한 클래스로 쏠림"(Target_Spot이 실제보다 훨씬 많이 할당되는 것)
    문제가 구조적으로 생기지 않을 수 있다는 기대로 추가했다.

    공식 구현은 전체 target feature/예측을 momentum으로 갱신하는 memory bank를 쓰지만, 이
    프로젝트 데이터셋은 300~1,100장으로 작아서 매 epoch 시작 시 전체를 다시 계산하는 것으로
    단순화했다(SHOT의 매 epoch 재클러스터링과 같은 패턴) — EATA/R-GV처럼 "포스터/공식 코드
    규모(ImageNet/Office-Home급)를 작은 데이터셋에 맞춰 축소 적용한 근사판"이다.

    attract: sum_{j in kNN(i)} softmax(sim_ij) * <p_i, p_j>  (최대화)
    disperse: mean_{j notin kNN(i), j != i} <p_i, p_j>  (최소화, 예측 collapse 방지)
    loss = -attract + disperse_weight * disperse"""
    class_mask = class_mask_for(dataset_key, device)

    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.features.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(model.features.parameters(), lr=lr)

    for _ in range(epochs):
        model.eval()
        feats_list, probs_list = [], []
        with torch.no_grad():
            for images, _ in target_loader:
                images = images.to(device)
                f = model.head.pool(model.features(images)).flatten(1)
                lg = model.head.fc(f) + class_mask
                probs_list.append(lg.softmax(dim=1))
                feats_list.append(f)
        bank_feats = torch.cat(feats_list)
        bank_probs = torch.cat(probs_list)
        n = bank_feats.size(0)
        k_eff = min(k, n - 1)

        feats_n = F.normalize(bank_feats, dim=1)
        sims = feats_n @ feats_n.T
        sims.fill_diagonal_(-2.0)
        topk_sim, topk_idx = sims.topk(k_eff, dim=1)
        topk_w = topk_sim.softmax(dim=1)  # 이웃별 가중치(유사도 softmax)

        neighbor_mask = torch.zeros(n, n, dtype=torch.bool, device=device)
        neighbor_mask.scatter_(1, topk_idx, torch.ones_like(topk_idx, dtype=torch.bool))
        non_neighbor_mask = ~neighbor_mask
        non_neighbor_mask.fill_diagonal_(False)
        non_neighbor_counts = non_neighbor_mask.sum(dim=1).clamp(min=1).float()

        model.train()
        idx = 0
        for images, _ in target_loader:
            images = images.to(device)
            bsz = images.size(0)
            batch_idx = torch.arange(idx, idx + bsz, device=device)
            idx += bsz

            optimizer.zero_grad()
            feats = model.head.pool(model.features(images)).flatten(1)
            logits = model.head.fc(feats) + class_mask
            probs = logits.softmax(dim=1)

            batch_topk_idx = topk_idx[batch_idx]
            batch_topk_w = topk_w[batch_idx]
            neighbor_probs = bank_probs[batch_topk_idx]  # (bsz, k, C)
            dot_ij = (probs.unsqueeze(1) * neighbor_probs).sum(dim=2)  # (bsz, k) = <p_i, p_j>
            attract = (batch_topk_w * dot_ij).sum(dim=1)

            non_neigh_mask_batch = non_neighbor_mask[batch_idx].float()
            non_neigh_counts_batch = non_neighbor_counts[batch_idx]
            disperse = (probs @ bank_probs.T) * non_neigh_mask_batch
            disperse = disperse.sum(dim=1) / non_neigh_counts_batch

            loss = (-attract + disperse_weight * disperse).mean()
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def apply_cowa(
    model: nn.Module,
    target_loader,
    device,
    dataset_key: str,
    epochs: int = 2,
    lr: float = 1e-3,
    cls_loss_weight: float = 0.3,
    div_weight: float = 1.0,
    model_weight_alpha: float = 0.5,
) -> nn.Module:
    """CoWA-JMDS — Lee et al., *Confidence Score for Source-Free Unsupervised Domain
    Adaptation*, ICML 2022, 공식 코드 github.com/Jhyun17/CoWA-JMDS의 핵심 아이디어("Joint
    Model-Data Score")를 축소 적용한다. APS처럼 신뢰도 상위 gamma%만 쓰고 나머지는 완전히
    버리는 하드 필터링과 달리, 모든 샘플에 **연속적인 soft weight**를 매긴다 — 지금까지
    시도한 APS(hard filtering)·balanced_cap(할당 개수 제한)·AaD(kNN attract/disperse) 전부
    "이분법적으로 끊어내는" 접근이었는데, CoWA는 "얼마나 믿을지"를 점수로 매겨 전부 쓰는
    세 번째 메커니즘이다.

    data score: SHOT의 weighted k-means 결과(클래스 중심까지의 코사인 거리)를 클래스별
    1-성분 가우시안으로 보고, 각 샘플이 자기 pseudo-label 클래스에 얼마나 "전형적인지"를
    밀도로 추정한다(공식 구현은 전체 분포에 GMM을 적합하지만, 이 프로젝트 규모(300~1,100장)에
    맞춰 이미 계산해둔 클래스 중심·코사인 거리로 근사했다).
    model score: source 분류기 자신이 그 pseudo-label에 준 softmax confidence.
    JMDS = model_score^alpha * data_score^(1-alpha) — 두 점수를 곱해 최종 가중치로 쓰고,
    class_balance처럼 평균 1로 정규화해 cross-entropy에 곱한다."""
    allowed_idx = sorted({CLASS_NAMES.index(c) for c in EXTERNAL_PRESENT_CLASSES[dataset_key]})
    class_mask = class_mask_for(dataset_key, device)
    num_classes = len(CLASS_NAMES)
    disallowed = [i for i in range(num_classes) if i not in allowed_idx]

    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.features.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(model.features.parameters(), lr=lr)

    for _ in range(epochs):
        model.eval()
        feats_list, probs_list = [], []
        with torch.no_grad():
            for images, _ in target_loader:
                images = images.to(device)
                f = model.head.pool(model.features(images)).flatten(1)
                lg = model.head.fc(f) + class_mask
                probs_list.append(lg.softmax(dim=1))
                feats_list.append(f)
        feats = torch.cat(feats_list)
        probs = torch.cat(probs_list)

        def assign_with_sims(weight: torch.Tensor):
            centroids = (weight.T @ feats) / (weight.sum(dim=0).unsqueeze(1) + 1e-8)
            feats_n = F.normalize(feats, dim=1)
            centroids_n = F.normalize(centroids, dim=1)
            sims = feats_n @ centroids_n.T
            if disallowed:
                sims[:, disallowed] = -1e9
            return sims.argmax(dim=1), sims

        pseudo, _ = assign_with_sims(probs)
        pseudo, sims = assign_with_sims(F.one_hot(pseudo, num_classes).float())

        own_sim = sims.gather(1, pseudo.unsqueeze(1)).squeeze(1)  # 자기 클래스 중심까지의 코사인 유사도
        data_score = torch.ones_like(own_sim)
        for c in allowed_idx:
            idx_c = (pseudo == c).nonzero(as_tuple=True)[0]
            if idx_c.numel() < 2:
                continue
            mean_c = own_sim[idx_c].mean()
            var_c = own_sim[idx_c].var(unbiased=False).clamp(min=1e-4)
            data_score[idx_c] = torch.exp(-((own_sim[idx_c] - mean_c) ** 2) / (2 * var_c))

        model_score = probs.gather(1, pseudo.unsqueeze(1)).squeeze(1)
        jmds = (model_score.clamp(min=1e-6) ** model_weight_alpha) * (data_score.clamp(min=1e-6) ** (1 - model_weight_alpha))
        ce_weight = jmds / jmds.mean().clamp(min=1e-8)  # 평균 1로 정규화 (class_balance와 같은 패턴)

        model.train()
        idx = 0
        for images, _ in target_loader:
            images = images.to(device)
            bsz = images.size(0)
            batch_pseudo = pseudo[idx: idx + bsz].to(device)
            batch_weight = ce_weight[idx: idx + bsz].to(device)
            idx += bsz

            optimizer.zero_grad()
            batch_feats = model.head.pool(model.features(images)).flatten(1)
            logits = model.head.fc(batch_feats) + class_mask
            batch_probs = logits.softmax(dim=1)

            im_loss = _entropy_of(batch_probs).mean() - div_weight * _entropy_of(batch_probs.mean(dim=0))
            per_sample_ce = F.cross_entropy(logits, batch_pseudo, reduction="none")
            cls_loss = (batch_weight * per_sample_ce).mean()
            loss = im_loss + cls_loss_weight * cls_loss
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
    shot_cacl: bool = False,
    shot_cacl_temperature: float = 0.1,
    shot_balanced_cap: bool = False,
    shot_balanced_cap_ratio: float = 1.5,
    shot_exclude_classes: tuple[str, ...] = (),
    shot_aad_weight: float = 0.0,
    shot_aad_k: int = 4,
    eata_lr: float = 0.1,
    eata_e0_factor: float = 0.4,
    eata_d0: float = 0.05,
    eata_fisher_alpha: float = 1.0,
    rgv_rounds: int = 3,
    rgv_lr: float = 1e-3,
    rgv_cls_weight: float = 0.3,
    rgv_align_weight: float = 0.3,
    rgv_start_ratio: float = 0.3,
    rgv_end_ratio: float = 0.9,
    aad_epochs: int = 2,
    aad_lr: float = 1e-3,
    aad_k: int = 4,
    aad_disperse_weight: float = 1.0,
    cowa_epochs: int = 2,
    cowa_lr: float = 1e-3,
    cowa_cls_weight: float = 0.3,
    cowa_alpha: float = 0.5,
) -> dict:
    """masked=True면 class_mask_for()로 target에 없는 클래스를 가린다.
    TENT는 적응 단계(entropy 손실)에도 마스크를 넣어 mask-aware하게 적응하고,
    AdaBN은 통계 재계산 자체가 loss-free라 적응 단계는 그대로 두고 최종 평가에만 마스크를 적용한다.
    SHOT·EATA·R-GV·AaD·CoWA는 마스킹이 방법 자체에 내장돼 있어 masked 인자와 무관하게 항상 적용된다.

    div_weight: TENT 전용, >0이면 diversity 정규화(collapse 방지)를 추가한다.
    adabn_blend_momentum: AdaBN 전용, 주면 완전 교체 대신 해당 momentum으로 source/target 통계를 블렌딩한다."""
    set_seed(seed)
    model = load_base_model(model_name, base_run_name, seed, img_size, device)
    loader = make_external_dataloader(data_dir, img_size, batch_size, interpolation=INTERPOLATION_MODES[interp])
    masked = masked or method in ("shot", "eata", "rgv", "aad", "cowa")
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
            class_balance=shot_class_balance, cacl=shot_cacl, cacl_temperature=shot_cacl_temperature,
            balanced_cap=shot_balanced_cap, balanced_cap_ratio=shot_balanced_cap_ratio,
            exclude_classes=shot_exclude_classes, aad_weight=shot_aad_weight, aad_k=shot_aad_k,
        )
    elif method == "eata":
        fisher, anchor = _compute_fisher(model_name, base_run_name, seed, img_size, device)
        model = apply_eata(
            model, loader, device, fisher, anchor, lr=eata_lr, e0_factor=eata_e0_factor, d0=eata_d0,
            fisher_alpha=eata_fisher_alpha, class_mask=class_mask,
        )
    elif method == "rgv":
        model = apply_rgv(
            model, loader, device, dataset_key, rounds=rgv_rounds, lr=rgv_lr, cls_loss_weight=rgv_cls_weight,
            align_weight=rgv_align_weight, start_ratio=rgv_start_ratio, end_ratio=rgv_end_ratio,
        )
    elif method == "aad":
        model = apply_aad(
            model, loader, device, dataset_key, epochs=aad_epochs, lr=aad_lr, k=aad_k,
            disperse_weight=aad_disperse_weight,
        )
    elif method == "cowa":
        model = apply_cowa(
            model, loader, device, dataset_key, epochs=cowa_epochs, lr=cowa_lr,
            cls_loss_weight=cowa_cls_weight, model_weight_alpha=cowa_alpha,
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
    if method == "shot" and shot_cacl:
        run_name += "_cacl"
    if method == "shot" and shot_balanced_cap:
        run_name += f"_bcap{shot_balanced_cap_ratio:g}"
    if method == "shot" and shot_exclude_classes:
        run_name += f"_excl{len(shot_exclude_classes)}"
    if method == "shot" and shot_aad_weight > 0:
        run_name += f"_aadw{shot_aad_weight:g}_aadk{shot_aad_k}"
    if method == "eata" and (eata_e0_factor != 0.4 or eata_d0 != 0.05 or eata_fisher_alpha != 1.0):
        run_name += f"_e0{eata_e0_factor:g}_d0{eata_d0:g}_fa{eata_fisher_alpha:g}"
    if method == "rgv" and (rgv_rounds != 3 or rgv_align_weight != 0.3):
        run_name += f"_r{rgv_rounds}_aw{rgv_align_weight:g}"
    if method == "aad" and (aad_epochs != 2 or aad_k != 4 or aad_disperse_weight != 1.0):
        run_name += f"_ep{aad_epochs}_k{aad_k}_dw{aad_disperse_weight:g}"
    if method == "cowa" and (cowa_epochs != 2 or cowa_cls_weight != 0.3 or cowa_alpha != 0.5):
        run_name += f"_ep{cowa_epochs}_cw{cowa_cls_weight:g}_a{cowa_alpha:g}"
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
    parser.add_argument(
        "--shot-cacl", action="store_true",
        help="SHOT 전용: UPA의 CACL(Class-Aware Contrastive Learning)을 추가한다 — APS confident "
        "set 안에서 같은 pseudo-label끼리 feature를 당기는 contrastive loss (aps와 같이 쓰는 걸 권장)",
    )
    parser.add_argument("--shot-cacl-temperature", type=float, default=0.1, help="CACL 전용: contrastive loss temperature (논문 기본값 0.1)")
    parser.add_argument(
        "--shot-balanced-cap", action="store_true",
        help="SHOT 전용: pseudo-label 클러스터링 단계에서 클래스별 할당 개수에 상한을 둬 "
        "한 클래스로의 쏠림(Bangladesh의 Target_Spot 쏠림 등)을 억제한다 (class_balance와 독립적)",
    )
    parser.add_argument(
        "--shot-balanced-cap-ratio", type=float, default=1.5,
        help="balanced-cap 전용: 상한 = ceil(샘플 수 / 허용 클래스 수 * 이 값) (기본값 1.5)",
    )
    parser.add_argument(
        "--shot-exclude-classes", type=str, default="",
        help="SHOT 전용: pseudo-labeling·학습 손실에서 완전히 제외할 클래스 이름(CLASS_NAMES 기준, "
        "쉼표로 구분). eval 시 마스킹에는 영향 없음 — oracle 진단에서 recall이 낮게 나온 구조적으로 "
        "어려운 클래스를 빼고 나머지 클래스끼리 더 깨끗하게 분리시키려는 용도",
    )
    parser.add_argument(
        "--shot-aad-weight", type=float, default=0.0,
        help="SHOT 전용: AaD(attract/disperse)를 보조 손실로 추가할 가중치(0이면 비활성, SHOT+AaD 하이브리드)",
    )
    parser.add_argument("--shot-aad-k", type=int, default=4, help="SHOT+AaD 하이브리드 전용: attract 대상 최근접 이웃 수")
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
    parser.add_argument(
        "--rgv-rounds", type=int, default=3,
        help="R-GV 전용(포스터 기반 근사 구현, 논문 본문 미확인): progressive 라운드 수",
    )
    parser.add_argument("--rgv-lr", type=float, default=1e-3)
    parser.add_argument("--rgv-cls-weight", type=float, default=0.3, help="R-GV 전용: confident set pseudo-label CE 가중치")
    parser.add_argument("--rgv-align-weight", type=float, default=0.3, help="R-GV 전용: D_t,l/D_t,u 간 semantic alignment 가중치")
    parser.add_argument("--rgv-start-ratio", type=float, default=0.3, help="R-GV 전용: 1라운드에서 confident로 고르는 클래스별 비율")
    parser.add_argument("--rgv-end-ratio", type=float, default=0.9, help="R-GV 전용: 마지막 라운드에서 confident로 고르는 클래스별 비율")
    parser.add_argument("--aad-epochs", type=int, default=2, help="AaD 전용(공식 코드 momentum memory bank를 매 epoch 재계산으로 축소 적용): 적응 epoch 수")
    parser.add_argument("--aad-lr", type=float, default=1e-3)
    parser.add_argument("--aad-k", type=int, default=4, help="AaD 전용: attract 대상 최근접 이웃 수")
    parser.add_argument("--aad-disperse-weight", type=float, default=1.0, help="AaD 전용: disperse(비-이웃과 멀어지기) 항 가중치")
    parser.add_argument("--cowa-epochs", type=int, default=2, help="CoWA 전용: 적응 epoch 수")
    parser.add_argument("--cowa-lr", type=float, default=1e-3)
    parser.add_argument("--cowa-cls-weight", type=float, default=0.3, help="CoWA 전용: JMDS 가중 cross-entropy 항의 가중치")
    parser.add_argument("--cowa-alpha", type=float, default=0.5, help="CoWA 전용: JMDS = model_score^alpha * data_score^(1-alpha)")
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
                    shot_cacl=args.shot_cacl, shot_cacl_temperature=args.shot_cacl_temperature,
                    shot_balanced_cap=args.shot_balanced_cap, shot_balanced_cap_ratio=args.shot_balanced_cap_ratio,
                    shot_exclude_classes=tuple(c for c in args.shot_exclude_classes.split(",") if c),
                    shot_aad_weight=args.shot_aad_weight, shot_aad_k=args.shot_aad_k,
                    eata_lr=args.eata_lr, eata_e0_factor=args.eata_e0_factor, eata_d0=args.eata_d0,
                    eata_fisher_alpha=args.eata_fisher_alpha,
                    rgv_rounds=args.rgv_rounds, rgv_lr=args.rgv_lr, rgv_cls_weight=args.rgv_cls_weight,
                    rgv_align_weight=args.rgv_align_weight, rgv_start_ratio=args.rgv_start_ratio,
                    rgv_end_ratio=args.rgv_end_ratio,
                    aad_epochs=args.aad_epochs, aad_lr=args.aad_lr, aad_k=args.aad_k,
                    aad_disperse_weight=args.aad_disperse_weight,
                    cowa_epochs=args.cowa_epochs, cowa_lr=args.cowa_lr, cowa_cls_weight=args.cowa_cls_weight,
                    cowa_alpha=args.cowa_alpha,
                )
            )

    print("\n=== 요약 ===")
    for r in results:
        print(f"{r['dataset']:16s} {r['run_name']:32s} acc={r['accuracy']:.4f}  macro_f1={r['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
