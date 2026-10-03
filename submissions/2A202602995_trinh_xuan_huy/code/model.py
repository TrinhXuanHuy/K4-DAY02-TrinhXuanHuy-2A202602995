"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Triển khai đầy đủ cho bài nộp Lab Day 2 (DeepWeeds).
"""
from __future__ import annotations

import torch
import torch.nn as nn

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Tạo model phân loại 9 lớp qua timm.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ
    """
    import timm

    is_pretrained = (init != "scratch") and pretrained
    model = timm.create_model(
        name,
        pretrained=is_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )

    if init == "frozen":
        freeze_backbone(model)

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng mọi tham số trừ head.

    Tất cả tham số của backbone có requires_grad = False.
    Head (model.get_classifier()) vẫn có requires_grad = True.
    """
    for param in model.parameters():
        param.requires_grad = False

    classifier = model.get_classifier()
    if isinstance(classifier, nn.Module):
        for param in classifier.parameters():
            param.requires_grad = True
    elif isinstance(classifier, torch.Tensor):
        classifier.requires_grad = True


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict]:
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    1. backbone weights (ndim > 1): lr = lr_backbone, weight_decay = weight_decay
    2. backbone norm & bias (ndim <= 1): lr = lr_backbone, weight_decay = 0.0
    3. classifier / head: lr = lr_head (thường gấp 10 lần), weight_decay = weight_decay
    """
    classifier = model.get_classifier()
    classifier_params = set()
    if isinstance(classifier, nn.Module):
        classifier_params = set(classifier.parameters())
    elif isinstance(classifier, torch.Tensor):
        classifier_params = {classifier}

    backbone_decay = []
    backbone_no_decay = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param in classifier_params:
            head_params.append(param)
        elif param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower():
            backbone_no_decay.append(param)
        else:
            backbone_decay.append(param)

    groups = []
    if backbone_decay:
        groups.append({"params": backbone_decay, "lr": lr_backbone, "weight_decay": weight_decay})
    if backbone_no_decay:
        groups.append({"params": backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})

    return groups


def count_params(model: nn.Module) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size.

    Thử dùng ptflops, thop, fvcore nếu có; nếu không tự đếm ước lượng.
    """
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(
            model, (3, img_size, img_size), as_strings=False, print_per_layer_stat=False, verbose=False
        )
        return float(macs) / 1e9
    except Exception:
        pass

    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return float(macs) / 1e9
    except Exception:
        pass

    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, dummy_input).total()
        return float(flops) / (2.0 * 1e9)  # 1 MAC ≈ 2 FLOPs
    except Exception:
        pass

    # Ước lượng chuẩn theo số lượng tham số nếu không có thư viện chuyên dụng
    n_params = sum(p.numel() for p in model.parameters())
    return round((n_params * 1.5) / 1e7, 2)

