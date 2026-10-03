"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Triển khai đầy đủ cho bài nộp Lab Day 2 (DeepWeeds).
Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar


def predict_logits(model: nn.Module, loader, device: torch.device, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file."""
    model.eval()
    all_filenames = []
    all_targets = []
    all_logits = []
    use_cuda = device.type == "cuda"

    with torch.inference_mode():
        for images, targets, filenames in loader:
            images = images.to(device, non_blocking=True)
            if view is not None:
                images = view(images)

            with torch.cuda.amp.autocast(enabled=use_cuda):
                outputs = model(images)

            all_filenames.extend(filenames)
            all_targets.append(targets.numpy() if isinstance(targets, torch.Tensor) else targets)
            all_logits.append(outputs.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    return all_filenames, y_true, logits


def view_identity(x: torch.Tensor) -> torch.Tensor:
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    """Lật ngang batch ảnh (N, C, H, W) trên chiều rộng W (slide trang 75)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x: torch.Tensor, crop: int = 224) -> list[torch.Tensor]:
    """5 crop: 4 góc và crop ở giữa."""
    _, _, h, w = x.shape
    crops = [
        x[:, :, :crop, :crop],                  # Top-left
        x[:, :, :crop, w - crop:],              # Top-right
        x[:, :, h - crop:, :crop],              # Bottom-left
        x[:, :, h - crop:, w - crop:],          # Bottom-right
        x[:, :, (h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2],  # Center
    ]
    return crops


def views_multiscale(x: torch.Tensor, sizes: list[int] = (224, 256, 288)) -> list[torch.Tensor]:
    """Resize batch về từng kích thước trong `sizes`."""
    return [F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False) for s in sizes]


def aggregate_views(logits_per_view: list[np.ndarray], space: str = "prob") -> np.ndarray:
    """Gộp K lượt chạy của TTA thành một dự đoán xác suất (N, 9).

    - space="prob": trung bình softmax của từng view
    - space="logit": trung bình logit rồi softmax
    """
    if space == "prob":
        probs_list = []
        for lg in logits_per_view:
            shift = lg - np.max(lg, axis=1, keepdims=True)
            exp_lg = np.exp(shift)
            probs = exp_lg / np.sum(exp_lg, axis=1, keepdims=True)
            probs_list.append(probs)
        avg_probs = np.mean(probs_list, axis=0)
        return avg_probs / np.sum(avg_probs, axis=1, keepdims=True)
    elif space == "logit":
        avg_logits = np.mean(logits_per_view, axis=0)
        shift = avg_logits - np.max(avg_logits, axis=1, keepdims=True)
        exp_lg = np.exp(shift)
        return exp_lg / np.sum(exp_lg, axis=1, keepdims=True)
    else:
        raise ValueError(f"Không nhận diện space='{space}' (chỉ chọn 'prob' hoặc 'logit')")


def ensemble_probs(list_of_probs: list[np.ndarray]) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed)."""
    avg = np.mean(list_of_probs, axis=0)
    return avg / np.sum(avg, axis=1, keepdims=True)


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T) (slide trang 69)."""
    y = np.asarray(val_labels, dtype=np.int64)
    logits = np.asarray(val_logits, dtype=np.float64)

    def nll_obj(t: float) -> float:
        scaled = logits / max(1e-4, t)
        shift = scaled - np.max(scaled, axis=1, keepdims=True)
        exp_s = np.exp(shift)
        probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        p_correct = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
        return float(-np.mean(np.log(p_correct)))

    res = minimize_scalar(nll_obj, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Trả về softmax(logits / T)."""
    scaled = np.asarray(logits, dtype=np.float64) / max(1e-4, T)
    shift = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_s = np.exp(shift)
    return exp_s / np.sum(exp_s, axis=1, keepdims=True)


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Gộp BatchNorm vào tích chập liền trước để tối ưu hóa suy luận (slide trang 71, 75)."""
    model = copy.deepcopy(model)
    model.eval()

    try:
        from torch.nn.utils.fusion import fuse_conv_bn_eval
        # Duyệt và gộp các cặp Conv2d + BatchNorm2d
        for name, module in model.named_children():
            children = list(module.named_children())
            for i in range(len(children) - 1):
                c_name, c_mod = children[i]
                b_name, b_mod = children[i + 1]
                if isinstance(c_mod, nn.Conv2d) and isinstance(b_mod, nn.BatchNorm2d):
                    fused = fuse_conv_bn_eval(c_mod, b_mod)
                    setattr(module, c_name, fused)
                    setattr(module, b_name, nn.Identity())
    except Exception as e:
        print(f"Lưu ý: Không thể gộp BN tự động ({e}), giữ nguyên mô hình.")

    return model

