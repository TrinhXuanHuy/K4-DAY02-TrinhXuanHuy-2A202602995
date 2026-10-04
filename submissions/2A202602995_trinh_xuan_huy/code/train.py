"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Triển khai đầy đủ cho bài nộp Lab Day 2 (DeepWeeds).
Dùng MỘT hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

# Hỗ trợ import dataset, model, losses, eval từ cùng thư mục hoặc root
CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(ROOT_DIR))

import dataset as ds
import losses as ls
import model as mdl
from eval import compute_metrics, save_predictions


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curves_dir: str = "curves"
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên để đảm bảo tính tái lập."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW với 3 nhóm tham số (backbone, norm/bias, head)."""
    groups = mdl.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0 theo bước lặp."""
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(1e-6, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


class EMA:
    """Exponential Moving Average của trọng số mô hình."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: param.data.clone().detach()
            for name, param in model.named_parameters() if param.requires_grad
        }
        self.backup = {}

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        self.backup = {
            name: param.data.clone().detach()
            for name, param in model.named_parameters() if param.requires_grad
        }
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, scheduler, scaler,
                    cfg: Config, device: torch.device, ema: EMA | None = None,
                    epoch: int = 1) -> dict[str, float]:
    """Huấn luyện 1 epoch với log tiến độ thời gian thực."""
    model.train()
    if cfg.init == "frozen":
        mdl.freeze_backbone(model)
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()

    total_loss = 0.0
    total_samples = 0
    use_cuda = device.type == "cuda"
    total_batches = len(loader)

    for batch_idx, (images, targets, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.amp and use_cuda):
            if cfg.mix:
                mixed_imgs, mixed_targets = ls.mix_batch(images, targets, cfg.mix_alpha, cfg.mix)
                outputs = model(mixed_imgs)
                loss = ls.mixed_loss(criterion, outputs, mixed_targets)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)

        if cfg.amp and use_cuda:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if (batch_idx + 1) % 30 == 0 or (batch_idx + 1) == total_batches:
            pct = 100.0 * (batch_idx + 1) / total_batches
            print(f"  [{cfg.exp_id}|Epoch {epoch:02d}/{cfg.epochs:02d}] Batch {batch_idx+1:03d}/{total_batches:03d} ({pct:3.0f}%) | Step Loss: {loss.item():.4f}", flush=True)

    avg_loss = total_loss / max(1, total_samples)
    current_lr = optimizer.param_groups[0]["lr"]
    return {"train_loss": avg_loss, "lr": current_lr}


def evaluate(model: nn.Module, loader, criterion, device: torch.device):
    """Đánh giá model trên một DataLoader, không tính gradient.

    Trả về (filenames, y_true, logits, loss).
    """
    model.eval()
    all_filenames = []
    all_targets = []
    all_logits = []
    total_loss = 0.0
    total_samples = 0
    use_cuda = device.type == "cuda"

    with torch.inference_mode():
        for images, targets, filenames in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = images.size(0)

            with autocast(enabled=use_cuda):
                outputs = model(images)
                loss = criterion(outputs, targets)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_filenames.extend(filenames)
            all_targets.append(targets.cpu().numpy())
            all_logits.append(outputs.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    avg_loss = total_loss / max(1, total_samples)
    return all_filenames, y_true, logits, avg_loss


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ và lưu biểu đồ loss train/val và macro-F1 val theo epoch."""
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    val_macro_f1s = [h["val_macro_f1"] for h in history]
    val_top1s = [h["val_top1"] for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color)
    p1 = ax1.plot(epochs, train_losses, "r--o", label="Train Loss")
    p2 = ax1.plot(epochs, val_losses, "r-s", label="Val Loss")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Metric (Val)", color=color)
    p3 = ax2.plot(epochs, val_macro_f1s, "b-^", label="Val Macro-F1")
    p4 = ax2.plot(epochs, val_top1s, "g--v", label="Val Top-1 Acc")
    ax2.tick_params(axis="y", labelcolor=color)

    lines = p1 + p2 + p3 + p4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right")

    plt.title(title)
    fig.tight_layout()

    out_file = Path(path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150)
    plt.close(fig)


def run(cfg: Config) -> dict[str, Any]:
    """Quy trình huấn luyện hoàn chỉnh một cấu hình."""
    start_time = time.time()
    set_seed(cfg.seed)

    save_dir = run_dir(cfg)
    save_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.curves_dir).mkdir(parents=True, exist_ok=True)

    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print(f"[{cfg.exp_id}|seed{cfg.seed}] ⚠️ CẢNH BÁO: ĐANG CHẠY TRÊN CPU! Tốc độ sẽ rất chậm (~12-15 phút/epoch). Trên Colab, vui lòng chọn menu: 'Thời gian chạy' (Runtime) -> 'Thay đổi loại thời gian chạy' -> Chọn 'T4 GPU'!", flush=True)
    else:
        print(f"[{cfg.exp_id}|seed{cfg.seed}] Bắt đầu huấn luyện trên GPU: {torch.cuda.get_device_name(0)} (AMP: {cfg.amp})", flush=True)

    # Tự động ưu tiên thư mục ảnh cục bộ /content/data/images (nếu có) để tăng tốc 100 lần và tránh treo ổ Google Drive
    effective_images_dir = cfg.images_dir
    local_images = Path("/content/data/images")
    if local_images.exists() and any(local_images.iterdir()):
        effective_images_dir = str(local_images)
        print(f"[{cfg.exp_id}|seed{cfg.seed}] Đang đọc ảnh từ SSD cục bộ Colab ({effective_images_dir}) - Tốc độ cực nhanh!", flush=True)
    else:
        # Nếu đang đọc trực tiếp từ Google Drive, tự đặt num_workers = 0 để tránh nghẽn/treo tiến trình DataLoader
        if "/content/drive" in str(Path(effective_images_dir).resolve()):
            cfg.num_workers = 0
            print(f"[{cfg.exp_id}|seed{cfg.seed}] Đang đọc ảnh từ Google Drive (đặt num_workers=0 để chống nghẽn/treo ổ đĩa)!", flush=True)

    # 1. Đọc và kiểm tra split
    train_df, val_df, test_df = ds.load_split(cfg.labels_dir, cfg.fold)
    ds.check_split(train_df, val_df, test_df, effective_images_dir)

    # 2. Tạo DataLoaders
    train_tf = ds.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    eval_tf = ds.build_transforms(train=False, img_size=cfg.img_size)

    train_loader = ds.make_loader(
        train_df, effective_images_dir, train_tf, cfg.batch_size, train=True,
        sampler=cfg.sampler, num_workers=cfg.num_workers
    )
    val_loader = ds.make_loader(
        val_df, effective_images_dir, eval_tf, cfg.batch_size, train=False,
        num_workers=cfg.num_workers
    )

    # 3. Tạo Model
    model = mdl.build_model(
        cfg.backbone, pretrained=True, num_classes=ds.NUM_CLASSES,
        drop_rate=cfg.drop_rate, init=cfg.init
    ).to(device)

    n_params = mdl.count_params(model)
    gmacs = mdl.count_gmacs(model, cfg.img_size)

    # 4. Criterion, Optimizer, Scheduler, EMA
    criterion_kwargs = {}
    if cfg.loss == "ls":
        criterion_kwargs["smoothing"] = cfg.label_smoothing
    elif cfg.loss == "focal":
        criterion_kwargs["gamma"] = cfg.focal_gamma
    elif cfg.loss == "ce_weighted":
        weights = ls.class_weights(train_df["Label"].value_counts().to_dict(), beta=cfg.class_weight_beta or 0.0)
        criterion_kwargs["weight"] = weights.to(device)

    criterion = ls.build_criterion(cfg.loss, **criterion_kwargs)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = GradScaler(enabled=cfg.amp and device.type == "cuda")

    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay is not None else None

    # 5. Huấn luyện từng Epoch & Cơ chế Checkpoint Phục hồi khi Sập máy
    best_macro_f1 = -1.0
    best_epoch = 0
    history = []
    checkpoint_path = save_dir / "best_model.pt"
    last_checkpoint_path = save_dir / "last_checkpoint.pt"
    history_csv_path = save_dir / "history.csv"

    # Kiểm tra 1: Nếu thí nghiệm này đã chạy đủ epochs từ trước, bỏ qua để tiết kiệm thời gian!
    if history_csv_path.exists() and checkpoint_path.exists():
        try:
            prev_hist = pd.read_csv(history_csv_path)
            if len(prev_hist) >= cfg.epochs:
                print(f"[{cfg.exp_id}|seed{cfg.seed}] ĐÃ HOÀN THÀNH {len(prev_hist)}/{cfg.epochs} epochs từ trước. Tự động nạp kết quả và bỏ qua train lại!")
                ckpt = torch.load(checkpoint_path, map_location=device)
                result = {
                    "exp_id": cfg.exp_id,
                    "seed": cfg.seed,
                    "backbone": cfg.backbone,
                    "best_epoch": ckpt.get("epoch", cfg.epochs),
                    "best_val_macro_f1": ckpt.get("macro_f1", 0.0),
                    "best_val_top1": ckpt.get("top1", 0.0),
                    "params_m": n_params,
                    "gmacs": gmacs,
                    "avg_epoch_time_s": prev_hist["time_s"].mean() if "time_s" in prev_hist else 0.0,
                    "total_time_s": prev_hist["time_s"].sum() if "time_s" in prev_hist else 0.0,
                }
                return result
        except Exception:
            pass

    # Kiểm tra 2: Nếu có checkpoint của epoch gần nhất (do sập máy giữa chừng), nạp lại và chạy tiếp!
    start_epoch = 1
    if last_checkpoint_path.exists():
        try:
            last_ckpt = torch.load(last_checkpoint_path, map_location=device)
            model.load_state_dict(last_ckpt["model_state_dict"])
            optimizer.load_state_dict(last_ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(last_ckpt["scheduler_state_dict"])
            if cfg.amp and device.type == "cuda" and "scaler_state_dict" in last_ckpt:
                scaler.load_state_dict(last_ckpt["scaler_state_dict"])
            if ema is not None and last_ckpt.get("ema_shadow"):
                ema.shadow = {k: v.to(device) for k, v in last_ckpt["ema_shadow"].items()}
            best_macro_f1 = last_ckpt.get("best_macro_f1", -1.0)
            best_epoch = last_ckpt.get("best_epoch", 0)
            history = last_ckpt.get("history", [])
            start_epoch = last_ckpt["epoch"] + 1
            print(f"[{cfg.exp_id}|seed{cfg.seed}] >>> PHÁT HIỆN SẬP NGUỒN! Đang phục hồi từ Epoch {start_epoch}/{cfg.epochs}...")
        except Exception as e:
            print(f"[{cfg.exp_id}|seed{cfg.seed}] Không thể phục hồi checkpoint cũ ({e}), bắt đầu lại từ Epoch 1.")
            start_epoch = 1

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        train_res = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema,
            epoch=epoch
        )

        if ema is not None:
            ema.copy_to(model)

        val_names, val_targets, val_logits, val_loss = evaluate(model, val_loader, criterion, device)
        val_probs = np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True))
        val_probs = val_probs / np.sum(val_probs, axis=1, keepdims=True)
        val_metrics = compute_metrics(val_targets, val_probs.argmax(1), val_probs)

        if ema is not None:
            ema.restore(model)

        epoch_time = time.time() - t0
        macro_f1 = val_metrics["macro_f1"]
        top1 = val_metrics["top1"]

        history_item = {
            "epoch": epoch,
            "train_loss": train_res["train_loss"],
            "val_loss": val_loss,
            "val_macro_f1": macro_f1,
            "val_top1": top1,
            "lr": train_res["lr"],
            "time_s": epoch_time,
        }
        history.append(history_item)

        print(f"[{cfg.exp_id}|seed{cfg.seed}] Epoch {epoch}/{cfg.epochs} - "
              f"Train Loss: {train_res['train_loss']:.4f} - Val Loss: {val_loss:.4f} - "
              f"Val Macro-F1: {macro_f1:.4f} - Val Top-1: {top1:.4f} ({epoch_time:.1f}s)", flush=True)

        # Lưu best model theo Macro-F1 Val
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "macro_f1": macro_f1,
                "top1": top1,
            }, checkpoint_path)

        # Lưu checkpoint sau MỖI EPOCH để nếu sập máy có thể chạy tiếp ngay!
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "ema_shadow": {k: v.cpu() for k, v in ema.shadow.items()} if ema is not None else None,
            "best_macro_f1": best_macro_f1,
            "best_epoch": best_epoch,
            "history": history,
        }, last_checkpoint_path)

    # 6. Đánh giá lại checkpoint tốt nhất trên Val và lưu predictions
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_names, val_targets, val_logits, _ = evaluate(model, val_loader, criterion, device)
    val_probs = np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True))
    val_probs = val_probs / np.sum(val_probs, axis=1, keepdims=True)
    val_pred_file = pred_path(cfg, "val")
    save_predictions(val_pred_file, val_names, val_targets, val_probs)

    # 7. NẾU bật save_test_predictions (chỉ ở Bước 4): Đánh giá TEST đúng 1 lần
    test_metrics = None
    if cfg.save_test_predictions:
        test_loader = ds.make_loader(
            test_df, effective_images_dir, eval_tf, cfg.batch_size, train=False,
            num_workers=cfg.num_workers
        )
        test_names, test_targets, test_logits, _ = evaluate(model, test_loader, criterion, device)
        test_probs = np.exp(test_logits - np.max(test_logits, axis=1, keepdims=True))
        test_probs = test_probs / np.sum(test_probs, axis=1, keepdims=True)
        test_pred_file = pred_path(cfg, "test")
        save_predictions(test_pred_file, test_names, test_targets, test_probs)
        test_metrics = compute_metrics(test_targets, test_probs.argmax(1), test_probs)

    # 8. Lưu history & biểu đồ
    pd.DataFrame(history).to_csv(save_dir / "history.csv", index=False)
    curve_img_path = Path(cfg.curves_dir) / f"{cfg.exp_id}_{cfg.backbone}.png"
    plot_curves(history, curve_img_path, f"{cfg.exp_id} - {cfg.backbone} (Seed {cfg.seed})")

    avg_epoch_time = sum(h["time_s"] for h in history) / len(history)

    result = {
        "exp_id": cfg.exp_id,
        "seed": cfg.seed,
        "backbone": cfg.backbone,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "best_val_top1": checkpoint["top1"],
        "params_m": n_params,
        "gmacs": gmacs,
        "avg_epoch_time_s": avg_epoch_time,
        "total_time_s": time.time() - start_time,
    }
    if test_metrics is not None:
        result["test_macro_f1"] = test_metrics["macro_f1"]
        result["test_top1"] = test_metrics["top1"]
        result["test_ece"] = test_metrics["ece"]

    return result


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Biến ['key=val', ...] thành dict có ép kiểu theo Config."""
    type_map = {f.name: f.type for f in fields(Config)}
    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Tham số không hợp lệ: '{pair}', cần có dạng KEY=VALUE")
        k, v = pair.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k not in type_map:
            raise KeyError(f"Trường '{k}' không tồn tại trong Config. Các trường hợp lệ: {list(type_map.keys())}")

        if v.lower() in ("none", "null"):
            overrides[k] = None
        elif type_map[k] in (int, "int"):
            overrides[k] = int(v)
        elif type_map[k] in (float, "float"):
            overrides[k] = float(v)
        elif type_map[k] in (bool, "bool"):
            overrides[k] = v.lower() in ("true", "1", "yes")
        else:
            overrides[k] = v

    return overrides


def main() -> None:
    """CLI runner: python train.py --set exp_id=B01 backbone=resnet50 seed=0."""
    parser = argparse.ArgumentParser(description="Chạy huấn luyện thí nghiệm DeepWeeds")
    parser.add_argument("--set", nargs="*", default=[], help="Ghi đè tham số dạng KEY=VALUE")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    res = run(cfg)
    print("\n=== KẾT QUẢ THÍ NGHIỆM ===")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

