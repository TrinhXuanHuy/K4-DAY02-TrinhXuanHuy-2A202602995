"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Triển khai đầy đủ cho bài nộp Lab Day 2 (DeepWeeds).
Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    KHÔNG sửa, lọc hay chia lại dữ liệu.
    """
    labels_path = Path(labels_dir)
    train_file = labels_path / f"train_subset{fold}.csv"
    val_file = labels_path / f"val_subset{fold}.csv"
    test_file = labels_path / f"test_subset{fold}.csv"

    if not train_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {val_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {test_file}")

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    labels_file = labels_path / "labels.csv"
    species_map = {}
    if labels_file.exists():
        try:
            labels_info = pd.read_csv(labels_file)
            if "Species" in labels_info.columns and "Label" in labels_info.columns:
                species_map = dict(zip(labels_info["Label"], labels_info["Species"]))
        except Exception:
            pass

    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        missing = [c for c in ["Filename", "Label"] if c not in df.columns]
        if missing:
            raise ValueError(f"{name} thiếu cột {missing}")
        if "Species" not in df.columns:
            if species_map:
                df["Species"] = df["Label"].map(species_map)
            else:
                df["Species"] = df["Label"].map(lambda x: CLASS_NAMES[x] if 0 <= x < len(CLASS_NAMES) else str(x))

    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path | None = None) -> dict[str, Any]:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    1. số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
    2. giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
    3. hợp ba tập phải bằng đúng 17.509 ảnh
    4. mọi Filename đều tồn tại trong `images_dir` (nếu images_dir tồn tại)
    """
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    n_total = n_train + n_val + n_test

    train_files = set(train_df["Filename"])
    val_files = set(val_df["Filename"])
    test_files = set(test_df["Filename"])

    train_val_overlap = len(train_files & val_files)
    train_test_overlap = len(train_files & test_files)
    val_test_overlap = len(val_files & test_files)

    if train_val_overlap > 0:
        raise AssertionError(f"LỖI: train và val có {train_val_overlap} ảnh trùng nhau!")
    if train_test_overlap > 0:
        raise AssertionError(f"LỖI: train và test có {train_test_overlap} ảnh trùng nhau!")
    if val_test_overlap > 0:
        raise AssertionError(f"LỖI: val và test có {val_test_overlap} ảnh trùng nhau!")

    all_files = train_files | val_files | test_files
    if len(all_files) != 17509:
        raise AssertionError(f"LỖI: Tổng số ảnh trong hợp 3 tập là {len(all_files)}, kỳ vọng đúng 17509!")

    if images_dir is not None:
        img_path = Path(images_dir)
        if img_path.exists():
            missing = [f for f in all_files if not (img_path / f).exists()]
            if missing:
                raise FileNotFoundError(f"LỖI: Thiếu {len(missing)} file ảnh trong {images_dir} (ví dụ: {missing[:5]})")

    per_class_train = train_df["Label"].value_counts().sort_index().to_dict()
    per_class_val = val_df["Label"].value_counts().sort_index().to_dict()
    per_class_test = test_df["Label"].value_counts().sort_index().to_dict()

    summary = {
        "n": {
            "train": n_train,
            "val": n_val,
            "test": n_test,
            "total": n_total,
            "ratio": (round(n_train / n_total, 4), round(n_val / n_total, 4), round(n_test / n_total, 4)),
        },
        "per_class": {
            "train": per_class_train,
            "val": per_class_val,
            "test": per_class_test,
        },
        "overlap": {
            "train_val": train_val_overlap,
            "train_test": train_test_overlap,
            "val_test": val_test_overlap,
        },
        "total_unique": len(all_files),
    }

    print(f"=== KIỂM TRA CHIA DỮ LIỆU THÀNH CÔNG ===")
    print(f"Train: {n_train} ({n_train/n_total*100:.2f}%), Val: {n_val} ({n_val/n_total*100:.2f}%), Test: {n_test} ({n_test/n_total*100:.2f}%)")
    print(f"Tổng số ảnh duy nhất: {len(all_files)} / 17509")
    print(f"Giao các tập: train ∩ val = {train_val_overlap}, train ∩ test = {train_test_overlap}, val ∩ test = {val_test_overlap}")
    return summary


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo torchvision transforms theo `train` và mức augmentation `aug`.

    Giá trị `aug`:
      - "basic": RandomResizedCrop + RandomHorizontalFlip
      - "color": basic + ColorJitter (đổi màu, độ sáng, tương phản)
      - "trivial": basic + TrivialAugmentWide
      - "randaug": basic + RandAugment
    Val/test: Resize 256 -> CenterCrop(img_size) (hoặc Resize(img_size, img_size)).
    """
    import torchvision.transforms as T

    if train:
        aug_ops = []
        if aug == "color":
            aug_ops.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
        elif aug == "trivial":
            aug_ops.append(T.TrivialAugmentWide())
        elif aug == "randaug":
            aug_ops.append(T.RandAugment(num_ops=2, magnitude=9))

        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            *aug_ops,
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        if img_size == 256:
            return T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class DeepWeedsDataset:
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].tolist()
        self.labels = self.df["Label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int) -> tuple[Any, int, str]:
        filename = self.filenames[i]
        label = self.labels[i]
        path = self.images_dir / filename
        with Image.open(path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, int(label), filename


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """Tạo torch DataLoader."""
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    dataset = DeepWeedsDataset(df, images_dir, transform)

    if train:
        if sampler == "balanced":
            counts = df["Label"].value_counts().to_dict()
            sample_weights = [1.0 / counts[lbl] for lbl in dataset.labels]
            sampler_obj = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler_obj,
                shuffle=False,
                drop_last=True,
                pin_memory=torch.cuda.is_available(),
                num_workers=num_workers,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=torch.cuda.is_available(),
                num_workers=num_workers,
            )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=num_workers,
        )

