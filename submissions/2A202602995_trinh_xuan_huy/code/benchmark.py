"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Triển khai đầy đủ cho bài nộp Lab Day 2 (DeepWeeds).
Quy tắc đo:
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99
"""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict[str, float]:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây (ms).

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.
    """
    # 1. Warmup
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    # 2. Đo lặp lại
    times = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times = np.array(times, dtype=np.float64)
    return {
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
        "mean": float(np.mean(times)),
        "std": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
        "n": int(iters),
    }


def latency_report(model: nn.Module, batch_size: int, img_size: int, dtype: str = "fp32",
                   device: str = "cuda", warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size)."""
    model.eval()
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    model = model.to(dev)

    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=dev)

    if dtype == "fp16":
        model = model.half()
        dummy_input = dummy_input.half()

    sync_fn = torch.cuda.synchronize if dev.type == "cuda" else None

    @torch.inference_mode()
    def forward_fn():
        if dtype == "amp" and dev.type == "cuda":
            with torch.cuda.amp.autocast():
                _ = model(dummy_input)
        else:
            _ = model(dummy_input)

    res = bench(forward_fn, warmup=warmup, iters=iters, sync=sync_fn)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    throughput = (batch_size / (res["p50"] / 1000.0)) if res["p50"] > 0 else 0.0

    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": res["p50"],
        "p95": res["p95"],
        "p99": res["p99"],
        "mean": res["mean"],
        "images_per_s": round(throughput, 2),
        "torch": torch.__version__,
    }


def tta_latency(model: nn.Module, k_views: int = 2, batch_size: int = 1, img_size: int = 224,
                dtype: str = "fp32", device: str = "cuda", warmup: int = 10, iters: int = 50) -> dict:
    """Đo độ trễ thực tế của TTA K views so với 1 view thông thường."""
    model.eval()
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    model = model.to(dev)

    dummy_views = [torch.randn(batch_size, 3, img_size, img_size, device=dev) for _ in range(k_views)]
    sync_fn = torch.cuda.synchronize if dev.type == "cuda" else None

    @torch.inference_mode()
    def tta_fn():
        outputs = []
        for v in dummy_views:
            outputs.append(model(v))
        return outputs

    res = bench(tta_fn, warmup=warmup, iters=iters, sync=sync_fn)
    return {
        "k_views": k_views,
        "p50": res["p50"],
        "p95": res["p95"],
        "p99": res["p99"],
        "mean": res["mean"],
    }

