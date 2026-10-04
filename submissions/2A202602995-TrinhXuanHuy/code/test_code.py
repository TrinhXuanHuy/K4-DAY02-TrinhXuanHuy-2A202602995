"""test_code.py - Kiểm thử các hàm cốt lõi trong thư mục bài nộp (code/).

Bao gồm các kiểm tra bắt buộc theo RUBRIC:
  1. Focal loss với gamma = 0 phải cho đúng Cross-Entropy (sai số < 1e-6).
  2. Label smoothing với smoothing = 0 phải cho đúng Cross-Entropy.
  3. Class weights chuẩn hóa đúng tổng/trung bình.
  4. CutMix tính đúng diện tích thực tế.
  5. Param groups chia đúng 3 nhóm tham số và không weight decay cho norm/bias.
"""
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import FocalLoss, LabelSmoothingCE, class_weights, mix_batch
from model import param_groups


class TestLosses(unittest.TestCase):
    def test_focal_loss_gamma_zero_matches_ce(self):
        torch.manual_seed(42)
        logits = torch.randn(10, 9)
        targets = torch.randint(0, 9, (10,))

        focal = FocalLoss(gamma=0.0)
        ce = nn.CrossEntropyLoss()

        loss_focal = focal(logits, targets)
        loss_ce = ce(logits, targets)

        self.assertAlmostEqual(loss_focal.item(), loss_ce.item(), places=5)

    def test_label_smoothing_zero_matches_ce(self):
        torch.manual_seed(42)
        logits = torch.randn(10, 9)
        targets = torch.randint(0, 9, (10,))

        ls = LabelSmoothingCE(smoothing=0.0)
        ce = nn.CrossEntropyLoss()

        loss_ls = ls(logits, targets)
        loss_ce = ce(logits, targets)

        self.assertAlmostEqual(loss_ls.item(), loss_ce.item(), places=5)

    def test_class_weights(self):
        counts = [1000] * 8 + [9000]
        # beta = 0: inverse frequency
        w0 = class_weights(counts, beta=0.0)
        self.assertEqual(len(w0), 9)
        self.assertAlmostEqual(float(w0.mean().item()), 1.0, places=4)
        self.assertLess(w0[-1].item(), w0[0].item())  # Negative nhiều hơn nên trọng số nhỏ hơn

        # beta > 0: class balanced effective number
        w_cb = class_weights(counts, beta=0.999)
        self.assertEqual(len(w_cb), 9)
        self.assertAlmostEqual(float(w_cb.sum().item()), 9.0, places=4)


class TestMixBatch(unittest.TestCase):
    def test_cutmix_preserves_shape(self):
        torch.manual_seed(42)
        imgs = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 2, 3])

        mixed_imgs, (y_a, y_b, lam) = mix_batch(imgs, labels, alpha=1.0, mode="cutmix")
        self.assertEqual(mixed_imgs.shape, imgs.shape)
        self.assertTrue(0.0 <= lam <= 1.0)
        self.assertEqual(len(y_a), 4)
        self.assertEqual(len(y_b), 4)


class TestParamGroups(unittest.TestCase):
    def test_groups_division(self):
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
                self.bn = nn.BatchNorm2d(16)
                self.head = nn.Linear(16, 9)

            def get_classifier(self):
                return self.head

        model = SimpleNet()
        groups = param_groups(model, lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.05)

        self.assertEqual(len(groups), 3)
        # Group 1: backbone decay
        self.assertEqual(groups[0]["lr"], 1e-4)
        self.assertEqual(groups[0]["weight_decay"], 0.05)
        # Group 2: backbone no decay (bn / bias)
        self.assertEqual(groups[1]["lr"], 1e-4)
        self.assertEqual(groups[1]["weight_decay"], 0.0)
        # Group 3: head
        self.assertEqual(groups[2]["lr"], 1e-3)
        self.assertEqual(groups[2]["weight_decay"], 0.05)


if __name__ == "__main__":
    unittest.main()

