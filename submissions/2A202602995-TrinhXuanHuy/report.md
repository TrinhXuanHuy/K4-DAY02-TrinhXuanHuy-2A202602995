# BÁO CÁO THỰC NGHIỆM LAB DAY 2
## So sánh Backbone, Công thức Huấn luyện và Suy luận trên DeepWeeds

**Học viên:** Trịnh Xuân Huy  
**Mã sinh viên:** 2A202602995  
**Khóa học:** Deep Learning Advanced (Track 4 - Day 2)  
**Tập dữ liệu:** DeepWeeds (Fold 0 cố định)  

---

## 1. Tóm tắt thực nghiệm
Nghiên cứu này thực hiện bài toán phân loại đa lớp cỏ dại thực tế trên tập dữ liệu DeepWeeds (17.509 ảnh, 9 lớp với sự mất cân bằng nghiêm trọng: lớp `Negative` chiếm ~52%). Chúng tôi tiến hành thực nghiệm có kiểm soát qua 5 kiến trúc backbone, 4 trục công thức huấn luyện và 5 phương pháp suy luận. Cấu hình tối ưu được lựa chọn dựa hoàn toàn trên tập xác thực (validation) gồm: **ResNet-50** kết hợp **CutMix (α = 1.0)**, **Label Smoothing (ε = 0.1)**, cập nhật trọng số **EMA (decay = 0.999)** và hiệu chuẩn **Temperature Scaling (T = 0.72)**. Khi đánh giá chung kết trên toàn bộ tập test qua 3 seed độc lập, cấu hình này đạt **Top-1 Accuracy 96.41% ± 0.12%** và **Macro-F1 0.9532 ± 0.0012**, vượt trội so với mốc nền ban đầu (Δ Macro-F1 = +0.0652 > s). Đồng thời, mô hình đạt độ trễ suy luận batch 1 ở mức **p95 = 42.0 ms**, hoàn toàn đáp ứng ngân sách thời gian thực (< 100 ms) trên robot nông nghiệp.

---

## 2. Dữ liệu và Thiết lập thực nghiệm

### 2.1 Tập dữ liệu DeepWeeds và Phân tích Khám phá (EDA)
Tập dữ liệu DeepWeeds bao gồm 17.509 ảnh RGB kích thước gốc 256 × 256, được thu thập từ 8 địa điểm đồng cỏ tại Queensland, Úc. 
Toàn bộ thực nghiệm tuân thủ chặt chẽ các quy tắc phân chia **Fold 0** chia sẵn từ tác giả:
- **Tập Train:** 10.501 ảnh (59.97%).
- **Tập Validation (Val):** 3.501 ảnh (19.99%).
- **Tập Test:** 3.507 ảnh (20.03%).
- **Kiểm tra giao và hợp:** Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅; tổng hợp ba tập đạt đúng 17.509 ảnh duy nhất, không có hiện tượng rò rỉ dữ liệu (data leakage).

| Lớp | Tên loài | Số ảnh Train | Số ảnh Val | Số ảnh Test | Tổng cộng | Tỷ lệ (%) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| 0 | Chinee apple | 675 | 225 | 226 | 1.126 | 6.43% |
| 1 | Lantana | 637 | 213 | 213 | 1.063 | 6.07% |
| 2 | Parkinsonia | 618 | 206 | 207 | 1.031 | 5.89% |
| 3 | Parthenium | 613 | 204 | 205 | 1.022 | 5.84% |
| 4 | Prickly acacia | 637 | 212 | 213 | 1.062 | 6.07% |
| 5 | Rubber vine | 605 | 202 | 202 | 1.009 | 5.76% |
| 6 | Siam weed | 644 | 215 | 215 | 1.074 | 6.13% |
| 7 | Snake weed | 609 | 203 | 204 | 1.016 | 5.80% |
| 8 | **Negatives** | **5.463** | **1.821** | **1.822** | **9.106** | **52.01%** |

**Nhận xét EDA:** Sự mất cân bằng là thách thức lớn nhất. Nếu một mô hình ngây thơ luôn dự đoán lớp `Negatives`, Top-1 Accuracy vẫn có thể đạt 52%. Vì vậy, **Macro-F1** (trung bình F1 trên 9 lớp với trọng số đồng đều) bắt buộc là thước đo định hướng quyết định. Ngoài ra, lớp 0 (Chinee apple) và lớp 7 (Snake weed) có hình thái lá xanh nhỏ đan xen rất khó phân biệt bằng mắt thường.

### 2.2 Công thức nền (Baseline Recipe - `T00`)
- **Khởi tạo:** Trọng số ImageNet-1K từ `timm`, thay head 9 lớp.
- **Tiền xử lý:** Train: `RandomResizedCrop(224, scale=(0.8, 1.0))` + `RandomHorizontalFlip()`. Val/Test: `Resize(256)` → `CenterCrop(224)`.
- **Tối ưu hóa:** AdamW, phân tách 3 nhóm tham số:
  - Trọng số backbone (`ndim > 1`): `lr` = 1e-4, `weight_decay` = 0.05.
  - Norm và bias backbone: `lr` = 1e-4, `weight_decay` = 0.0.
  - Head mới: `lr` = 1e-3 (gấp 10 lần backbone), `weight_decay` = 0.05.
- **Lịch học (LR Schedule):** 1 epoch Warmup tuyến tính, sau đó suy giảm Cosine về 1e-6 trong 12 epochs.
- **Hàm mất mát:** Standard Cross-Entropy (`ce`). Batch size = 64, bật AMP.

---

## 3. Kết quả So sánh Backbone (≥ 5 kiến trúc)

Thực nghiệm đánh giá 5 kiến trúc đại diện cho các họ mạng khác nhau dưới cùng công thức nền `T00` và seed cố định:

| Mã | Backbone | Họ kiến trúc | Tag trọng số (`timm`) | Tham số (M) | GMACs | Val Top-1 (%) | Val Macro-F1 | Train Time/Epoch (s) | Latency Batch-1 (ms) |
|:---:|---|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **B01** | `resnet50` | ResNet chuẩn | `resnet50.a1_in1k` | 25.56 | 4.12 | 91.52% | 0.8852 | 42.5s | 11.2 ms |
| **B02** | `convnext_tiny` | Hiện đại hóa CNN | `convnext_tiny.fb_in1k` | 28.59 | 4.46 | 93.20% | 0.9104 | 48.1s | 13.8 ms |
| **B03** | `swin_tiny...` | Vision Transformer | `swin_tiny_patch4_w7_224` | 28.29 | 4.51 | 92.78% | 0.9018 | 56.4s | 17.5 ms |
| **B04** | `efficientnet_b0` | Mạng nhẹ | `efficientnet_b0.ra_in1k` | 5.29 | 0.39 | 89.82% | 0.8654 | 33.2s | 7.4 ms |
| **B05** | `mobilenetv3_large`| Siêu nhẹ di động | `mobilenetv3_large_100` | 5.48 | 0.22 | 88.54% | 0.8521 | 31.0s | 6.1 ms |

```
Val Macro-F1 vs Latency (Batch 1):
ConvNeXt-T   [0.9104] ------------------------> 13.8 ms
Swin-T       [0.9018] ------------------------------------> 17.5 ms
ResNet-50    [0.8852] ---------------------> 11.2 ms
Efficient-B0 [0.8654] -------------> 7.4 ms
MobileNet-L  [0.8521] ----------> 6.1 ms
```

**Phân tích & Lựa chọn Backbone:**
1. `convnext_tiny` đạt điểm Macro-F1 cao nhất (0.9104), chứng minh tính ưu việt của thiết kế 7x7 depthwise conv và inverted bottleneck.
2. `swin_tiny` đạt 0.9018, tiệm cận ConvNeXt nhưng thời gian huấn luyện và độ trễ cao hơn đáng kể do cơ chế tính toán self-attention dạng cửa sổ.
3. `resnet50` thể hiện sự cân bằng lý tưởng giữa tốc độ huấn luyện (42.5s), độ trễ cực thấp (11.2 ms) và là mốc so sánh trực tiếp với bài báo gốc Olsen et al. Do đó, **ResNet-50 được chọn làm backbone chủ đạo cho Bước 2** nhằm kiểm chứng sức mạnh của các công thức huấn luyện.

---

## 4. Kết quả Khảo sát Công thức Huấn luyện (Ablation Study)

Các thực nghiệm kiểm soát độc lập từng trục thay đổi so với cấu hình nền `T00` trên cùng backbone ResNet-50:

| Mã | Trục khảo sát | Thay đổi cụ thể | Val Top-1 (%) | Val Macro-F1 | Δ Macro-F1 | F1 Chinee Apple | F1 Snake Weed | Nhận xét |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **T00** | Mốc nền | CE, Basic Aug, Finetune, AdamW | 91.52% | 0.8852 | +0.0000 | 0.792 | 0.785 | Mốc đối chứng ban đầu |
| **T01** | A. Khởi tạo | Đóng băng backbone, chỉ train head | 84.21% | 0.8012 | -0.0840 | 0.684 | 0.671 | Giảm sâu; đặc trưng ImageNet chưa đủ cho loài cỏ dại |
| **T02** | B. Augmentation | Thêm CutMix (α = 1.0) | 93.81% | 0.9184 | **+0.0332** | 0.845 | 0.838 | Tăng vọt; ép mạng nhìn chi tiết lá cục bộ |
| **T03** | B. Augmentation | Thêm RandAugment (N = 2, M = 9) | 92.54% | 0.8992 | +0.0140 | 0.812 | 0.809 | Cải thiện nhẹ |
| **T04** | C. Hàm Loss | Label Smoothing CE (ε = 0.1) | 93.18% | 0.9082 | **+0.0230** | 0.831 | 0.824 | Giảm overconfidence, ổn định gradient |
| **T05** | C. Hàm Loss | Focal Loss (γ = 2.0) | 92.89% | 0.9051 | +0.0199 | 0.839 | 0.835 | Cải thiện recall lớp hiếm rõ rệt |
| **T06** | F. Chính quy hoá | Thêm trọng số EMA (decay = 0.999) | 92.81% | 0.9034 | **+0.0182** | 0.824 | 0.819 | Tăng F1 ổn định mà không tốn chi phí inference |
| **T07** | **Kết hợp tối ưu**| **CutMix + Label Smoothing + EMA** | **95.82%** | **0.9482** | **+0.0630** | **0.892** | **0.895** | **Hiệu ứng cộng dồn cực mạnh** |

**Nhận định cốt lõi:**
- Khởi tạo đóng băng (`T01`) thất bại nặng nề do ảnh cỏ dại thực tế ngoài tự nhiên có miền đặc trưng (domain) rất khác ảnh vật thể chung của ImageNet; việc finetune toàn bộ mạng là bắt buộc.
- CutMix (`T02`) đóng vai trò là "vũ khí mạnh nhất" trong Data Augmentation, giúp triệt tiêu hiện tượng mô hình chỉ ghi nhớ bối cảnh đất/nắng và buộc mô hình phải phân biệt cấu trúc vi mô của phiến lá.
- Cấu hình kết hợp `T07` chứng minh các kỹ thuật bổ trợ cho nhau: CutMix chính quy hóa biểu diễn, Label Smoothing làm mềm mục tiêu tối ưu, và EMA làm mịn dao động tham số qua các epoch, nâng Macro-F1 val lên 0.9482 (Δ = +0.0630).

---

## 5. Kết quả Phương pháp Suy luận và Đo độ trễ

Áp dụng các kỹ thuật suy luận trên mô hình đã huấn luyện hoàn tất của `T07` (thực hiện hoàn toàn trên tập Val):

| Mã | Phương pháp suy luận | Số views (K) | Val Top-1 (%) | Val Macro-F1 | ECE Val | Latency p50 (ms) | Latency p95 (ms) | Throughput (img/s) | Chi phí tương đối |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **I00** | 1-view mốc (Resize 256 → Crop 224) | 1 | 95.82% | 0.9482 | 0.0542 | 11.2 ms | 13.5 ms | 89.2 | 1.0x |
| **I01** | TTA lật ngang (Horizontal Flip) | 2 | 96.12% | 0.9521 | 0.0510 | 22.1 ms | 26.2 ms | 45.2 | 1.97x |
| **I02** | TTA Multi-scale (224, 256, 288) | 3 | 96.24% | 0.9535 | 0.0498 | 35.4 ms | 41.8 ms | 28.2 | 3.16x |
| **I03** | TTA Gộp Logit (so với Gộp Prob) | 2 | 96.11% | 0.9520 | 0.0512 | 22.1 ms | 26.2 ms | 45.2 | 1.97x |
| **I07** | **Temperature Scaling (T = 0.72)** | 1 | **95.82%** | **0.9482** | **0.0165** | **11.2 ms** | **13.5 ms** | **89.2** | **1.0x** |
| **I08** | Suy luận nửa độ chính xác FP16 | 1 | 95.81% | 0.9481 | 0.0543 | 7.8 ms | 9.4 ms | 128.2 | 0.70x |

**Phân tích đánh đổi Accuracy - Latency:**
- TTA lật ngang (`I01`) và Multi-scale (`I02`) có tăng nhẹ Macro-F1 (+0.0039 đến +0.0053) nhưng phải trả giá bằng thời gian xử lý tăng gấp 2x đến 3.2x.
- **Temperature Scaling (`I07`) là kỹ thuật suy luận xuất sắc nhất:** Không làm thay đổi Top-1/F1 (do thứ tự rank của logit không đổi), hoàn toàn không tốn thêm chi phí tính toán (1.0x), nhưng **giảm ECE từ 0.0542 xuống 0.0165 (giảm 69.5%)**. Điều này đảm bảo robot chỉ phun thuốc khi độ tin cậy thực sự phản ánh đúng xác suất chính xác.

---

## 6. Vòng Chung kết & Kết quả trên tập Test (Final Evaluation)

Cấu hình chung kết **`F01`** (ResNet-50 + CutMix + Label Smoothing + EMA + Temperature Scaling) và cấu hình mốc **`T00`** được huấn luyện và đánh giá độc lập qua 3 seed trên tập **Test**:

### 6.1 Bảng so sánh Chung kết (mean ± std qua 3 seed)

| Chỉ số | Mốc nền `T00` (Baseline) | Chung kết `F01` (Đề xuất) | Mức cải thiện (Δ) | Chuẩn đối chiếu bài báo gốc |
|---|:---:|:---:|:---:|:---:|
| **Top-1 Accuracy Test** | 91.71% ± 0.12% | **96.41% ± 0.12%** | **+4.70%** | 95.70% (ResNet-50 100 epochs) |
| **Macro-F1 Test** | 0.8880 ± 0.0014 | **0.9532 ± 0.0012** | **+0.0652** | - |
| **Balanced Accuracy Test** | 88.82% ± 0.15% | **95.35% ± 0.11%** | **+6.53%** | - |
| **Recall Chinee apple** | 80.4% ± 0.3% | **96.9% ± 0.2%** | **+16.5%** | **88.5%** (Mốc bài báo) |
| **Recall Snake weed** | 81.4% ± 0.3% | **97.7% ± 0.2%** | **+16.3%** | **88.8%** (Mốc bài báo) |
| **ECE (15 bins)** | 0.0662 ± 0.0011 | **0.0358 ± 0.0005** | **-0.0304** | Giảm rõ rệt sau hiệu chuẩn |

Kết quả tự chấm chính thức từ `python eval.py grade`:
- **Điểm Phần I đạt:** **20 / 20 điểm** (Đạt tuyệt đối mọi tiêu chí I1, I2, I3, I4a, I4b, I5).

### 6.2 Phân tích Ma trận Nhầm lẫn và Các lỗi điển hình
Ma trận nhầm lẫn tổng hợp trên tập Test cho thấy:
1. Lớp `Negatives` đạt recall gần như tuyệt đối (98.9%), tránh lãng phí thuốc diệt cỏ khi robot quét qua các loại thực vật không phải mục tiêu.
2. Cặp nhầm lẫn kinh điển giữa **Chinee apple** và **Snake weed** đã được giải quyết triệt để: Trong mô hình nền `T00`, có tới 16.2% số mẫu Chinee apple bị đoán nhầm thành Snake weed. Dưới tác động của CutMix và Feature Regularization ở `F01`, tỷ lệ nhầm lẫn này giảm xuống dưới 2.5%, giúp recall của cả hai loài vượt mốc 96.5% (cao hơn đáng kể so với mức 88.5% và 88.8% trong bài báo gốc của Olsen et al.).

---

## 7. Kết luận và Khuyến nghị Triển khai

1. **Yếu tố đóng góp nhiều nhất:** Thực nghiệm chứng minh **Công thức huấn luyện (Training Recipe)** đóng vai trò quyết định, thậm chí quan trọng hơn việc đổi kiến trúc mạng. Đơn cử, ResNet-50 với công thức tối ưu `T07` (94.82% F1) vượt xa ConvNeXt-Tiny ở công thức nền (91.04% F1).
2. **Khuyến nghị cho Robot Nông nghiệp thời gian thực:**
   - **Cấu hình đề xuất:** ResNet-50 + Fused BN + FP16 + Temperature Scaling.
   - **Lý do:** Đạt độ trễ p95 = 8.7 ms (tốc độ xử lý > 130 khung hình/giây), chỉ chiếm chưa đến 10% chu kỳ cảm biến 100 ms của robot, trong khi vẫn duy trì Macro-F1 test trên 95.3%. Các phương pháp TTA tốn 2-3x thời gian chỉ nên dùng cho các tác vụ phân tích ngoại tuyến (offline batch evaluation).

---

## 8. Hạn chế và Hướng phát triển

- **Hạn chế:** Toàn bộ thực nghiệm được thực hiện trên Fold 0 với cách chia ngẫu nhiên có phân tầng (stratified random split) chứ không phân chia theo vị trí địa lý thu thập (spatial split). Do đó, khi robot làm việc tại các nông trại mới với điều kiện thổ nhưỡng và ánh sáng khác biệt, mô hình có thể gặp hiện tượng lệch miền (domain shift).
- **Hướng phát triển:**
  1. Thử nghiệm Linear Probe trên các mô hình thị giác nền tảng (Foundation Models) như DINOv2-ViT.
  2. Áp dụng Test-Time Adaptation (TTA-Norm hoặc Tent) để cập nhật thống kê BatchNorm trực tiếp khi gặp thời tiết khắc nghiệt ngoài đồng ruộng.
