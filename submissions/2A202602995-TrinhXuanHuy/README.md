# Báo cáo Thực nghiệm Lab Day 2 — DeepWeeds Classification
**Môn học:** Deep Learning Advanced (Track 4 - Day 2)  
**Học viên:** Trịnh Xuân Huy  
**Mã sinh viên (MSV):** 2A202602995  
**Thư mục bài nộp:** `submissions/2A202602995-TrinhXuanHuy/`  

---

## 1. Liên kết Notebook chạy thực nghiệm
- **Google Colab Notebook (Link chính thức):** [https://drive.google.com/file/d/1Rz41j5dr1bm9BCJvqPvv171HhOHkUBn6/view?usp=sharing](https://drive.google.com/file/d/1Rz41j5dr1bm9BCJvqPvv171HhOHkUBn6/view?usp=sharing)
*(File notebook `code/lab_day2.ipynb` đã hoàn thiện và chạy đầy đủ từ Bước 0 đến Bước 5 trên GPU T4)*
- **Kaggle Notebooks:** Khuyến nghị môi trường GPU T4 / P100 với tùy chọn *Save & Run All*.

---

## 2. Môi trường & Phiên bản thư viện
Thực nghiệm được thiết kế và kiểm thử trên môi trường Python 3.10 / 3.11 với các thư viện:
- `python >= 3.10`
- `torch >= 2.0.0`
- `torchvision >= 0.15.0`
- `timm >= 0.9.12`
- `pandas >= 2.0.0`
- `numpy >= 1.24.0`
- `scikit-learn >= 1.3.0`
- `openpyxl >= 3.1.0`
- `matplotlib >= 3.7.0`
- `scipy >= 1.10.0`

---

## 3. Cấu trúc thư mục bài nộp
```
submissions/2A202602995-TrinhXuanHuy/
├── README.md               # File này: link thực thi, môi trường, hướng dẫn chạy
├── results.xlsx            # File Excel gồm 7 sheets tổng hợp toàn bộ kết quả thí nghiệm
├── report.md               # Báo cáo chi tiết thực nghiệm khoa học và phân tích lỗi
├── curves/                 # Ảnh biểu đồ loss train/val và macro-F1 val theo epoch cho từng thí nghiệm
│   ├── B01_resnet50.png
│   ├── B02_convnext_tiny.png
│   ├── B03_swin_tiny_patch4_window7_224.png
│   ├── B04_efficientnet_b0.png
│   ├── B05_mobilenetv3_large_100.png
│   ├── T00_resnet50.png
│   ├── T01_resnet50.png
│   ├── T02_resnet50.png
│   ├── T04_resnet50.png
│   ├── T06_resnet50.png
│   ├── T07_resnet50.png
│   └── F01_resnet50.png
├── predictions/            # File CSV dự đoán test và val theo định dạng eval.py
│   ├── F01_seed0_test.csv
│   ├── F01_seed1_test.csv
│   ├── F01_seed2_test.csv
│   ├── F01_seed0_val.csv
│   ├── F01_uncal_seed0_test.csv
│   ├── T00_seed0_test.csv
│   ├── T00_seed1_test.csv
│   └── T00_seed2_test.csv
└── code/                   # Toàn bộ mã nguồn hoàn thiện
    ├── dataset.py          # Data pipeline, kiểm tra split dữ liệu, augmentation
    ├── model.py            # Backbone timm, param groups 3 nhóm, count params/gmacs
    ├── losses.py           # Focal Loss, Label Smoothing CE, Class Weights, CutMix
    ├── train.py            # Vòng huấn luyện dùng chung, AMP, Cosine Warmup, EMA
    ├── inference.py        # TTA (lật/crop), Ensemble, Temperature Scaling, Conv-BN
    ├── benchmark.py        # Đo độ trễ p50/p95/p99 chuẩn xác GPU synchronize
    ├── test_code.py        # Unit test kiểm tra tính đúng đắn các hàm tự viết
    └── lab_day2.ipynb      # Notebook Colab/Kaggle tích hợp quy trình End-to-End
```

---

## 4. Hướng dẫn chạy lại thực nghiệm (Reproducibility)

### Bước 0: Chuẩn bị dữ liệu và cài đặt gói
```bash
pip install -q timm openpyxl ptflops
# Tải ảnh từ Zenodo (kiểm tra MD5 b7b30f96d466fba86016aa5a26606e0f)
# Tải các file nhãn từ https://github.com/AlexOlsen/DeepWeeds
```

### Bước 1: Huấn luyện dòng lệnh (CLI)
Có thể chạy từng thí nghiệm qua file `train.py`:
```bash
# Huấn luyện baseline T00
python code/train.py --set exp_id=T00 backbone=resnet50 seed=0 epochs=12

# Huấn luyện cấu hình chung kết F01 với CutMix, Label Smoothing và EMA
python code/train.py --set exp_id=F01 backbone=resnet50 mix=cutmix loss=ls label_smoothing=0.1 ema_decay=0.999 seed=0 epochs=12 save_test_predictions=True
```

### Bước 2: Tự động chấm điểm bằng `eval.py`
```bash
# 1. Tính toán metrics chính thức
python eval.py score --pred "predictions/F01_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01

# 2. Tự chấm Phần I của RUBRIC (thang 20 điểm)
python eval.py grade \
    --final "predictions/F01_seed*_test.csv" \
    --baseline "predictions/T00_seed*_test.csv" \
    --uncal "predictions/F01_uncal_seed*_test.csv" \
    --final-val "predictions/F01_seed*_val.csv" \
    --latency-p95-ms 42.0 --latency-method proper \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv
```

