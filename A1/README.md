# A1 — CNN vs Transformer trên CIFAR-100

CO5085 · Học sâu và ứng dụng trong thị giác máy tính · HK261 (2026–2027)
Học viên: **Trần Hoài Tâm** – MSHV 2470743 (làm cá nhân) · Giảng viên: Lê Thành Sách, Nguyễn Quốc Minh

- Trang A1: https://tamtran9x.github.io/2470743-CO5085-Assignment/A1/
- Báo cáo PDF: [`report/report.pdf`](report/report.pdf)

So sánh **ResNet-50** (CNN) và **DeiT-S** (Vision Transformer) trên **CIFAR-100**, mỗi mô hình ở 4 chế độ:
from scratch, pretrained + freeze toàn bộ backbone, pretrained + freeze nửa đầu, pretrained + full fine-tune.
Cả hai backbone đều chỉ được pretrain trên ImageNet-1k (`resnet50.tv2_in1k`, `deit_small_patch16_224.fb_in1k`), có kích thước tương đương (~22–24M tham số, ~4–5 GMACs).

## Kết quả (test, 10.000 ảnh)

<!-- RESULTS_TABLE -->
| Model | Chế độ | Test acc (%) | Top-5 (%) | Macro-F1 (%) | Val acc (%) | Trainable (M) | Train (phút) | s/epoch | Peak mem (GB) |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 (CNN) | From scratch | 71.37 | 90.48 | 71.27 | 71.30 | 23.71 | 100.4 | 201 | 3.26 |
| DeiT-S (ViT) | From scratch | 48.26 | 71.55 | 47.76 | 47.48 | 21.70 | 85.5 | 171 | 2.62 |
| ResNet-50 (CNN) | Freeze backbone | 75.19 | 94.62 | 75.17 | 75.28 | 0.20 | 10.9 | 66 | 0.64 |
| DeiT-S (ViT) | Freeze backbone | 74.52 | 94.28 | 74.51 | 75.60 | 0.04 | 8.6 | 52 | 0.45 |
| ResNet-50 (CNN) | Freeze một phần | 85.53 | 97.15 | 85.53 | 84.54 | 22.27 | 19.1 | 115 | 1.29 |
| DeiT-S (ViT) | Freeze một phần | 85.26 | 96.92 | 85.22 | 85.36 | 10.69 | 18.6 | 111 | 1.48 |
| ResNet-50 (CNN) | Full fine-tune | 86.01 | 97.66 | 86.03 | 85.68 | 23.71 | 33.4 | 200 | 3.26 |
| DeiT-S (ViT) | Full fine-tune | 88.57 | 97.94 | 88.60 | 87.62 | 21.70 | 28.8 | 173 | 2.62 |
<!-- /RESULTS_TABLE -->

Bảng đầy đủ: [`results/summary.csv`](results/summary.csv). Phân tích lỗi: [`results/analysis.json`](results/analysis.json).

**Kết luận chính** (chi tiết và kiểm định thống kê trong báo cáo PDF, Mục 6–8):

- Pretrained + full fine-tune tốt nhất ở cả hai họ; DeiT-S hơn ResNet-50 2,56 điểm (McNemar p < 0,001). Ensemble hai mô hình (trung bình softmax) đạt 89,67%.
- From scratch thì ngược lại: ResNet-50 hơn DeiT-S 23 điểm. ViT không có inductive bias của convolution, nên với 45k ảnh nó học thuộc (train 99,7%, val ~47%). Pretraining tăng 40 điểm cho DeiT-S và 15 điểm cho ResNet-50.
- Freeze backbone nhanh hơn ~3 lần và ít bộ nhớ hơn 5 lần, nhưng kém full fine-tune 11–14 điểm do domain gap. Ở chế độ này hai họ không khác nhau có ý nghĩa (p = 0,13).
- Freeze một phần là điểm cân bằng tốt cho ResNet-50 (kém 0,48 điểm, giảm 43% thời gian và 60% bộ nhớ). Với DeiT-S nó kém full fine-tune 3,3 điểm.
- Khoảng một nửa số lỗi của mô hình tốt nhất là nhầm giữa các lớp trong cùng superclass (maple/oak tree, boy/girl/man/woman, otter/seal).

## Protocol tóm tắt

| | |
|---|---|
| Dữ liệu | CIFAR-100 (100 lớp / 20 superclass). Train 45.000 · Val 5.000 (tách phân tầng 50 ảnh/lớp, seed 0) · Test 10.000 (chính thức) |
| Đầu vào | crop 32 + pad 4, lật ngang → resize bilinear 224×224 → chuẩn hoá ImageNet (val/test: không augment) |
| Tối ưu | AdamW, wd 0,05, batch 64, cosine + warmup, label smoothing 0,1, grad clip 1,0, bfloat16 |
| Scratch | 30 epoch, lr 5e-4 |
| Pretrained | 10 epoch, lr backbone 1e-4, lr head 1e-3 (freeze backbone: chỉ head) |
| Chọn mô hình | checkpoint có val accuracy cao nhất → đánh giá test một lần |
| Phần cứng | RTX 3050 Laptop 6 GB, PyTorch 2.5.1 + CUDA 12.4, timm 1.0.30 |

## Cấu trúc thư mục

```
A1/
├── index.html            # trang A1 trên GitHub Pages
├── src/
│   ├── data.py           # tải CIFAR-100, split phân tầng, DataLoader + augmentation trên GPU
│   ├── models.py         # ResNet-50 / DeiT-S và 4 chế độ freeze
│   ├── train.py          # huấn luyện 1 cấu hình + đánh giá test
│   ├── eda.py            # thống kê và hình EDA
│   ├── analyze.py        # bảng tổng hợp, biểu đồ, phân tích lỗi
│   └── plotstyle.py      # bảng màu chung cho mọi hình
├── scripts/
│   ├── run_all.py        # chạy cả 8 cấu hình (bỏ qua run đã xong, resume run dở)
│   ├── benchmark.py      # đo tốc độ/bộ nhớ để chọn kích thước ảnh, batch
│   └── build_report.py   # biên dịch report/report.typ → report.pdf
├── results/              # metrics.json, history.csv, preds.npz của từng run + summary
├── figures/              # hình dùng trong báo cáo và trang A1
├── logs/                 # log huấn luyện
└── report/               # báo cáo (Typst) và PDF
```

## Tái lập

```bash
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
python -m src.eda
python -m scripts.run_all
python -m src.analyze
python -m scripts.build_report
```

Chạy một cấu hình riêng: `python -m src.train --arch {resnet50,deit_s} --mode {scratch,linear,partial,full}`.
Dữ liệu (~185 MB) được tải tự động từ mirror Hugging Face `uoft-cs/cifar100` và cache tại `data/cifar100.npz`.
Trên GPU 6 GB, toàn bộ 8 cấu hình mất khoảng 5–6 giờ.

> Ghi chú Windows: nếu bật Smart App Control, các bản PyTorch mới (2.11) có thể bị chặn nạp DLL (`WinError 4551`); bản 2.5.1+cu124 chạy được bình thường.
