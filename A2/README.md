# A2 — Phân đoạn ngữ nghĩa trên Pascal VOC 2012: SegFormer so với DeepLabV3+

CO5085 · Học sâu và ứng dụng trong thị giác máy tính · HK261 (2026–2027)
Học viên: **Trần Hoài Tâm** – MSHV 2470743
Giảng viên: Lê Thành Sách, Nguyễn Quốc Minh

- Trang A2: https://tamtran9x.github.io/2470743-CO5085-Assignment/A2/
- Báo cáo PDF: [`report/report.pdf`](report/report.pdf)
- **Chủ đề:** phân đoạn ảnh (semantic segmentation), chủ đề 1 của đề A2.
- **Paper được phân tích:** E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, P. Luo. *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.* NeurIPS 2021. [arXiv:2105.15203](https://arxiv.org/abs/2105.15203)

Bài tập tự cài đặt **SegFormer** (encoder MiT + decoder MLP) và **DeepLabV3+** bằng PyTorch, huấn luyện trên **Pascal VOC 2012 + SBD** (10.082 ảnh), đánh giá trên VOC 2012 val (1.449 ảnh), rồi đối chiếu các nhận định của paper với thí nghiệm: kích thước mô hình, vai trò của encoder đối với decoder MLP, effective receptive field, độ bền với nhiễu ảnh và với độ phân giải.

## Kết quả (VOC 2012 val, single-scale)

| Mô hình | Pre-train | mIoU (%) | mDice (%) | Boundary mIoU (%) | Tham số (M) | GMACs | FPS | Train (phút) | Mem (GB) |
|---|---|---|---|---|---|---|---|---|---|
| DeepLabV3+ R50 | IN-1k | 75,67 | 84,97 | 67,30 | 26,7 | 37,7 | 59 | 66 | 3,9 |
| ResNet-50 + MLP | IN-1k | 69,16 | 80,48 | 60,73 | 28,8 | 29,9 | 52 | 72 | 4,1 |
| SegFormer-B0 | IN-1k | 69,57 | 80,63 | 58,17 | 3,7 | 3,6 | 106 | 29 | 1,9 |
| SegFormer-B1 | IN-1k | 75,63 | 84,93 | 64,70 | 13,7 | 11,1 | 72 | 40 | 2,8 |
| SegFormer-B2 | IN-1k | 79,93 | 87,77 | 70,15 | 27,4 | 22,7 | 36 | 79 | 4,9 |
| SegFormer-B0 (scratch) | – | 19,51 | 28,08 | 13,30 | 3,7 | 3,6 | 107 | 29 | 1,9 |

Boundary mIoU: mIoU trên dải 8 pixel quanh đường biên của nhãn thật. GMACs cho ảnh 512×512; FPS với batch 1; thời gian và bộ nhớ khi huấn luyện batch 8 trên RTX 3050 Laptop 6 GB. Bảng đầy đủ: [`results/summary.csv`](results/summary.csv); kiểm định và phân tích lỗi: [`results/analysis.json`](results/analysis.json).

**Kết luận chính** (chi tiết trong báo cáo PDF, Mục 6–8):

- **SegFormer-B2 hơn DeepLabV3+ ResNet-50 4,25 điểm mIoU** với cùng ~27 triệu tham số và cùng công thức huấn luyện (CI 95% [+2,90; +5,68], bootstrap ghép cặp), tốt hơn ở 20/21 lớp. Đổi lại nó suy luận chậm hơn (36 so với 59 FPS) dù cần ít GMACs hơn.
- **SegFormer-B1 ngang DeepLabV3+ với một nửa số tham số** (chênh +0,04 điểm, CI [−1,48; +1,54]), nhanh hơn và bền với nhiễu hơn; nhưng DeepLabV3+ hơn 2,6 điểm ở vùng biên.
- **Decoder MLP cần encoder Transformer** (Bảng 1d của paper): ResNet-50 + MLP kém MiT-B2 + MLP 10,8 điểm (paper: 10,7 trên ADE20K). Phép đo effective receptive field giải thích vì sao: 90% gradient của đầu ra decoder nằm trong 227 px với ResNet-50 + MLP, so với 461 px (DeepLabV3+) và 473 px (SegFormer-B2).
- **SegFormer bền với nhiễu ảnh hơn**: trung bình trên 27 điều kiện nhiễu, B2 giữ 85% mIoU so với 68% của DeepLabV3+. Nhưng nó **không** bền với độ phân giải hơn CNN.
- **Pretraining là bắt buộc**: SegFormer-B0 không pretrain chỉ đạt 19,51% so với 69,57%.
- **Decoder của paper không "nhẹ" về bộ nhớ**: cài nguyên bản, B2 cần 7,2 GB khi huấn luyện; gộp các phép tuyến tính (cùng kết quả) còn 4,7 GB và GMACs giảm từ 60,7 xuống 22,7.

## Protocol tóm tắt

| | |
|---|---|
| Dữ liệu | VOC 2012 + SBD ("train_aug"): train 10.082 · minival 500 (chỉ vẽ đường học) · val 1.449 (tập kiểm tra, chấm một lần) |
| Augmentation | co giãn 0,5–2,0 → đệm + cắt 512×512 → lật ngang → biến đổi màu → chuẩn hoá ImageNet |
| Tối ưu | AdamW, lr 6e-5 (encoder) / 6e-4 (decoder), wd 0,01, warmup 500 vòng lặp + poly, 10.000 vòng lặp × batch 8, bfloat16 |
| Loss | cross-entropy từng pixel, bỏ qua nhãn void (255) |
| Đánh giá | trọng số ở vòng lặp cuối (không chọn checkpoint), cả ảnh đệm lên 512×512, single-scale |
| Metric | mIoU, mDice, pixel accuracy, boundary mIoU; CI 95% bằng bootstrap theo ảnh |
| Phần cứng | RTX 3050 Laptop 6 GB, PyTorch 2.5.1 + CUDA 12.4, timm 1.0.30 |

## Cấu trúc thư mục

```
A2/
├── index.html            # trang A2 trên GitHub Pages (sinh từ page/index.template.html)
├── src/
│   ├── data.py           # Dataset VOC + SBD, augmentation ảnh/nhãn, DataLoader, tập đánh giá
│   ├── models.py         # SegFormer (MiT + decoder MLP) và DeepLabV3+ tự cài đặt
│   ├── metrics.py        # ma trận nhầm lẫn, mIoU/Dice, vòng đánh giá, bootstrap
│   ├── train.py          # huấn luyện 1 cấu hình + đánh giá trên VOC 2012 val
│   ├── eda.py            # thống kê và hình EDA
│   ├── erf.py            # effective receptive field (Hình 3 của paper)
│   ├── robustness.py     # nhiễu ảnh và độ phân giải khi suy luận (Bảng 5, 1c của paper)
│   ├── analyze.py        # bảng tổng hợp, kiểm định, phân tích lỗi, hình kết quả
│   └── plotstyle.py      # bảng màu chung cho mọi hình
├── scripts/
│   ├── prepare_data.py   # tải VOC 2012 + SBD, chuyển nhãn SBD, tạo split
│   ├── run_all.py        # chạy cả 6 cấu hình (bỏ qua run đã xong, resume run dở)
│   ├── post_train.ps1    # mọi bước cần GPU theo thứ tự (Windows)
│   ├── benchmark.py      # đo tốc độ/bộ nhớ; so hai cách tính decoder MLP
│   ├── verify_segformer.py  # so bản tự cài đặt với bản tham chiếu của transformers
│   ├── demo.py           # phân đoạn ảnh bất kỳ bằng mô hình đã huấn luyện
│   ├── build_report.py   # biên dịch report/report.typ → report.pdf
│   ├── build_overleaf.py # sinh report/overleaf/main.tex (XeLaTeX) từ template
│   ├── build_page.py     # sinh index.html và README.md từ results/
│   └── make_dist.py      # gom thư mục để đưa lên GitHub
├── results/              # metrics.json, history.csv, preds.npz của từng run + các bảng tổng hợp
├── figures/              # hình dùng trong báo cáo và trang A2
├── demo/                 # kết quả demo trên ảnh val
├── logs/                 # log huấn luyện
└── report/               # báo cáo (Typst), PDF và bản Overleaf
```

## Tái lập

```bash
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
python -m scripts.prepare_data
python -m src.eda
python -m scripts.run_all
python -m scripts.benchmark --crops 512 --json results/benchmark.json
python -m src.erf
python -m src.robustness
python -m src.analyze
python -m scripts.build_report
python -m scripts.build_page
```

Chạy một cấu hình riêng: `python -m src.train --arch {deeplabv3plus_r50,resnet50_mlp,segformer_b0,segformer_b1,segformer_b2}`; thêm `--no-pretrained` để bỏ pretraining.
Demo trên ảnh bất kỳ: `python -m scripts.demo --images anh.jpg` (cần `checkpoints/`, không nằm trong repository vì dung lượng).
Dữ liệu (~3,4 GB) được tải từ máy chủ chính thức của VOC (Oxford) và SBD (Berkeley). Trên GPU 6 GB, sáu cấu hình mất khoảng 5,3 giờ huấn luyện, cộng khoảng 1 giờ cho các thí nghiệm chỉ suy luận.

> Ghi chú: trên laptop, hãy cắm sạc khi huấn luyện. Khi chạy bằng pin, GPU bị giới hạn công suất và chậm hơn khoảng 20 lần; `scripts/post_train.ps1` tự chờ đến khi có nguồn điện.
