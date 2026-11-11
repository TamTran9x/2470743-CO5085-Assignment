// Báo cáo Bài tập lớn A1 — CO5085. Biên dịch: python -m scripts.build_report
#let summary = csv("../results/summary.csv", row-type: dictionary)
#let eda = json("../results/eda.json")
#let analysis = json("../results/analysis.json")

// số thập phân theo kiểu Việt Nam (dấu phẩy)
#let fmt(x, d: 2) = {
  let s = str(calc.round(float(x), digits: d))
  if d == 0 { return s }
  if not s.contains(".") { s = s + "." }
  (s + "0" * (d - s.split(".").at(1).len())).replace(".", ",")
}
#let arch-label = (resnet50: "ResNet-50", deit_s: "DeiT-S")
#let mode-label = (scratch: "From scratch", linear: "Freeze backbone", partial: "Freeze một phần", full: "Full fine-tune")
#let run(name) = summary.find(r => r.run == name)
#let acc(name) = fmt(run(name).test_acc)
#let pct(x, d: 1) = fmt(100 * float(x), d: d)
#let gain(a, b) = fmt(float(run(a).test_acc) - float(run(b).test_acc))
#let ratio(a, b, key) = fmt(float(run(a).at(key)) / float(run(b).at(key)), d: 1)
#let pval(p) = if p < 0.001 { "< 0,001" } else { fmt(p, d: 2) }
#let A = analysis.runs
#let mc = analysis.mcnemar_cnn_vs_vit
#let ov = analysis.overlap_best_cnn_vs_vit

#set document(title: "A1 — CNN vs Transformer trên CIFAR-100", author: "Trần Hoài Tâm")
#set page(paper: "a4", margin: (x: 2.3cm, y: 2.4cm), numbering: "1")
#set text(font: "Times New Roman", size: 11.5pt, lang: "vi")
#set par(justify: true, leading: 0.7em, spacing: 1.1em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => { v(0.6em); it; v(0.3em) }
#show figure.caption: set text(size: 10pt)
#set table(stroke: (x, y) => (top: if y <= 1 { 0.6pt } else { 0pt }, bottom: 0.6pt + luma(200)), inset: (x: 5pt, y: 4pt))
#show table: set text(size: 9.5pt)
#show link: set text(fill: rgb("#1c5cab"))
#show raw.where(block: true): set text(size: 8.5pt)

// ------------------------------------------------------------------ title page
#page(numbering: none)[
  #align(center)[
    #v(1.5cm)
    #text(size: 16pt, weight: "bold")[ĐẠI HỌC BÁCH KHOA – ĐHQG TP.HCM] \
    #text(size: 12pt)[Khoa Khoa học và Kỹ thuật Máy tính]
    #v(3cm)
    #line(length: 100%, stroke: 0.6pt)
    #text(size: 22pt)[Bài tập lớn A1] \
    #v(0.2em)
    #text(size: 17pt)[CNN vs Transformer trên CIFAR-100: \ from-scratch, pretrained và các kiểu đóng băng]
    #v(0.3em)
    #text(size: 12pt)[Học sâu và ứng dụng trong thị giác máy tính – CO5085]
    #line(length: 100%, stroke: 0.6pt)
    #v(2.5cm)
    #table(columns: (auto, auto), stroke: none, align: left, inset: 5pt,
      [*Học kỳ:*], [HK261 (2026–2027)],
      [*Giảng viên:*], [Lê Thành Sách, Nguyễn Quốc Minh],
      [*Học viên:*], [Trần Hoài Tâm – MSHV 2470743 (làm cá nhân)],
      [*Mã nguồn:*], [#link("https://github.com/TamTran9x/2470743-CO5085-Assignment/tree/main/A1")[github.com/TamTran9x/2470743-CO5085-Assignment/A1]],
      [*Trang A1:*], [#link("https://tamtran9x.github.io/2470743-CO5085-Assignment/A1/")[tamtran9x.github.io/2470743-CO5085-Assignment/A1]],
    )
    #v(1fr)
    TP. Hồ Chí Minh, tháng 11/2026
  ]
]

#outline(depth: 2, indent: auto)
#pagebreak()

// ------------------------------------------------------------------ 1
= Giới thiệu

Bài tập so sánh hai họ kiến trúc phân loại ảnh, mạng tích chập (CNN) và Vision Transformer (ViT), trên một tập ảnh đủ lớn và khó là *CIFAR-100*. Mỗi kiến trúc được chạy ở bốn chế độ huấn luyện: (i) huấn luyện từ đầu (from scratch); (ii) dùng trọng số pretrained trên ImageNet-1k và đóng băng toàn bộ backbone, chỉ học lớp phân loại (linear probe); (iii) pretrained, đóng băng nửa đầu mạng; (iv) pretrained, fine-tune toàn bộ. Tổng cộng 8 cấu hình chạy trên cùng một split, cùng pipeline dữ liệu và cùng cách đo. Báo cáo trình bày lựa chọn dataset và protocol (mốc A1.1), sau đó là kết quả, chi phí, phân tích lỗi và thảo luận (mốc A1.2).

Mọi con số trong báo cáo được sinh tự động từ `results/` bởi mã nguồn trong thư mục A1. Bảng biểu được đọc trực tiếp từ `results/summary.csv` khi biên dịch, nên báo cáo luôn khớp với lần chạy thật.

// ------------------------------------------------------------------ 2
= Tập dữ liệu (A1.1)

== Lựa chọn và giấy phép

*CIFAR-100* (Krizhevsky, 2009) gồm #eda.n_train_official + #eda.n_test = 60.000 ảnh màu 32×32, thuộc 100 lớp (fine label), được gom thành 20 superclass (coarse label), mỗi superclass có đúng 5 lớp. Mỗi lớp có 500 ảnh train và 100 ảnh test, tức dữ liệu cân bằng hoàn toàn.

- *Nguồn:* bản gốc tại `cs.toronto.edu/~kriz/cifar.html`. Mã nguồn tải bản mirror chính thức `uoft-cs/cifar100` trên Hugging Face Hub (định dạng parquet, cùng ảnh và cùng split train/test), vì máy chủ gốc tải rất chậm (\~10 KB/s khi thực hiện).
- *Giấy phép:* tác giả không công bố license chính thức; dataset được cung cấp công khai cho nghiên cứu và giảng dạy, là một trong các benchmark phổ biến nhất (dataset card trên Hugging Face ghi `license: unknown`). Ảnh được trích từ tập 80 Million Tiny Images; tập mẹ này đã bị rút lại năm 2020, nhưng CIFAR-10/100 vẫn được duy trì và dùng rộng rãi. Bài tập chỉ dùng dữ liệu cho mục đích học tập.
- *Lý do chọn:* (1) có trong danh sách gợi ý của đề và không thuộc nhóm bị cấm (MNIST, Fashion-MNIST, CIFAR-10); (2) 100 lớp, nhiều lớp rất gần nhau trong cùng superclass (maple/oak/willow tree, boy/girl/man/woman...), khó hơn nhiều so với CIFAR-10; (3) ảnh 32×32 khác xa phân phối ImageNet 224×224, tạo ra *domain gap* để thấy rõ khác biệt giữa đóng băng và fine-tune; (4) toàn bộ tập nằm gọn trong bộ nhớ GPU nên chạy đủ 8 cấu hình ở độ phân giải 224 trên GPU laptop 6 GB.

== Chia tập (split)

#figure(
  table(columns: 4, align: (left, right, right, left),
    table.header([*Tập*], [*Số ảnh*], [*Ảnh/lớp*], [*Vai trò*]),
    [Train], [#eda.n_train], [450], [Huấn luyện],
    [Validation], [#eda.n_val], [50], [Chọn epoch tốt nhất, so sánh khi huấn luyện],
    [Test (chính thức)], [#eda.n_test], [100], [Chỉ đánh giá một lần cho checkpoint đã chọn],
  ),
  caption: [Chia tập. Validation được tách *phân tầng* (50 ảnh/lớp, seed 0) từ 50.000 ảnh train chính thức.],
) <tab-split>

Tập test chính thức được giữ riêng: mọi lựa chọn (epoch tốt nhất, chế độ tốt nhất của từng họ) đều dựa trên validation. Split cố định, giống nhau cho cả 8 cấu hình, và tái tạo được bằng `stratified_split(seed=0)` trong `src/data.py`.

== Phân tích khám phá (EDA)

#figure(image("../figures/eda_samples.png", width: 88%),
  caption: [Mỗi lớp một ảnh ngẫu nhiên từ tập train. Ảnh có độ phân giải rất thấp; nhiều lớp chỉ phân biệt được qua chi tiết nhỏ hoặc ngữ cảnh.]) <fig-samples>

- *Cân bằng lớp:* train có đúng #eda.train_per_class.at(0) ảnh/lớp, validation #eda.val_per_class.at(0) ảnh/lớp, test #eda.test_per_class.at(0) ảnh/lớp. Vì vậy accuracy và macro-F1 gần như trùng nhau, và không cần trọng số lớp hay oversampling.
- *Thống kê pixel* (train, thang [0, 1]): mean = (#eda.channel_mean.map(x => fmt(x, d: 3)).join("; ")), std = (#eda.channel_std.map(x => fmt(x, d: 3)).join("; ")). Giá trị này khác thống kê ImageNet (mean 0,485/0,456/0,406, std 0,229/0,224/0,225), nhưng mọi cấu hình vẫn chuẩn hoá bằng thống kê ImageNet để khớp với các backbone pretrained. Mô hình from-scratch cũng dùng cùng chuẩn hoá cho công bằng; chênh lệch nhỏ này không ảnh hưởng đáng kể vì mạng có BatchNorm/LayerNorm.
- *Cấu trúc phân cấp:* 20 superclass × 5 lớp (@tab-super). Nhiều lỗi được dự đoán sẽ rơi vào lớp "anh em" trong cùng superclass, điều này được kiểm chứng ở Mục 6.

#figure(image("../figures/eda_pixel_hist.png", width: 75%),
  caption: [Phân bố cường độ pixel theo kênh. Đỉnh ở 0 và 255 đến từ nền đen/trắng (ảnh sản phẩm, ảnh cắt nền).])

#figure(
  table(columns: (auto, 1fr), align: (left, left),
    table.header([*Superclass*], [*5 lớp con*]),
    ..eda.superclasses.pairs().map(((k, v)) => (raw(k), v.map(c => c.replace("_", " ")).join(", "))).flatten(),
  ),
  caption: [20 superclass của CIFAR-100.],
) <tab-super>

// ------------------------------------------------------------------ 3
= DataLoader và augmentation

*DataLoader đặt trên GPU.* Toàn bộ 60.000 ảnh uint8 (\~180 MB) được nạp lên GPU một lần (`GPULoader` trong `src/data.py`). Mỗi batch được cắt chỉ số, augment, resize và chuẩn hoá ngay trên GPU. Trên Windows, `DataLoader` đa tiến trình khởi động chậm và giải nén ảnh trên CPU dễ thành nút thắt; cách làm này giữ GPU luôn bận (xem thông lượng ở @tab-main).

*Pipeline* (giống hệt cho cả 8 cấu hình):
+ *Train:* random crop 32×32 với zero-padding 4 pixel (độc lập từng ảnh) → lật ngang ngẫu nhiên (p = 0,5).
+ Resize bilinear 32 → *224×224*, đúng độ phân giải mà cả hai backbone pretrained được huấn luyện (DeiT-S patch 16 → 14×14 = 196 token, không cần nội suy position embedding).
+ Chuẩn hoá theo mean/std ImageNet; với CNN, tensor được chuyển sang `channels_last` để tận dụng Tensor Core.
+ *Val/Test:* chỉ resize + chuẩn hoá, không augment.

Augmentation được giữ ở mức cổ điển cho CIFAR (crop + flip) cộng với label smoothing 0,1, và áp dụng *như nhau* cho mọi cấu hình, để khác biệt giữa các cấu hình chỉ đến từ kiến trúc và chế độ huấn luyện. Các kỹ thuật mạnh hơn (RandAugment, Mixup/CutMix) có lợi cho ViT from scratch nhưng sẽ làm lẫn biến số; phần này được bàn ở Mục 8.

#figure(image("../figures/aug_examples.png", width: 78%),
  caption: [Ảnh gốc, ba lần crop+flip ngẫu nhiên, và đầu vào 224×224 thực tế của mô hình (đã khử chuẩn hoá để hiển thị).])

// ------------------------------------------------------------------ 4
= Mô hình và chế độ huấn luyện

== Hai kiến trúc

Hai mô hình được chọn có kích thước và chi phí tương đương, và đều chỉ được pretrain trên *ImageNet-1k* (không dùng ImageNet-21k), để so sánh công bằng:

#figure(
  table(columns: 6, align: (left, left, right, right, right, left),
    table.header([*Họ*], [*Mô hình (timm)*], [*Tham số*], [*GMACs \@224*], [*IN-1k top-1*], [*Đặc điểm*]),
    [CNN], [`resnet50.tv2_in1k`], [#fmt(run("resnet50_full").total_M) M], [#fmt(run("resnet50_full").gmacs)], [80,9%], [4 stage bottleneck, BatchNorm],
    [Transformer], [`deit_small_patch16_224.fb_in1k`], [#fmt(run("deit_s_full").total_M) M], [#fmt(run("deit_s_full").gmacs)], [79,9%], [12 block, dim 384, 6 head, LayerNorm],
  ),
  caption: [Hai kiến trúc (số tham số tính với head 100 lớp; GMACs đo bằng `FlopCounterMode` của PyTorch).],
) <tab-models>

- *ResNet-50* (He và cộng sự, 2016): trọng số torchvision V2, huấn luyện 600 epoch với TrivialAugment, Mixup/CutMix, label smoothing và EMA.
- *DeiT-S* (Touvron và cộng sự, 2021): ViT-Small/16 huấn luyện 300 epoch trên ImageNet-1k với RandAugment, Mixup/CutMix và random erasing (bản không distillation). Hai công thức pretrain đều "hiện đại" và có độ chính xác ImageNet gần nhau.

Head phân loại 1000 lớp được thay bằng một lớp tuyến tính 100 lớp khởi tạo ngẫu nhiên.

== Bốn chế độ huấn luyện

#figure(
  table(columns: 4, align: (left, left, left, left),
    table.header([*Chế độ*], [*Khởi tạo*], [*ResNet-50: phần đóng băng*], [*DeiT-S: phần đóng băng*]),
    [From scratch], [Ngẫu nhiên], [–], [–],
    [Freeze backbone], [IN-1k], [Toàn bộ trừ `fc`], [Toàn bộ trừ `head`],
    [Freeze một phần], [IN-1k], [stem + `layer1`–`layer2` (2/4 stage)], [patch-embed, pos-embed, cls, block 0–5 (6/12 block)],
    [Full fine-tune], [IN-1k], [–], [–],
  ),
  caption: [Định nghĩa các chế độ. "Freeze một phần" đóng băng nửa đầu theo độ sâu của mỗi mạng.],
) <tab-modes>

Phần bị đóng băng luôn ở chế độ `eval()` khi huấn luyện: BatchNorm của ResNet dùng running statistics của ImageNet và không cập nhật. Autograd không lưu activation và không tính gradient cho phần đóng băng, nên các chế độ freeze vừa nhanh hơn vừa tốn ít bộ nhớ hơn (@tab-main). Lưu ý: nửa đầu ResNet-50 chỉ chứa khoảng 6% tham số nhưng chiếm khoảng một nửa FLOPs, trong khi nửa đầu DeiT-S chứa khoảng một nửa tham số. Đây là khác biệt cấu trúc giữa hai họ (CNN dồn tham số về các stage sâu, ViT chia đều cho các block).

// ------------------------------------------------------------------ 5
= Protocol thí nghiệm

#figure(
  table(columns: 5, align: (left, center, center, center, center),
    table.header([*Siêu tham số*], [*From scratch*], [*Freeze backbone*], [*Freeze một phần*], [*Full fine-tune*]),
    [Epoch], [30], [10], [10], [10],
    [LR backbone], [5e-4], [–], [1e-4], [1e-4],
    [LR head], [5e-4], [1e-3], [1e-3], [1e-3],
    [Warmup (epoch)], [3], [1], [1], [1],
    table.cell(colspan: 5, align: left)[*Chung:* AdamW (β = 0,9/0,999), weight decay 0,05 (không áp cho bias, norm, pos-embed, cls token), batch 64, cosine decay theo từng bước sau warmup tuyến tính, label smoothing 0,1, gradient clipping 1,0, mixed precision bfloat16, ảnh 224×224, seed 42.],
  ),
  caption: [Siêu tham số. Hai kiến trúc dùng *chung* một bộ siêu tham số cho mỗi chế độ.],
) <tab-hparams>

- *Lựa chọn epoch:* mô hình pretrained hội tụ nhanh nên 10 epoch là đủ (đường val gần như phẳng ở cuối, xem @fig-curves). From scratch được 30 epoch, gấp ba. Với lịch cosine, val accuracy của from scratch cũng phẳng ở cuối, nhưng train accuracy đã gần 100%: mô hình bị giới hạn bởi overfitting nhiều hơn là bởi số epoch. Con số from scratch vì vậy là cận dưới so với một công thức huấn luyện dài và augmentation mạnh (Mục 8). Learning rate backbone khi fine-tune nhỏ hơn head 10 lần để không phá đặc trưng pretrained, trong khi head mới khởi tạo vẫn học nhanh.
- *Chọn mô hình:* lưu checkpoint có validation accuracy cao nhất, sau đó đánh giá *một lần* trên test.
- *Metric:* top-1 accuracy, top-5 accuracy, macro-F1, accuracy cấp superclass (dự đoán đúng superclass), confusion matrix 100×100.
- *Chi phí:* số tham số tổng/trainable, GMACs, thời gian train (tổng thời gian các epoch, không tính validation), thông lượng train/inference (ảnh/s), bộ nhớ GPU đỉnh khi train (`torch.cuda.max_memory_allocated`).
- *Phần cứng:* NVIDIA RTX 3050 Laptop 6 GB (65 W), Intel i7-13620H, 16 GB RAM, Windows 11; PyTorch 2.5.1 + CUDA 12.4, timm 1.0.30.

// ------------------------------------------------------------------ 6
= Kết quả

#let main-rows = summary.map(r => (
  arch-label.at(r.arch), mode-label.at(r.mode), fmt(r.test_acc), fmt(r.test_top5), fmt(r.macro_f1), fmt(r.superclass_acc),
  fmt(r.trainable_M), fmt(r.train_min, d: 1), fmt(r.train_img_s, d: 0), fmt(r.peak_mem_gb),
))
#figure(
  {
    show table: set text(size: 8.5pt)
    table(columns: (auto, auto) + (1fr,) * 8, align: (left, left) + (right,) * 8,
      table.header([*Model*], [*Chế độ*], [*Top-1*], [*Top-5*], [*Macro-F1*], [*Super-class*], [*Trainable (M)*], [*Train (phút)*], [*Train (ảnh/s)*], [*Mem (GB)*]),
      ..main-rows.flatten(),
    )
  },
  caption: [Kết quả trên tập test CIFAR-100 (10.000 ảnh, %), checkpoint chọn theo validation. Chi phí đo trên RTX 3050 Laptop, batch 64, 224×224.],
) <tab-main>

#figure(image("../figures/fig_test_acc.png", width: 92%), caption: [Top-1 accuracy trên test theo chế độ huấn luyện.]) <fig-acc>

#figure(image("../figures/fig_curves.png", width: 100%),
  caption: [Đường học: validation accuracy (trên) và train accuracy trên batch đã augment (dưới). Trục x của from scratch dài gấp ba.]) <fig-curves>

#figure(image("../figures/fig_cost.png", width: 100%), caption: [Accuracy theo thời gian huấn luyện và theo bộ nhớ GPU đỉnh. Màu là họ mô hình, hình dạng điểm là chế độ huấn luyện. Số tham số trainable có trong @tab-main.]) <fig-cost>

== Nhận xét chính

- *Pretrained + full fine-tune tốt nhất ở cả hai họ:* DeiT-S đạt #acc("deit_s_full")%, ResNet-50 đạt #acc("resnet50_full")%. Đây cũng là chế độ có validation accuracy cao nhất của mỗi họ, nên chọn theo validation hay theo test đều cho cùng kết luận.
- *Thứ hạng hai họ đảo chiều.* From scratch, ResNet-50 hơn DeiT-S #gain("resnet50_scratch", "deit_s_scratch") điểm; khi full fine-tune, DeiT-S lại hơn ResNet-50 #gain("deit_s_full", "resnet50_full") điểm.
- *Pretraining quan trọng với ViT hơn nhiều so với CNN.* So với from scratch, full fine-tune tăng #gain("resnet50_full", "resnet50_scratch") điểm cho ResNet-50 nhưng tăng tới #gain("deit_s_full", "deit_s_scratch") điểm cho DeiT-S, dù from scratch được gấp ba số epoch và thời gian train dài gấp \~3 lần.
- *Chỉ một lớp tuyến tính trên đặc trưng pretrained đã thắng from scratch:* freeze backbone của DeiT-S (#fmt(run("deit_s_linear").trainable_M) M tham số trainable, #fmt(run("deit_s_linear").train_min, d: 1) phút) đạt #acc("deit_s_linear")%, cao hơn ResNet-50 from scratch (#acc("resnet50_scratch")%, #fmt(run("resnet50_scratch").train_min, d: 1) phút).
- *Freeze backbone* rẻ nhất nhưng thấp hơn full fine-tune #gain("resnet50_full", "resnet50_linear") (ResNet-50) và #gain("deit_s_full", "deit_s_linear") điểm (DeiT-S). *Freeze một phần* gần như bằng full fine-tune với ResNet-50 (kém #gain("resnet50_full", "resnet50_partial") điểm) nhưng kém rõ hơn với DeiT-S (#gain("deit_s_full", "deit_s_partial") điểm).
- Dữ liệu cân bằng nên macro-F1 bám sát top-1 (chênh không quá 0,5 điểm); top-5 và superclass accuracy giữ đúng thứ hạng của top-1.

== Chênh lệch CNN–ViT có ý nghĩa thống kê không?

Mỗi cấu hình chỉ chạy một seed, nên cần kiểm tra chênh lệch giữa hai họ có vượt mức nhiễu hay không. Hai mô hình được đánh giá trên *cùng* 10.000 ảnh test, nên dùng kiểm định McNemar (bản chính xác, dạng nhị thức) trên các ảnh mà đúng một trong hai mô hình đoán đúng.

#figure(
  table(columns: 5, align: (left, right, right, right, right),
    table.header([*Chế độ*], [*Chỉ ResNet-50 đúng*], [*Chỉ DeiT-S đúng*], [*DeiT-S − ResNet-50*], [*p*]),
    ..("scratch", "linear", "partial", "full").map(m => (
      mode-label.at(m), str(mc.at(m).only_a), str(mc.at(m).only_b),
      gain("deit_s_" + m, "resnet50_" + m), pval(mc.at(m).p_value),
    )).flatten(),
  ),
  caption: [Kiểm định McNemar giữa ResNet-50 và DeiT-S trong cùng chế độ (số ảnh test; chênh lệch tính bằng điểm %).],
) <tab-mcnemar>

Chênh lệch ở *from scratch* và *full fine-tune* có ý nghĩa rất mạnh (p < 0,001). Ở *freeze backbone* (p = #pval(mc.linear.p_value)) và *freeze một phần* (p = #pval(mc.partial.p_value)), hai họ *không khác nhau có ý nghĩa*: với các chế độ đóng băng, đặc trưng ImageNet của hai backbone hữu ích gần như ngang nhau cho CIFAR-100.

== Phân tích lỗi

Phần này phân tích chế độ tốt nhất (theo validation) của mỗi họ: full fine-tune.

=== Lỗi theo superclass

#figure(image("../figures/fig_superclass_confusion.png", width: 100%),
  caption: [Confusion matrix cấp superclass. Ô màu: % ảnh của superclass thật bị đoán sang superclass khác. Số trên đường chéo: accuracy của superclass đó.]) <fig-super>

- Accuracy cấp superclass đạt #fmt(run("resnet50_full").superclass_acc)% (ResNet-50) và #fmt(run("deit_s_full").superclass_acc)% (DeiT-S), cao hơn top-1 khoảng 6–7 điểm.
- *Hơn một nửa số lỗi là nhầm giữa các lớp "anh em" trong cùng superclass:* #pct(A.deit_s_full.within_superclass_error_share)% lỗi của DeiT-S và #pct(A.resnet50_full.within_superclass_error_share)% lỗi của ResNet-50. Nếu lỗi phân bố ngẫu nhiên thì tỉ lệ này chỉ là 4/99 ≈ 4%. Tỉ lệ này *tăng theo chất lượng mô hình*: #pct(A.deit_s_scratch.within_superclass_error_share)% (DeiT-S scratch), #pct(A.resnet50_scratch.within_superclass_error_share)% (ResNet-50 scratch), #pct(A.resnet50_linear.within_superclass_error_share)–#pct(A.deit_s_linear.within_superclass_error_share)% (freeze backbone), #pct(A.deit_s_partial.within_superclass_error_share)–#pct(A.deit_s_full.within_superclass_error_share)% (freeze một phần, full fine-tune). Mô hình càng tốt thì lỗi còn lại càng "hợp lý" về ngữ nghĩa.
- Các cặp superclass hay nhầm nhất của hai mô hình gần như trùng nhau: *fish ↔ aquatic mammals* (cá mập/cá đuối với cá voi/cá heo), *vehicles 1 ↔ vehicles 2* (bus, streetcar, pickup truck, tractor... bị chia tùy ý vào hai nhóm), *medium → small mammals*, *reptiles → non-insect invertebrates* (snake → worm) và *large natural outdoor scenes → trees* (forest → cây). Đây là những cặp gần nhau về hình dạng hoặc do cách chia superclass của CIFAR-100 không hoàn toàn theo thị giác.
- DeiT-S from scratch có kiểu lỗi khác hẳn: nhầm vehicles 1 → vehicles 2 tới #fmt(A.deit_s_scratch.top_superclass_confusions.at(0).pct, d: 1)%, medium mammals → large carnivores #fmt(A.deit_s_scratch.top_superclass_confusions.at(2).pct, d: 1)%, food containers → household electrical devices #fmt(A.deit_s_scratch.top_superclass_confusions.at(3).pct, d: 1)%. Mô hình mới chỉ học được tín hiệu thô (màu, bố cục) chứ chưa học được cấu trúc ngữ nghĩa.

=== Lỗi theo lớp

#figure(image("../figures/fig_confusion_full.png", width: 80%),
  caption: [Confusion matrix 100 lớp của DeiT-S full fine-tune, chỉ hiển thị phần lỗi (đường chéo để trống), lớp được nhóm theo superclass. Các khối đậm nằm trong các ô vuông trên đường chéo.]) <fig-cm>

Lỗi tập trung thành vài khối rõ rệt trên đường chéo khối (@fig-cm):
- *people* (baby, boy, girl, man, woman): lớp khó nhất của DeiT-S là girl (#pct(A.deit_s_full.hardest_classes.at(0).acc, d: 0)%), tiếp theo là boy, man. Ở 32×32 khuôn mặt chỉ còn vài pixel, nên tuổi và giới tính phải đoán qua tóc, quần áo, bối cảnh.
- *trees* (maple, oak, palm, pine, willow): maple ↔ oak là cặp nhầm nhiều nhất của cả hai mô hình (DeiT-S: #A.deit_s_full.top_confusions.at(0).count + #A.deit_s_full.top_confusions.at(1).count ảnh; ResNet-50: #A.resnet50_full.top_confusions.at(0).count + #A.resnet50_full.top_confusions.at(1).count ảnh). Ở độ phân giải này hình dạng lá gần như mất hết, chỉ còn tán cây.
- *aquatic mammals* (otter ↔ seal) và *small/medium mammals* (shrew, possum, squirrel): động vật nhỏ màu nâu trên nền tự nhiên.
- Các lớp dễ nhất có hình dạng hoặc màu sắc đặc trưng: #A.deit_s_full.easiest_classes.slice(0, 3).map(c => c.class.replace("_", " ") + " (" + pct(c.acc, d: 0) + "%)").join(", ").

#figure(image("../figures/fig_misclassified.png", width: 100%),
  caption: [24 lỗi có độ tự tin cao nhất của DeiT-S full fine-tune.]) <fig-wrong>

Xem các lỗi tự tin nhất (@fig-wrong), một phần đáng kể *không hẳn là lỗi của mô hình*: ảnh nhãn "plain" có một cái cây lớn ở giữa (đoán oak tree), ảnh nhãn "girl"/"boy" là trẻ nhỏ (đoán baby), ảnh "bowl" và "cup" rất giống nhau, nhiều ảnh cây chỉ là một tán lá mờ. Đây là giới hạn của nhãn đơn và độ phân giải 32×32, không cải thiện được chỉ bằng mô hình lớn hơn.

=== CNN và ViT khác nhau ở lớp nào?

#figure(image("../figures/fig_per_class_diff.png", width: 72%),
  caption: [Chênh lệch accuracy theo lớp giữa DeiT-S và ResNet-50 (full fine-tune): 12 lớp mỗi bên có chênh lệch lớn nhất.]) <fig-diff>

DeiT-S thắng nhiều lớp với cách biệt lớn (willow tree +13, possum +12, lobster +10, kangaroo, house, ray +9 điểm), trong khi các lớp ResNet-50 thắng chỉ cách biệt tối đa 5 điểm (bed, tiger, crab, bowl, sweet pepper, orange). Các lớp DeiT-S thắng phần nhiều cần *hình dạng tổng thể* (dáng rủ của willow, kangaroo, ray, lobster); vài lớp ResNet-50 thắng có *kết cấu, màu sắc* đặc trưng (vằn tiger, orange, sweet pepper). Điều này khớp với nhận định CNN thiên về texture và ViT thiên về shape (Geirhos và cộng sự, 2019; Naseer và cộng sự, 2021). Tuy vậy, mỗi lớp chỉ có 100 ảnh test, nên độ lệch chuẩn của hiệu accuracy một lớp vào khoảng 5 điểm. Chỉ các chênh lệch từ \~10 điểm trở lên mới đáng tin; phần còn lại nên xem là xu hướng.

=== Lỗi của hai họ bổ sung cho nhau

Trên tập test, hai mô hình full fine-tune cùng đúng #pct(ov.both_correct)%, cùng sai #pct(ov.both_wrong)%, chỉ ResNet-50 đúng #pct(ov.only_a)% và chỉ DeiT-S đúng #pct(ov.only_b)%. Lấy trung bình xác suất softmax của hai mô hình (ensemble, không huấn luyện thêm) đạt *#pct(ov.ensemble_acc, d: 2)%*, cao hơn mô hình tốt nhất #fmt(100 * float(ov.ensemble_acc) - float(run("deit_s_full").test_acc)) điểm. Cận trên nếu luôn chọn đúng mô hình (oracle) là #pct(1 - float(ov.both_wrong))%. Hai họ kiến trúc mắc lỗi đủ khác nhau để kết hợp có lợi.

// ------------------------------------------------------------------ 7
= Thảo luận

== Pretrained so với from scratch

Với 45.000 ảnh train, cả hai mô hình from scratch đều overfit nặng (@fig-curves): ResNet-50 có train accuracy #fmt(run("resnet50_scratch").final_train_acc, d: 1)% nhưng val chỉ \~71%; DeiT-S có train #fmt(run("deit_s_scratch").final_train_acc, d: 1)%, val \~47%. Ảnh 32×32 phóng lên 224×224 không thêm thông tin, nên mô hình \~22–24M tham số dễ dàng thuộc lòng tập train. Pretraining trên ImageNet-1k (1,28 triệu ảnh) cung cấp đặc trưng tổng quát mà 45.000 ảnh không đủ để học. Kết quả: full fine-tune hơn from scratch #gain("resnet50_full", "resnet50_scratch")–#gain("deit_s_full", "deit_s_scratch") điểm, với chỉ một phần ba số epoch, và val accuracy vượt 75% ngay sau epoch đầu tiên.

== Khi nào ViT cần pretraining?

*Trong thí nghiệm này: luôn cần.* DeiT-S from scratch chỉ đạt #acc("deit_s_scratch")%, thấp hơn ResNet-50 from scratch #gain("resnet50_scratch", "deit_s_scratch") điểm. Đường học cho thấy nguyên nhân không phải thiếu epoch: từ epoch 18, val accuracy của DeiT-S gần như đứng yên quanh 46–47%, trong khi train accuracy tăng từ 75% lên 99,7%. Mô hình đang học thuộc dữ liệu chứ không tổng quát hoá thêm.

Lý do nằm ở *inductive bias*. Convolution có sẵn tính cục bộ và tính bất biến tịnh tiến, nên ResNet-50 học được đặc trưng cục bộ hữu ích ngay cả với ít dữ liệu. Self-attention không có các giả định này: DeiT-S phải tự học từ dữ liệu rằng các pixel gần nhau thì liên quan, và với 45.000 ảnh nó học được rất ít. Điều này khớp với các nghiên cứu trước. ViT chỉ vượt ResNet khi được pretrain trên 14–300 triệu ảnh (Dosovitskiy và cộng sự, 2021). DeiT huấn luyện được ViT trên ImageNet-1k nhờ 300 epoch cùng augmentation và regularization rất mạnh (RandAugment, Mixup, CutMix, random erasing, stochastic depth) (Touvron và cộng sự, 2021). Protocol from scratch của bài này không có các yếu tố đó.

*Khi đã được pretrain*, ViT không còn bất lợi: cùng dữ liệu pretrain ImageNet-1k với ResNet-50, DeiT-S full fine-tune tốt hơn #gain("deit_s_full", "resnet50_full") điểm (có ý nghĩa thống kê, @tab-mcnemar). Tóm lại, ViT cần pretraining khi dữ liệu nhỏ (cỡ vài chục nghìn ảnh) và công thức huấn luyện đơn giản; ngược lại, nếu có backbone pretrained thì ViT cạnh tranh tốt hoặc tốt hơn CNN cùng cỡ.

== Khi nào đóng băng có lợi?

*Freeze backbone (linear probe)* nhanh hơn full fine-tune \~#ratio("resnet50_full", "resnet50_linear", "train_min")–#ratio("deit_s_full", "deit_s_linear", "train_min") lần, dùng ít hơn 5 lần bộ nhớ, và chỉ phải lưu một head \~0,04–0,2 M tham số cho mỗi tác vụ. Đổi lại, accuracy thấp hơn 11–14 điểm. Ở đây mô hình *underfit chứ không overfit*: train accuracy trên ảnh đã augment chỉ #fmt(run("resnet50_linear").final_train_acc, d: 1)% (ResNet-50) và #fmt(run("deit_s_linear").final_train_acc, d: 1)% (DeiT-S), sát val. Đặc trưng lớp cuối của ImageNet không tách tuyến tính được hết 100 lớp CIFAR. Nguyên nhân là *domain gap*: ảnh CIFAR phóng to bị mờ, khác xa ảnh ImageNet sắc nét, và các đặc trưng sâu đã chuyên biệt cho 1000 lớp ImageNet. Freeze backbone có lợi khi tài nguyên rất hạn chế, khi cần baseline nhanh, khi nhiều tác vụ dùng chung một backbone, hoặc khi dữ liệu quá ít để fine-tune mà không overfit.

*Freeze một phần là điểm cân bằng tốt nhất cho ResNet-50:* chỉ kém full fine-tune #gain("resnet50_full", "resnet50_partial") điểm nhưng thời gian train giảm #fmt(100 * (1 - float(run("resnet50_partial").train_min) / float(run("resnet50_full").train_min)), d: 0)% và bộ nhớ giảm #fmt(100 * (1 - float(run("resnet50_partial").peak_mem_gb) / float(run("resnet50_full").peak_mem_gb)), d: 0)%. Hai stage đầu của CNN học các đặc trưng chung như cạnh, góc, kết cấu, vốn chuyển giao tốt giữa các tập dữ liệu (Yosinski và cộng sự, 2014). Các stage này lại chiếm phần lớn activation độ phân giải cao, nên bỏ qua backward ở đó tiết kiệm nhiều chi phí. Ngoài ra, layer3–layer4 vẫn được train và chứa #fmt(run("resnet50_partial").trainable_pct, d: 0)% tham số.

Với DeiT-S, freeze một phần kém full fine-tune #gain("deit_s_full", "deit_s_partial") điểm, mất nhiều hơn hẳn. Khi đóng băng, patch embedding và 6 block đầu (một nửa số tham số) giữ nguyên. Patch embedding là một phép chiếu tuyến tính học trên patch ImageNet sắc nét, và ở ViT mọi token đi qua cùng các block đó, nên phần đóng băng không thể thích nghi với patch mờ của ảnh CIFAR phóng to. Vì vậy, với ViT trên dữ liệu khác miền, nên fine-tune toàn bộ hoặc chỉ đóng băng rất ít block. Một lựa chọn trung gian là giảm learning rate dần theo độ sâu (layer-wise LR decay) thay vì đóng băng cứng.

*Full fine-tune* luôn cho accuracy cao nhất. Nó tốn nhiều chi phí nhất, nhưng trên GPU 6 GB vẫn vừa (#fmt(run("resnet50_full").peak_mem_gb) GB với ResNet-50). Khi tài nguyên cho phép, đây là lựa chọn mặc định.

== Chi phí

- *Thời gian và bộ nhớ:* trên GPU này, DeiT-S train nhanh hơn ResNet-50 (#fmt(run("deit_s_full").train_img_s, d: 0) so với #fmt(run("resnet50_full").train_img_s, d: 0) ảnh/s khi full fine-tune), suy luận nhanh hơn (#fmt(run("deit_s_full").infer_img_s, d: 0) so với #fmt(run("resnet50_full").infer_img_s, d: 0) ảnh/s) và tốn ít bộ nhớ hơn (#fmt(run("deit_s_full").peak_mem_gb) so với #fmt(run("resnet50_full").peak_mem_gb) GB), dù GMACs cao hơn một chút (#fmt(run("deit_s_full").gmacs) so với #fmt(run("resnet50_full").gmacs)). Các phép nhân ma trận lớn của Transformer tận dụng Tensor Core với bfloat16 tốt hơn convolution và BatchNorm. Kết luận này phụ thuộc phần cứng và phiên bản thư viện.
- *Đường biên Pareto* (accuracy cao nhất ở mỗi mức thời gian, @fig-cost) gồm freeze backbone, freeze một phần và DeiT-S full fine-tune. ResNet-50 full fine-tune bị DeiT-S full fine-tune vượt ở cả accuracy lẫn thời gian, còn hai cấu hình from scratch tốn nhiều thời gian nhất mà accuracy thấp nhất.

#figure(
  table(columns: (auto, 1fr), align: (left, left),
    table.header([*Tình huống*], [*Lựa chọn đề xuất (theo kết quả trên CIFAR-100)*]),
    [Cần accuracy cao nhất], [DeiT-S pretrained, full fine-tune (#acc("deit_s_full")%); thêm ensemble với ResNet-50 nếu chấp nhận gấp đôi chi phí suy luận (#pct(ov.ensemble_acc, d: 2)%)],
    [Ngân sách train/bộ nhớ vừa phải], [ResNet-50 hoặc DeiT-S freeze một phần (\~85,4%, \~19 phút, \~1,3–1,5 GB)],
    [Rất ít tài nguyên, nhiều tác vụ chung backbone], [Freeze backbone (\~75%, \~10 phút, < 0,7 GB)],
    [Không có trọng số pretrained, dữ liệu nhỏ], [CNN (ResNet-50 from scratch, #acc("resnet50_scratch")%); tránh ViT from scratch nếu không có augmentation mạnh và lịch huấn luyện dài],
  ),
  caption: [Tóm tắt khuyến nghị.],
) <tab-reco>

// ------------------------------------------------------------------ 8
= Hạn chế và hướng phát triển

- *Một seed duy nhất.* Ngân sách tính toán (\~5,5 giờ cho 8 cấu hình trên GPU laptop) chỉ đủ cho một seed. Kiểm định McNemar (@tab-mcnemar) giúp phân biệt chênh lệch thật với nhiễu trên tập test, nhưng không đo được phương sai do khởi tạo và thứ tự dữ liệu. Các chênh lệch dưới \~1 điểm nên được xác nhận bằng nhiều seed.
- *Công thức from scratch đơn giản.* Chỉ dùng 30 epoch, crop + flip và label smoothing; không có RandAugment, Mixup/CutMix, stochastic depth hay lịch dài hàng trăm epoch. Kết quả from scratch, nhất là của DeiT-S, là cận dưới. Hướng phát triển: công thức kiểu DeiT, hoặc distillation từ ResNet-50 sang DeiT-S qua distillation token để truyền inductive bias của CNN.
- *Độ phân giải 224 cho ảnh 32×32.* Phóng ảnh lên 224 giữ khớp với backbone pretrained nhưng tốn gấp \~49 lần tính toán so với 32×32, và mỗi patch 16×16 của DeiT-S chỉ ứng với \~2×2 pixel gốc. Với from scratch, ViT dùng patch nhỏ ở độ phân giải gốc (ví dụ patch 4 trên ảnh 32×32) là so sánh tự nhiên hơn.
- *Siêu tham số dùng chung, không tinh chỉnh riêng.* Không tìm learning rate riêng cho từng mô hình, không dùng layer-wise LR decay, vốn thường giúp fine-tune ViT tốt hơn.
- *"Một nửa" định nghĩa theo độ sâu*, nên số tham số trainable khác nhau (#fmt(run("resnet50_partial").trainable_pct, d: 0)% với ResNet-50, #fmt(run("deit_s_partial").trainable_pct, d: 0)% với DeiT-S). Có thể bổ sung các mức đóng băng khác (theo tỉ lệ tham số hoặc FLOPs bằng nhau) để có đường cong accuracy theo mức đóng băng.
- *Phạm vi mô hình.* Chỉ một CNN và một ViT cùng pretrain trên ImageNet-1k có giám sát. Có thể mở rộng sang ConvNeXt, Swin, hoặc pretrain lớn hơn/tự giám sát (ImageNet-21k, DINO, MAE).
- *Đo thời gian trên laptop* có thể bị ảnh hưởng bởi nhiệt độ và tác vụ nền; con số thời gian chỉ nên dùng để so sánh tương đối giữa các cấu hình.

// ------------------------------------------------------------------ 9
= Tái lập kết quả

```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
python -m src.eda                # thống kê + hình EDA
python -m scripts.run_all        # 8 cấu hình (bỏ qua run đã xong, resume run dở)
python -m src.analyze            # bảng tổng hợp, biểu đồ, phân tích lỗi
python -m scripts.build_report   # biên dịch báo cáo PDF này
```

Một cấu hình riêng: `python -m src.train --arch deit_s --mode full`. Mỗi run ghi `config.json`, `history.csv`, `metrics.json` và `preds.npz` (logits test) vào `results/<arch>_<mode>/`. Riêng số đo thời gian phụ thuộc phần cứng.

= Tài liệu tham khảo

#set par(justify: false)
+ A. Krizhevsky. _Learning Multiple Layers of Features from Tiny Images._ Tech. report, University of Toronto, 2009.
+ K. He, X. Zhang, S. Ren, J. Sun. _Deep Residual Learning for Image Recognition._ CVPR 2016.
+ A. Dosovitskiy và cộng sự. _An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale._ ICLR 2021.
+ H. Touvron và cộng sự. _Training data-efficient image transformers & distillation through attention._ ICML 2021.
+ R. Wightman, H. Touvron, H. Jégou. _ResNet strikes back: An improved training procedure in timm._ 2021.
+ S. Kornblith, J. Shlens, Q. V. Le. _Do Better ImageNet Models Transfer Better?_ CVPR 2019.
+ M. Raghu và cộng sự. _Do Vision Transformers See Like Convolutional Neural Networks?_ NeurIPS 2021.
+ R. Wightman. _PyTorch Image Models (timm)._ github.com/huggingface/pytorch-image-models.
+ R. Geirhos và cộng sự. _ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness._ ICLR 2019.
+ M. Naseer và cộng sự. _Intriguing Properties of Vision Transformers._ NeurIPS 2021.
+ J. Yosinski, J. Clune, Y. Bengio, H. Lipson. _How transferable are features in deep neural networks?_ NeurIPS 2014.
+ Q. McNemar. _Note on the sampling error of the difference between correlated proportions or percentages._ Psychometrika, 1947.
