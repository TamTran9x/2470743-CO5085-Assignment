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
