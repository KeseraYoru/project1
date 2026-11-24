# Project 1 提交说明（How to run）

用 NumPy 复现 OpenCLIP ViT-B/32 图像编码器的前向计算：输入 `[B,3,224,224]`，输出每张图一个 512 维、长度为 1 的向量。不使用 PyTorch / OpenCLIP，不训练、不更新参数。

仓库（GitHub）：https://github.com/KeseraYoru/project1

## 1　文件清单

| 文件 | 说明 |
| --- | --- |
| `image_encoder.py` | **唯一修改的代码文件**，六个 TODO（10 处空位、22 行代码）全部实现，含中文注释 |
| `README.md` | 本文件：复现步骤与预期输出 |
| `REPORT.md` / `REPORT.docx` | 实验报告（数据流图、分阶段误差、形状变化、bug 修复、贡献说明） |
| `REPORT.html` | 报告的可视化版本，`REPORT.docx` 由它转换而来 |
| `figure_pipeline.png` | 报告图 1：图像塔前向数据流 |

按提交要求，**未**重复提交体积较大的 `weights/`、`assets/` 参数与参考数据，也未提交 `output_tokens.npz` 等中间产物；这些由作业附件提供。

## 2　环境

Python 3.9+，依赖见作业附件的 `requirements.txt`（numpy、pillow、imageio）。本次实测：**Python 3.14.3 + NumPy 2.5.4 + Pillow 13.0.0**，CPU 即可，无需 GPU、PyTorch 或联网。（`requirements.txt` 里 Pillow 上限写作 `<13` 属保守取值，实测 13.0.0 同样跑通；Pillow 只用来读图。）

```bash
python -m pip install -r requirements.txt
```

## 3　复现步骤

**第 1 步**：解压作业附件，把 `image_encoder.py` 放到与 `helpers.py`、`runtime.py`、`check.py`、`weights/`、`assets/` 同级的目录（即覆盖附件里的骨架文件）。

**第 2 步**：跑数值检查。建议按顺序看，第一个不一致的阶段就是问题所在。

```bash
python check.py --stage patches
python check.py --stage tokens
python check.py --stage block0
python check.py --stage last_tokens
python check.py --stage output
python check.py --stage batch
```

**第 3 步**：一次跑完全部阶段并生成输出。

```bash
python check.py --stage all
python run.py --output output_tokens.npz
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

## 4　预期输出

`python check.py --stage all` 应输出八行 PASS，最后一行为 `Requested numerical checks passed.`：

```
assets       PASS: 152 matching visual tensors and ten image/reference pairs
preprocess   PASS shape=(10, 3, 224, 224) max_abs=0.000e+00 mean_abs=0.000e+00 (atol=2e-06, rtol=0.0002)
patches      PASS shape=(2, 49, 3072)     max_abs=0.000e+00 mean_abs=0.000e+00 (atol=0.0002, rtol=0.0002)
tokens       PASS shape=(2, 50, 768)      max_abs=0.000e+00 mean_abs=0.000e+00 (atol=0.0002, rtol=0.0002)
block0       PASS shape=(2, 50, 768)      max_abs=7.540e-06 mean_abs=2.274e-07 (atol=0.0002, rtol=0.0002)
last_tokens  PASS shape=(2, 50, 768)      max_abs=4.387e-05 mean_abs=2.406e-06 (atol=0.0002, rtol=0.0002)
output       PASS shape=(10, 512)         max_abs=6.780e-07 mean_abs=8.215e-08 (atol=2e-05, rtol=0.0002)
batch        PASS shape=(3, 512)          max_abs=4.619e-07 mean_abs=8.133e-08 (atol=2e-05, rtol=0.0002)
```

`python run.py` 会打印 `Saved (10, 512) to output_tokens.npz`。不同 BLAS / NumPy 版本下 `max_abs` 末几位可能有微小差异，只要判定为 PASS 即正确。

## 5　实现要点

- 输入 `[B,3,224,224]`，patch 32×32，切成 7×7 = 49 块，每块 3×32×32 = 3072 维。
- 所有数组保持 `float32`；除 `visual.proj`（已是 `[in,out]`）外，线性层权重均按 `[out,in]` 存储，使用时需转置。
- 注意力中 `k` 为四维，只能交换最后两轴（`transpose(0,1,3,2)`），写 `.T` 会连带翻转 batch 与 head。
- 类接口 `ImageEncoder(weights)`、`LayerNorm`、`TransformerBlock(weights,prefix,heads)` 与最终项目保持一致，`causal` 开关为文本塔预留。

## 6　常见问题

| 现象 | 原因 |
| --- | --- |
| `patches: values differ` 但形状正确 | patchify 里通道轴 C 的位置错了，块内没有按 C→H→W 摊平 |
| `tokens` 起全阶段数值错 | CLS 没插在第 0 行，或位置编码没有广播到 batch |
| `AssertionError: use float32` | 缩放系数用了 float64 标量，把整个张量提升成 float64 |
| `matmul ... core dimension` | 权重没转置；只有 `visual.proj` 不需要转置 |
