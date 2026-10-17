# Playground Series S6E8 — Predicting Smartphone Addiction

二分类：预测 `addicted_label`，指标 **ROC-AUC**，提交 0~1 概率。

最终成绩对应 **mask_student**（5 折 OOF 0.970158）。公榜上 mask_r3 略高是 20% 切片噪声；私榜与 OOF 一致，student 更好。

## 如何运行

环境：conda `mew`（numpy / pandas / scipy / scikit-learn）。

1. 打开 **`Mask_Student.ipynb`**，Run All。
2. 写出 `submission.csv`。

不要跑 `Other_Experiments.ipynb` 来生成正式提交。那份只保留自训 GBDT 和中间 stacking，默认开关为 `False`，也不会覆盖 `submission.csv`。

```text
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
C = 1
```

元模型：logit 后 Newton L2 逻辑回归。169 列 = 公开 135 + extras 19 + 来源分组 LR 4 + mask 9 + mask 分组 LR 1 + student 1。

## 必要数据

| 路径 | 内容 |
|---|---|
| `data/train.csv`, `data/test.csv` | 竞赛数据 |
| `oof_lib/oof/` | 公开 74 个模型的 OOF / test |
| `oof_extra/weak50/` | 50 列弱模型 |
| `oof_extra/paiky11/` | 仅 5 折 6 列（无 `_10f`） |
| `oof_extra/fm/` | 5 个 FM |
| `oof_extra/mask/oof/` | 9 列 mask 增强 GBDT |
| `oof_extra/transductive/` | 蒸馏 student 的 OOF / test |

## 提交格式

```
id,addicted_label
691369,0.75
...
```
