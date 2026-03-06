# Day 24 · AutoML Pipeline for Omics Data

**13 classifiers + FLAML AutoML · Microbiome IBD vs Control · 9 figures**

Part of the [#30DaysOfBioinformatics](https://github.com/SubhadipJana1409) challenge.
Previous: [Day 23 – Drug Resistance Mutation Predictor](https://github.com/SubhadipJana1409/AutoML-Pipeline.git)

---

## Overview

AutoML automates model selection and hyperparameter tuning — critical for omics data where the optimal algorithm is unknown a priori. This pipeline benchmarks **13 classifiers** manually against **FLAML** (Fast and Lightweight AutoML, Microsoft Research), which searches RF, XGBoost, LightGBM, ExtraTrees, and more using cost-frugal Bayesian optimisation.

**Task:** Binary classification — IBD vs Control from 16S microbiome OTU profiles (CLR-transformed)

**3 Dataset variants tested:**
- **Balanced** — 50% IBD / 50% Control
- **Imbalanced** — 25% IBD (realistic clinical scenario)
- **Noisy** — 20% label noise (simulates label uncertainty / misdiagnosis)

---

## Manual Benchmark Models (13)

| Model | Type |
|-------|------|
| LogisticL1 | Lasso-penalised LR (sparse, interpretable) |
| LogisticL2 | Ridge-penalised LR (baseline) |
| SGD | Stochastic gradient descent |
| LDA | Linear Discriminant Analysis |
| SVM_linear | Linear kernel SVM |
| SVM_RBF | RBF kernel SVM |
| KNN | k-Nearest Neighbours (k=7) |
| DecisionTree | CART, depth=6 |
| **RandomForest** | **Best manual model: AUC=0.997** |
| ExtraTrees | Extremely randomised trees |
| GradientBoosting | sklearn GBM |
| AdaBoost | Adaptive boosting |
| NaiveBayes | Gaussian NB |

**FLAML AutoML** discovered **LightGBM** as the optimal pipeline: **AUC=0.998**

---

## Figures

| Figure | Description |
|--------|-------------|
| `fig1_dataset_overview.png` | PCA, class balance, OTU variance distribution |
| `fig2_leaderboard.png` | Ranked AUC bar chart — all 13 models + FLAML |
| `fig3_performance_grid.png` | Heatmap: AUC-ROC / AUC-PR / F1 / Accuracy / MCC |
| `fig4_roc_curves.png` | ROC curves — top 5 manual + FLAML |
| `fig5_pr_curves.png` | Precision-Recall curves |
| `fig6_cv_boxplot.png` | CV AUC stability across 5 folds per model |
| `fig7_feature_importance.png` | Top 20 discriminative OTUs (Gini importance) |
| `fig8_dataset_robustness.png` | Performance vs balanced / imbalanced / noisy data |
| `fig9_summary.png` | Best manual vs FLAML, confusion matrix, speed-accuracy trade-off |

---

## Quick Start

```bash
git clone https://github.com/SubhadipJana1409/day24-automl-omics
cd day24-automl-omics
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.main
```

---

## Project Structure

```
day24-automl-omics/
├── src/
│   ├── data/
│   │   └── simulator.py      # IBD/Control OTU simulator (CLR)
│   ├── models/
│   │   └── automl.py         # AutoMLPipeline (13 models + FLAML)
│   ├── visualization/
│   │   └── plots.py          # 9 figures
│   └── main.py
├── tests/
│   ├── test_simulator.py     # 10 tests
│   └── test_automl.py        # 14 tests
├── configs/config.yaml
└── requirements.txt
```

---

## Tests

```bash
pytest tests/ -v
# 24 passed
```

---

## Key Results

| Dataset | Random Forest AUC | FLAML AUC |
|---------|-------------------|-----------|
| Balanced | 0.997 | 0.998 |
| Imbalanced | 0.999 | — |
| Noisy (20% labels) | 0.779 | — |

Label noise drops performance substantially — consistent with clinical reality where IBD diagnosis involves significant subjectivity.

---

## References

1. Wang C, Wu J, et al. (2023). FLAML: A Fast and Lightweight AutoML Library. *Microsoft Research*.
2. Franzosa EA, et al. (2019). Gut microbiota features associated with Clostridioides difficile. *Nature Medicine*.
3. Duvallet C, et al. (2017). Meta-analysis of gut microbiome studies identifies disease-specific and shared responses. *Nature Communications*.

---

## License

MIT
