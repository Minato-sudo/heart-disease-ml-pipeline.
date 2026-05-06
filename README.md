# Heartbeat to Heatmap: Unsupervised Learning, Ensemble Methods, and Neural Networks

**DS-3002 Data Mining — Assignment #4 — Spring 2026**
**FAST-NUCES | BSDS Program**

| Field | Detail |
|---|---|
| **Name** | Muhammad Talha Arshad |
| **Roll No** | 23I-2548 |
| **Section** | BS-DS-A |
| **Seed** | 42 |

---

## Project Overview

This project builds a complete machine learning pipeline for **CardioAI Labs** — a fictional health-tech startup providing decision-support tools for community cardiologists. The pipeline covers:

- **Unsupervised Learning** — K-Means, Hierarchical Clustering, PCA, t-SNE on the UCI Heart Disease dataset
- **Bagging & Boosting** — Random Forest and XGBoost with SHAP explainability
- **Neural Networks** — Single-Layer Perceptron, Multi-Layer Perceptron, and CNN on MNIST
- **Local Dashboard** — Streamlit front-end for real-time heart disease risk prediction

---

## Repository Structure

```
heart-disease-ml-pipeline/
│
├── notebooks/
│   └── assignment4.ipynb          # Full pipeline notebook (Pre → A → B → C → D)
│
├── app/
│   ├── app.py                     # Streamlit dashboard (Part E)
│   ├── best_xgb.pkl               # Saved XGBoost model
│   ├── scaler.pkl                 # Fitted StandardScaler
│   ├── splits.pkl                 # Train/test split arrays
│   └── requirements.txt           # App dependencies
│
├── report/
│   └── i232548_Assignment4_Report.pdf   # Final LaTeX report
│
├── notebooks/outputs/             # All saved plot PNGs
│   ├── pre6_corr_heatmap.png
│   ├── a1_kmeans_elbow.png
│   ├── a1_pca_scatter.png
│   ├── a2_dendrogram.png
│   ├── a3_pca_variance.png
│   ├── a3_tsne.png
│   ├── b1_rf_oob.png
│   ├── b1_rf_feat_imp.png
│   ├── b2_xgb_logloss.png
│   ├── b2_xgb_shap.png
│   ├── b3_roc_comparison.png
│   ├── c1_slp_history.png
│   ├── c2_mlp_history.png
│   ├── c3_ablation.png
│   ├── d1_samples.png
│   ├── d2_cnn_history.png
│   ├── d2_cnn_confusion.png
│   ├── d3_filters.png
│   └── d3_feature_maps.png
│
├── requirements.txt               # Full environment dependencies
└── README.md
```

---

## Datasets

### Dataset 1 — UCI Heart Disease (Cleveland)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **File:** `processed.cleveland.data`
- **Size:** 303 rows × 14 columns (297 retained after cleaning)
- **Task:** Binary classification — heart disease present (1) vs absent (0)

**Download steps:**
1. Visit https://archive.ics.uci.edu/dataset/45/heart+disease
2. Download `processed.cleveland.data`
3. Place it in the `notebooks/` folder before running the notebook

### Dataset 2 — MNIST Handwritten Digits (subset)
- **Load method:** Built into Keras — no download required
```python
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, y_train = X_train[:12000], y_train[:12000]
X_test,  y_test  = X_test[:2000],  y_test[:2000]
```

---

## Environment Setup

### Option A — pip (recommended)

```bash
# Clone the repo
git clone https://github.com/Minato-sudo/heart-disease-ml-pipeline
cd heart-disease-ml-pipeline

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### Option B — conda

```bash
conda create -n cardioai python=3.10
conda activate cardioai
pip install -r requirements.txt
```

---

## Running the Notebook

```bash
# Make sure processed.cleveland.data is in notebooks/
jupyter notebook notebooks/assignment4.ipynb
```

> **Important:** Restart the kernel and run all cells in order (Kernel → Restart & Run All) to reproduce all results. All random seeds are fixed at `random_state=42`.

---

## Running the Dashboard (Part E)

```bash
cd app/
streamlit run app.py
```

The app will open at **http://localhost:8501**

**What the dashboard does:**
- Accepts 13 clinical measurements as inputs
- Predicts heart disease risk using the saved XGBoost model (AUC = 0.9247)
- Displays a colour-coded risk label (green = No Disease, red = Disease Present)
- Shows model confidence percentage
- Renders a Top 3 feature importance bar chart
- Provides a plain-English clinical explanation

Pre-populated with **Real Patient 1 (Low Risk)** for instant testing.

---

## Key Results Summary

| Part | Model | Accuracy | Macro F1 | AUC-ROC | Recall (Disease) |
|---|---|---|---|---|---|
| B1 | Random Forest | 0.8333 | 0.8316 | 0.9247 | 0.7857 |
| B2 | XGBoost | 0.8333 | 0.8303 | 0.9174 | 0.7500 |
| C1 | SLP | ~0.767 | ~0.754 | ~0.820 | — |
| C2 | **Best MLP** | **0.8667** | **0.8665** | **0.9375** | **0.8929** |

| Part | Model | Test Accuracy | Macro F1 |
|---|---|---|---|
| D1 | MLP Baseline (MNIST) | 0.9180 | 0.9174 |
| D2 | **CNN (MNIST)** | **0.9775** | **0.9774** |

**Key insight:** The MLP significantly outperforms ensemble methods in Recall (0.8929 vs 0.7857 for RF), which is the most important metric in cardiac screening — a false negative means a sick patient is sent home untreated.

---

## requirements.txt

```
numpy
pandas
scikit-learn
imbalanced-learn
xgboost
shap
tensorflow
matplotlib
seaborn
scipy
streamlit
joblib
jupyter
```

---

## References

- Detrano, R. et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*, 64(5), 304–310.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
- Lundberg, S.M. & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
- LeCun, Y. et al. (1998). Gradient-Based Learning Applied to Document Recognition. *Proceedings of the IEEE*, 86(11).
- van der Maaten, L. & Hinton, G. (2008). Visualizing Data using t-SNE. *JMLR*, 9, 2579–2605.

---

*DS-3002 Data Mining — Assignment #4 — Spring 2026 — FAST-NUCES*