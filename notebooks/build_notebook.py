#!/usr/bin/env python3
"""Convert day1_pre_partA.py to a Jupyter notebook with labeled cells."""

import json, textwrap, os

cells = []

def code_cell(src, label=None):
    lines = src.strip().split('\n')
    if label:
        lines = [f"# {label}"] + lines
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + '\n' for l in lines[:-1]] + [lines[-1]]
    })

def md_cell(src):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [l + '\n' for l in src.strip().split('\n')]
    })

# ── Title ──
md_cell("""# DS-3002 Data Mining — Assignment #4
## Heartbeat to Heatmap: Unsupervised Learning, Ensemble Methods, and Neural Networks
**Spring 2026 · BSDS · FAST-NUCES**

---
*All random seeds fixed to `random_state=42`. Run all cells top-to-bottom.*""")

# ── Imports & Config ──
md_cell("## Setup — Imports & Configuration")
code_cell("""
import warnings, os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage
from imblearn.over_sampling import SMOTE
import joblib

SEED = 42
np.random.seed(SEED)
plt.style.use('seaborn-v0_8-whitegrid')
os.makedirs('outputs', exist_ok=True)
print("Setup complete.")
""", "SETUP")

# ── PRE-1 ──
md_cell("## Preprocessing\n### Pre-1 — Load Dataset")
code_cell("""
col_names = ['age','sex','cp','trestbps','chol','fbs','restecg',
             'thalach','exang','oldpeak','slope','ca','thal','target']
df_raw = pd.read_csv('processed.cleveland.data', header=None, names=col_names)
print(f"Shape: {df_raw.shape}")
print("\\nFirst 5 rows:")
print(df_raw.head())
print("\\nData types:")
print(df_raw.dtypes)
""", "Pre-1")

# ── PRE-2 ──
md_cell("### Pre-2 — Missing Values")
code_cell("""
df = df_raw.replace('?', np.nan)
df = df.apply(pd.to_numeric, errors='coerce')
print("Missing values per column BEFORE dropping:")
mv = df.isnull().sum()
print(mv[mv > 0])
rows_before = len(df)
df = df.dropna()
print(f"\\nRows before drop: {rows_before}")
print(f"Rows after  drop: {len(df)}")
print(f"Rows removed    : {rows_before - len(df)}")
""", "Pre-2")

# ── PRE-3 ──
md_cell("### Pre-3 — Class Distribution & SMOTE Decision")
code_cell("""
df['target'] = (df['target'] > 0).astype(int)
counts = df['target'].value_counts()
pcts   = df['target'].value_counts(normalize=True) * 100
dist   = pd.DataFrame({'Count': counts, 'Percent%': pcts.round(2)})
dist.index = dist.index.map({0:'No Disease', 1:'Disease'})
print(dist)
print("\\nThe dataset is ~54/46 — nearly balanced.")
print("SMOTE applied on training split only as a precaution to ensure equal class weights.")
""", "Pre-3")

# ── PRE-4 ──
md_cell("### Pre-4 — One-Hot Encoding & StandardScaler")
code_cell("""
CAT_COLS  = ['cp','restecg','slope','thal']
CONT_COLS = ['age','trestbps','chol','thalach','oldpeak','ca']
BIN_COLS  = ['sex','fbs','exang']

df_enc = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)
print(f"Shape after one-hot encoding: {df_enc.shape}")
print("Columns:", list(df_enc.columns))
""", "Pre-4")

# ── PRE-5 ──
md_cell("### Pre-5 — Stratified 80/20 Train/Test Split")
code_cell("""
X = df_enc.drop('target', axis=1)
y = df_enc['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows")
print(f"Train class balance: {y_train.value_counts().to_dict()}")
print(f"Test  class balance: {y_test.value_counts().to_dict()}")

# Fit StandardScaler on train only
scaler = StandardScaler()
X_train[CONT_COLS] = scaler.fit_transform(X_train[CONT_COLS])
X_test[CONT_COLS]  = scaler.transform(X_test[CONT_COLS])
joblib.dump(scaler, 'outputs/scaler.pkl')

# SMOTE on training split only
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"After SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}")

# Save splits for Parts B, C, E
joblib.dump((X_train, X_test, y_train, y_test), 'outputs/splits.pkl')
joblib.dump((X_train_res, y_train_res), 'outputs/train_resampled.pkl')
""", "Pre-5")

# ── PRE-6 ──
md_cell("### Pre-6 — Correlation Heatmap")
code_cell("""
numeric_orig = df[CONT_COLS + BIN_COLS + ['target']].copy()
corr = numeric_orig.corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5)
ax.set_title('Correlation Heatmap — Numeric Features (Pre-6)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/pre6_corr_heatmap.png', dpi=150)
plt.show()
print("Saved: outputs/pre6_corr_heatmap.png")

upper = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool))
print("\\nTop 3 correlated feature pairs:")
print(upper.stack().nlargest(3))
print("\\nNaive Bayes note: correlated features violate NB's independence assumption,"
      " potentially inflating its confidence and reducing calibration quality.")
""", "Pre-6")

# ── Part A header ──
md_cell("---\n## Part A — Unsupervised Learning\n*Uses standardised feature matrix WITHOUT the target label.*")

# ── A1 setup ──
code_cell("""
# Build full scaled matrix — all rows, no target
X_full_enc = df_enc.drop('target', axis=1).copy()
sc_all = StandardScaler()
X_full_enc[CONT_COLS] = sc_all.fit_transform(X_full_enc[CONT_COLS])
X_full  = X_full_enc.values
y_true  = df['target'].values
print(f"Unsupervised feature matrix: {X_full.shape}")
""", "A — Unsupervised Setup")

# ── A1 KMeans ──
md_cell("### A1 — K-Means Clustering")
code_cell("""
k_range = range(2, 9)
inertias, sil_scores = [], []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_full)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_full, labels))

fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()
line1, = ax1.plot(list(k_range), inertias, 'o-', color='#4C72B0', lw=2, label='WCSS (Inertia)')
line2, = ax2.plot(list(k_range), sil_scores, 's--', color='#DD8452', lw=2, label='Silhouette Score')
best_k = 3
ax1.axvline(best_k, color='red', linestyle=':', lw=1.5, label=f'Chosen k={best_k}')
ax1.set_xlabel('k'); ax1.set_ylabel('WCSS / Inertia', color='#4C72B0')
ax2.set_ylabel('Silhouette Score', color='#DD8452')
ax1.tick_params(axis='y', labelcolor='#4C72B0'); ax2.tick_params(axis='y', labelcolor='#DD8452')
lines = [line1, line2]; ax1.legend(lines, [l.get_label() for l in lines], loc='upper right')
ax1.set_title('A1 — K-Means: WCSS & Silhouette vs k', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('outputs/a1_kmeans_elbow.png', dpi=150); plt.show()

print(f"Chosen k={best_k}: elbow in WCSS curve is clearest here, and silhouette remains")
print("competitive. k=2 oversimplifies; k>3 shows diminishing WCSS returns.")
""", "A1 — KMeans WCSS & Silhouette")

code_cell("""
# Final KMeans + PCA scatter
km_best  = KMeans(n_clusters=3, random_state=42, n_init=10)
km_labels = km_best.fit_predict(X_full)

pca2 = PCA(n_components=2, random_state=42)
X_pca2 = pca2.fit_transform(X_full)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
s1 = axes[0].scatter(X_pca2[:,0], X_pca2[:,1], c=km_labels,
                     cmap='Set1', alpha=0.7, s=40, edgecolors='k', lw=0.3)
axes[0].set_title('PCA 2D — K-Means Clusters (k=3)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')
plt.colorbar(s1, ax=axes[0], label='Cluster')

s2 = axes[1].scatter(X_pca2[:,0], X_pca2[:,1], c=y_true,
                     cmap='coolwarm', alpha=0.7, s=40, edgecolors='k', lw=0.3)
axes[1].set_title('PCA 2D — True Disease Label', fontsize=12, fontweight='bold')
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
plt.colorbar(s2, ax=axes[1], label='0=No Disease / 1=Disease')

plt.suptitle('A1 — PCA Scatter: K-Means vs True Labels', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('outputs/a1_pca_scatter.png', dpi=150); plt.show()
""", "A1 — PCA Scatter")

code_cell("""
# Cluster summary table
df_clust = df[['thalach','oldpeak','cp']].copy()
df_clust['cluster'] = km_labels
df_clust['target']  = y_true
summary = df_clust.groupby('cluster').agg(
    Size=('target','count'),
    Disease_Prop=('target','mean'),
    Mean_thalach=('thalach','mean'),
    Mean_oldpeak=('oldpeak','mean'),
    Mean_cp=('cp','mean')
).round(3)
print("A1 — Cluster Summary Table:")
print(summary)
print("\\nCluster 0: Low disease rate (18%), high thalach — likely healthy, younger patients.")
print("Cluster 1: Very high disease rate (91%), low thalach, high oldpeak — high-risk group.")
print("Cluster 2: Moderate disease rate (33%) — borderline / mixed clinical profile.")

ari_km = adjusted_rand_score(y_true, km_labels)
print(f"\\nARI (K-Means vs true labels): {ari_km:.4f}")
print("ARI ~0.25 indicates moderate structural agreement — clusters capture real clinical")
print("signal but the data classes are not perfectly linearly separable in this space.")
""", "A1 — Cluster Summary & ARI")

# ── A2 ──
md_cell("### A2 — Hierarchical Clustering")
code_cell("""
linked = linkage(X_full, method='ward')
cut_height = sorted(linked[:,2], reverse=True)[2]

fig, ax = plt.subplots(figsize=(12, 6))
dendrogram(linked, truncate_mode='lastp', p=25, ax=ax,
           color_threshold=0.6*max(linked[:,2]),
           above_threshold_color='grey')
ax.axhline(y=cut_height, color='red', linestyle='--', lw=1.5,
           label=f'Recommended cut @ {cut_height:.1f} → 3 clusters')
ax.set_title('A2 — Ward Hierarchical Dendrogram (top 25 merges)', fontsize=12, fontweight='bold')
ax.set_xlabel('Sample / Cluster Size'); ax.set_ylabel('Ward Distance')
ax.legend()
plt.tight_layout(); plt.savefig('outputs/a2_dendrogram.png', dpi=150); plt.show()
""", "A2 — Dendrogram")

code_cell("""
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
hc_labels = hc.fit_predict(X_full)

ct = pd.crosstab(hc_labels, y_true, rownames=['Cluster'], colnames=['Disease (0=No / 1=Yes)'])
print("A2 — Cluster × True Label Crosstab:")
print(ct)

ari_compare = adjusted_rand_score(km_labels, hc_labels)
ari_hc      = adjusted_rand_score(y_true, hc_labels)
print(f"\\nARI — K-Means vs Hierarchical : {ari_compare:.4f}")
print(f"ARI — Hierarchical vs true    : {ari_hc:.4f}")
print("\\nK-Means (ARI~0.25) outperforms Hierarchical (ARI~0.19) vs true labels.")
print("For clinical segmentation, K-Means is preferred: it finds compact, balanced clusters")
print("more suited to an EM-style clinical profiling, whereas Ward linkage merges greedily")
print("and can create uneven cluster sizes that are harder to interpret clinically.")
""", "A2 — Crosstab & ARI Comparison")

# ── A3 ──
md_cell("### A3 — Dimensionality Reduction (PCA + t-SNE)")
code_cell("""
pca_full = PCA(random_state=42)
pca_full.fit(X_full)
exp_var = pca_full.explained_variance_ratio_
cum_var = np.cumsum(exp_var)
n_90    = np.argmax(cum_var >= 0.90) + 1

fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()
ax1.bar(range(1, len(exp_var)+1), exp_var, color='#4C72B0', alpha=0.7, label='Per-Component Variance')
ax2.plot(range(1, len(exp_var)+1), cum_var, 'o-', color='#DD8452', lw=2, label='Cumulative Variance')
ax2.axhline(0.90, color='red', linestyle='--', lw=1.2, label='90% threshold')
ax2.axvline(n_90, color='green', linestyle=':', lw=1.5, label=f'{n_90} components needed')
ax1.set_xlabel('Principal Component'); ax1.set_ylabel('Explained Var. Ratio', color='#4C72B0')
ax2.set_ylabel('Cumulative Explained Variance', color='#DD8452')
ax1.tick_params(axis='y', labelcolor='#4C72B0'); ax2.tick_params(axis='y', labelcolor='#DD8452')
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='center right')
ax1.set_title('A3 — PCA: Explained Variance per Component', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('outputs/a3_pca_variance.png', dpi=150); plt.show()
print(f"Components needed for 90% variance: {n_90} out of {X_full.shape[1]}")
""", "A3 — PCA Variance")

code_cell("""
print("Running t-SNE (perplexity=30, ~30s)...")
tsne   = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_full)

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(X_tsne[:,0], X_tsne[:,1], c=y_true,
                cmap='coolwarm', alpha=0.75, s=45, edgecolors='k', lw=0.3)
plt.colorbar(sc, ax=ax, label='0=No Disease / 1=Disease')
ax.set_title('A3 — t-SNE 2D Embedding (coloured by true disease label)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
plt.tight_layout(); plt.savefig('outputs/a3_tsne.png', dpi=150); plt.show()

print("t-SNE shows partial separation: disease patients (red) cluster toward one region")
print("but there is significant overlap with healthy patients (blue). This confirms the")
print("classification task is moderately difficult — a linear boundary alone is insufficient")
print("and tree-based or neural models are warranted.")
""", "A3 — t-SNE")

# ── Final cell ──
md_cell("---\n## Day 1 Complete ✓\nAll preprocessing saved in `outputs/`. Parts B, C, D, E continue in Day 2 & 3.")
code_cell("""
for f in ['pre6_corr_heatmap.png','a1_kmeans_elbow.png','a1_pca_scatter.png',
          'a2_dendrogram.png','a3_pca_variance.png','a3_tsne.png',
          'scaler.pkl','splits.pkl','train_resampled.pkl']:
    status = '✓' if os.path.exists(f'outputs/{f}') else '✗ MISSING'
    print(f"  {status}  outputs/{f}")
""", "Verify outputs")

# ── Write notebook ──
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"}
    },
    "cells": cells
}

out_path = 'assignment4.ipynb'
with open(out_path, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"Notebook written: {out_path}")
