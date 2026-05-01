
# ============================================================
# DS-3002 Data Mining — Assignment #4
# Day 1: Preprocessing (Pre 1-6) + Part A (A1, A2, A3)
# random_state=42 everywhere
# ============================================================

import warnings
warnings.filterwarnings('ignore')
import os
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
COLORS = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B2','#937860','#DA8BC3']

os.makedirs('outputs', exist_ok=True)

# ============================================================
# PRE-1: Load the CSV, confirm shape, print first 5 rows
# ============================================================
print("\n" + "="*60)
print("PRE-1: Load Dataset")
print("="*60)

col_names = ['age','sex','cp','trestbps','chol','fbs','restecg',
             'thalach','exang','oldpeak','slope','ca','thal','target']

df_raw = pd.read_csv('processed.cleveland.data', header=None, names=col_names)
print(f"Shape: {df_raw.shape}")
print("\nFirst 5 rows:")
print(df_raw.head())
print("\nData types:")
print(df_raw.dtypes)

# ============================================================
# PRE-2: Missing values — '?' → NaN, drop, report
# ============================================================
print("\n" + "="*60)
print("PRE-2: Missing Values")
print("="*60)

df = df_raw.replace('?', np.nan)
df = df.apply(pd.to_numeric, errors='coerce')

print("Missing values per column BEFORE dropping:")
mv = df.isnull().sum()
print(mv[mv > 0])

rows_before = len(df)
df = df.dropna()
rows_after = len(df)
print(f"\nRows before drop: {rows_before}")
print(f"Rows after drop : {rows_after}")
print(f"Rows removed    : {rows_before - rows_after}")

# ============================================================
# PRE-3: Class distribution
# ============================================================
print("\n" + "="*60)
print("PRE-3: Class Distribution")
print("="*60)

df['target'] = (df['target'] > 0).astype(int)
counts = df['target'].value_counts()
pcts   = df['target'].value_counts(normalize=True) * 100
dist   = pd.DataFrame({'Count': counts, 'Percent': pcts.round(2)})
dist.index = dist.index.map({0:'No Disease', 1:'Disease'})
print(dist)
print("\nDataset is roughly balanced (~54/46). SMOTE applied on training split only as precaution.")

# ============================================================
# PRE-4: Encoding + Scaling (define; fit after split)
# ============================================================
CAT_COLS  = ['cp','restecg','slope','thal']
CONT_COLS = ['age','trestbps','chol','thalach','oldpeak','ca']
BIN_COLS  = ['sex','fbs','exang']

# One-hot encode categoricals
df_enc = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)
print("\nPRE-4: Shape after one-hot encoding:", df_enc.shape)

# ============================================================
# PRE-5: Stratified 80/20 train/test split
# ============================================================
print("\n" + "="*60)
print("PRE-5: Train/Test Split")
print("="*60)

X = df_enc.drop('target', axis=1)
y = df_enc['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
print(f"Train class balance: {y_train.value_counts().to_dict()}")
print(f"Test  class balance: {y_test.value_counts().to_dict()}")

# Fit StandardScaler on training set only
scaler = StandardScaler()
cont_cols_enc = CONT_COLS  # same names after get_dummies
X_train[cont_cols_enc] = scaler.fit_transform(X_train[cont_cols_enc])
X_test[cont_cols_enc]  = scaler.transform(X_test[cont_cols_enc])
joblib.dump(scaler, 'outputs/scaler.pkl')

# Apply SMOTE on training split only
sm = SMOTE(random_state=SEED)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"\nAfter SMOTE — Train class balance: {pd.Series(y_train_res).value_counts().to_dict()}")

# Save splits for reuse in Parts B, C, E
joblib.dump((X_train, X_test, y_train, y_test), 'outputs/splits.pkl')
joblib.dump((X_train_res, y_train_res), 'outputs/train_resampled.pkl')

# ============================================================
# PRE-6: Correlation heatmap of original numeric features
# ============================================================
print("\n" + "="*60)
print("PRE-6: Correlation Heatmap")
print("="*60)

numeric_orig = df[CONT_COLS + BIN_COLS + ['target']].copy()
corr = numeric_orig.corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5)
ax.set_title('Correlation Heatmap — Numeric Features (Pre-6)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/pre6_corr_heatmap.png', dpi=150)
plt.close()
print("Saved: outputs/pre6_corr_heatmap.png")

# Top 3 correlated pairs (abs, upper triangle)
corr_abs = corr.abs()
upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
top3 = upper.stack().nlargest(3)
print("\nTop 3 feature pairs by correlation:")
print(top3)
print("\nNote: thalach-age & oldpeak-target are expected. Strong correlations between"
      " numeric features can hurt Naive Bayes which assumes feature independence.")

# ============================================================
# PART A: Unsupervised Learning
# Use standardised feature matrix WITHOUT target label
# ============================================================

# Build full scaled matrix (all rows, no target)
X_full_raw = df[CONT_COLS + BIN_COLS].copy()
# Also include one-hot encoded cats for clustering
X_full_enc = df_enc.drop('target', axis=1).copy()
# Scale continuous cols on ALL data (no leakage — unsupervised)
sc_all = StandardScaler()
X_full_enc[CONT_COLS] = sc_all.fit_transform(X_full_enc[CONT_COLS])
X_full = X_full_enc.values
y_true = df['target'].values

print("\n\n" + "="*60)
print("PART A: Unsupervised Learning")
print("="*60)

# ============================================================
# A1: K-Means Clustering
# ============================================================
print("\nA1 — K-Means Clustering")

k_range = range(2, 9)
inertias, sil_scores = [], []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X_full)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_full, labels))

# Plot WCSS + Silhouette on dual y-axis
fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()

line1, = ax1.plot(list(k_range), inertias, 'o-', color='#4C72B0', lw=2, label='WCSS (Inertia)')
line2, = ax2.plot(list(k_range), sil_scores, 's--', color='#DD8452', lw=2, label='Silhouette Score')

best_k = 3   # elbow at k=3, reasonable silhouette
ax1.axvline(best_k, color='red', linestyle=':', lw=1.5, alpha=0.7, label=f'Chosen k={best_k}')
ax2.axvline(best_k, color='red', linestyle=':', lw=1.5, alpha=0.7)

ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
ax1.set_ylabel('WCSS / Inertia', color='#4C72B0', fontsize=11)
ax2.set_ylabel('Silhouette Score', color='#DD8452', fontsize=11)
ax1.tick_params(axis='y', labelcolor='#4C72B0')
ax2.tick_params(axis='y', labelcolor='#DD8452')
lines = [line1, line2]
ax1.legend(lines, [l.get_label() for l in lines], loc='upper right')
ax1.set_title('A1 — K-Means: WCSS & Silhouette Score vs k', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/a1_kmeans_elbow.png', dpi=150)
plt.close()
print(f"Saved: outputs/a1_kmeans_elbow.png  |  Chosen k={best_k}")

# Final KMeans with chosen k
km_best = KMeans(n_clusters=best_k, random_state=SEED, n_init=10)
km_labels = km_best.fit_predict(X_full)

# PCA 2D
pca2 = PCA(n_components=2, random_state=SEED)
X_pca2 = pca2.fit_transform(X_full)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# Left: coloured by KMeans cluster
scatter1 = axes[0].scatter(X_pca2[:,0], X_pca2[:,1], c=km_labels,
                           cmap='Set1', alpha=0.7, s=40, edgecolors='k', lw=0.3)
axes[0].set_title(f'PCA 2D — K-Means Labels (k={best_k})', fontsize=12, fontweight='bold')
axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

# Right: coloured by true disease label
scatter2 = axes[1].scatter(X_pca2[:,0], X_pca2[:,1], c=y_true,
                           cmap='coolwarm', alpha=0.7, s=40, edgecolors='k', lw=0.3)
axes[1].set_title('PCA 2D — True Disease Label', fontsize=12, fontweight='bold')
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
plt.colorbar(scatter2, ax=axes[1], label='0=No Disease, 1=Disease')

plt.suptitle('A1 — PCA Scatter: K-Means Clusters vs True Labels', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/a1_pca_scatter.png', dpi=150)
plt.close()
print("Saved: outputs/a1_pca_scatter.png")

# Cluster summary table
df_clust = df[['thalach','oldpeak','cp']].copy()
df_clust['cluster'] = km_labels
df_clust['target'] = y_true
summary = df_clust.groupby('cluster').agg(
    Size=('target','count'),
    Disease_Prop=('target','mean'),
    Mean_thalach=('thalach','mean'),
    Mean_oldpeak=('oldpeak','mean'),
    Mean_cp=('cp','mean')
).round(3)
print("\nA1 — Cluster Summary Table:")
print(summary.to_string())

# ARI vs true labels
ari_km = adjusted_rand_score(y_true, km_labels)
print(f"\nA1 — Adjusted Rand Index (KMeans vs true labels): {ari_km:.4f}")
print("Interpretation: ARI > 0.1 shows clusters have some alignment with disease "
      "labels but are imperfect — the data has clinical signal but also overlap.")

# ============================================================
# A2: Hierarchical Clustering
# ============================================================
print("\n" + "-"*50)
print("A2 — Hierarchical Clustering")

linked = linkage(X_full, method='ward')

fig, ax = plt.subplots(figsize=(12, 6))
dendrogram(linked, truncate_mode='lastp', p=25, ax=ax,
           color_threshold=0.6*max(linked[:,2]),
           above_threshold_color='grey')
cut_height = sorted(linked[:,2], reverse=True)[2]  # cut after 3 merges from top ≈ 3 clusters
ax.axhline(y=cut_height, color='red', linestyle='--', lw=1.5, label=f'Cut at {cut_height:.1f}')
ax.set_title('A2 — Hierarchical Clustering Dendrogram (Ward, top 25 merges)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Sample index / cluster size'); ax.set_ylabel('Distance')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/a2_dendrogram.png', dpi=150)
plt.close()
print("Saved: outputs/a2_dendrogram.png")

n_hier = best_k   # use same k for comparison
hc = AgglomerativeClustering(n_clusters=n_hier, linkage='ward')
hc_labels = hc.fit_predict(X_full)

# Crosstab
crosstab = pd.crosstab(pd.Series(hc_labels, name='Cluster'),
                        pd.Series(y_true, name='Disease'),
                        margins=True)
crosstab.columns = [f'col_{c}' if isinstance(c, int) else c for c in crosstab.columns]
print("\nA2 — Cluster × Disease Label Crosstab:")
ct = pd.crosstab(hc_labels, y_true, rownames=['Cluster'], colnames=['Disease (0/1)'])
ct.columns.name = 'Disease'
print(ct)

# ARI between KMeans and Hierarchical
ari_compare = adjusted_rand_score(km_labels, hc_labels)
print(f"\nA2 — ARI between K-Means and Hierarchical: {ari_compare:.4f}")
ari_hc = adjusted_rand_score(y_true, hc_labels)
print(f"A2 — ARI Hierarchical vs true labels:      {ari_hc:.4f}")

# ============================================================
# A3: Dimensionality Reduction — PCA + t-SNE
# ============================================================
print("\n" + "-"*50)
print("A3 — Dimensionality Reduction")

# PCA full
pca_full = PCA(random_state=SEED)
pca_full.fit(X_full)
exp_var   = pca_full.explained_variance_ratio_
cum_var   = np.cumsum(exp_var)
n_90      = np.argmax(cum_var >= 0.90) + 1
print(f"Components needed for 90% variance: {n_90}")

fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()
ax1.bar(range(1, len(exp_var)+1), exp_var, color='#4C72B0', alpha=0.7, label='Explained Variance')
ax2.plot(range(1, len(exp_var)+1), cum_var, 'o-', color='#DD8452', lw=2, label='Cumulative Variance')
ax2.axhline(0.90, color='red', linestyle='--', lw=1.2, label='90% threshold')
ax2.axvline(n_90, color='green', linestyle=':', lw=1.5, label=f'{n_90} components')
ax1.set_xlabel('Principal Component'); ax1.set_ylabel('Explained Var. Ratio', color='#4C72B0')
ax2.set_ylabel('Cumulative Explained Variance', color='#DD8452')
ax1.tick_params(axis='y', labelcolor='#4C72B0')
ax2.tick_params(axis='y', labelcolor='#DD8452')
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1+handles2, labels1+labels2, loc='center right')
ax1.set_title('A3 — PCA: Explained Variance per Component', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/a3_pca_variance.png', dpi=150)
plt.close()
print("Saved: outputs/a3_pca_variance.png")

# t-SNE
print("Running t-SNE (this may take ~30 seconds)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=SEED)
X_tsne = tsne.fit_transform(X_full)

fig, ax = plt.subplots(figsize=(8, 6))
colors_tsne = ['#4C72B0' if l==0 else '#DD8452' for l in y_true]
scatter = ax.scatter(X_tsne[:,0], X_tsne[:,1], c=y_true,
                     cmap='coolwarm', alpha=0.75, s=45, edgecolors='k', lw=0.3)
plt.colorbar(scatter, ax=ax, label='0=No Disease / 1=Disease')
ax.set_title('A3 — t-SNE 2D Embedding (coloured by true disease label)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
plt.tight_layout()
plt.savefig('outputs/a3_tsne.png', dpi=150)
plt.close()
print("Saved: outputs/a3_tsne.png")

# ============================================================
# Done — summary
# ============================================================
print("\n" + "="*60)
print("Day 1 complete. All outputs saved in notebooks/outputs/")
print("="*60)
output_files = [
    'pre6_corr_heatmap.png',
    'a1_kmeans_elbow.png',
    'a1_pca_scatter.png',
    'a2_dendrogram.png',
    'a3_pca_variance.png',
    'a3_tsne.png',
    'scaler.pkl',
    'splits.pkl',
    'train_resampled.pkl',
]
for f in output_files:
    path = f'outputs/{f}'
    exists = '✓' if os.path.exists(path) else '✗'
    print(f"  {exists}  {path}")
