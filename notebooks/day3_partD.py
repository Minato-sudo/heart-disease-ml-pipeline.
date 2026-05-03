import warnings
warnings.filterwarnings('ignore')
import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Flatten, Conv2D, MaxPooling2D,
                                      Dropout, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import joblib

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs('outputs', exist_ok=True)

# ============================================================
# D1: Data Preparation & Baseline
# ============================================================
print("=" * 60)
print("PART D: CNN on MNIST Digits")
print("=" * 60)

(X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()

# Subset: 12,000 train, 2,000 test
X_train_d = X_train_full[:12000]
y_train_d = y_train_full[:12000]
X_test_d  = X_test_full[:2000]
y_test_d  = y_test_full[:2000]

# Normalize to [0,1]
X_train_d = X_train_d.astype('float32') / 255.0
X_test_d  = X_test_d.astype('float32')  / 255.0

# Reshape to (28, 28, 1)
X_train_cnn = X_train_d.reshape(-1, 28, 28, 1)
X_test_cnn  = X_test_d.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train_cat = to_categorical(y_train_d, 10)
y_test_cat  = to_categorical(y_test_d,  10)

print(f"Train: {X_train_cnn.shape}  |  Test: {X_test_cnn.shape}")
print(f"Label distribution (train): {dict(zip(*np.unique(y_train_d, return_counts=True)))}")

# 5x2 sample grid — one image per digit class
fig, axes = plt.subplots(2, 5, figsize=(11, 5))
for digit in range(10):
    idx = np.where(y_train_d == digit)[0][0]
    ax  = axes[digit // 5][digit % 5]
    ax.imshow(X_train_d[idx], cmap='gray')
    ax.set_title(f'Digit: {digit}', fontsize=11, fontweight='bold')
    ax.axis('off')
plt.suptitle('D1 — Sample Images (one per digit class)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/d1_samples.png', dpi=150)
plt.close()
print("Saved: outputs/d1_samples.png")

# MLP Baseline (flattened images)
print("\nD1 — MLP Baseline...")
mlp_baseline = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
mlp_baseline.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
mlp_baseline.fit(X_train_cnn, y_train_cat, epochs=5, verbose=0,
                  validation_data=(X_test_cnn, y_test_cat))

_, baseline_acc = mlp_baseline.evaluate(X_test_cnn, y_test_cat, verbose=0)
print(f"MLP Baseline Test Accuracy: {baseline_acc:.4f}")
joblib.dump(baseline_acc, 'outputs/mlp_baseline_acc.pkl')

# ============================================================
# D2: Lightweight CNN
# ============================================================
print("\n" + "=" * 60)
print("D2 — Lightweight CNN")
print("=" * 60)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(X_train_cnn)

cnn = Sequential([
    Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same',
           input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

cnn.compile(optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy', metrics=['accuracy'])
cnn.summary()

start = time.time()
history_cnn = cnn.fit(
    datagen.flow(X_train_cnn, y_train_cat, batch_size=64),
    steps_per_epoch=len(X_train_cnn) // 64,
    epochs=15,
    validation_data=(X_test_cnn, y_test_cat),
    verbose=1
)
cnn_time = time.time() - start
print(f"\nCNN training time: {cnn_time:.1f}s")

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history_cnn.history['accuracy'],    label='Train Acc')
axes[0].plot(history_cnn.history['val_accuracy'], label='Val Acc')
axes[0].axhline(baseline_acc, color='gray', linestyle='--', label=f'MLP Baseline ({baseline_acc:.3f})')
axes[0].set_title('D2 — CNN: Accuracy per Epoch', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].legend()

axes[1].plot(history_cnn.history['loss'],     label='Train Loss')
axes[1].plot(history_cnn.history['val_loss'],  label='Val Loss')
axes[1].set_title('D2 — CNN: Loss per Epoch', fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].legend()
plt.tight_layout()
plt.savefig('outputs/d2_cnn_history.png', dpi=150)
plt.close()

# Evaluation
_, cnn_acc = cnn.evaluate(X_test_cnn, y_test_cat, verbose=0)
y_pred_cnn = np.argmax(cnn.predict(X_test_cnn, verbose=0), axis=1)
cnn_f1     = f1_score(y_test_d, y_pred_cnn, average='macro')
cm_cnn     = confusion_matrix(y_test_d, y_pred_cnn)

print(f"\nCNN Test Accuracy: {cnn_acc:.4f}")
print(f"CNN Macro F1:      {cnn_f1:.4f}")
print(f"MLP Baseline:      {baseline_acc:.4f}")

# Find epoch CNN first surpasses MLP baseline
surpass_epoch = next((i+1 for i, v in enumerate(history_cnn.history['val_accuracy'])
                      if v >= baseline_acc), None)
print(f"CNN surpasses MLP baseline at epoch: {surpass_epoch}")

# Confusion matrix heatmap
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=range(10), yticklabels=range(10))
ax.set_title('D2 — CNN: Confusion Matrix', fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout()
plt.savefig('outputs/d2_cnn_confusion.png', dpi=150)
plt.close()

# Most confused digit pairs
errors = cm_cnn.copy(); np.fill_diagonal(errors, 0)
top_confused = np.unravel_index(errors.argsort(axis=None)[-4:], errors.shape)
print("\nMost confused digit pairs:")
for i in range(len(top_confused[0])-1, -1, -1):
    r, c = top_confused[0][i], top_confused[1][i]
    print(f"  {r} misclassified as {c}: {errors[r, c]} times")

cnn.save('outputs/cnn_mnist.h5')
print("Saved: outputs/cnn_mnist.h5")

# ============================================================
# D3: Visualising What the CNN Learned
# ============================================================
print("\n" + "=" * 60)
print("D3 — CNN Visualisations")
print("=" * 60)

# First Conv2D filter weights — 4×4 grid of 16 filters
filters = cnn.layers[0].get_weights()[0]  # shape (3,3,1,16)
fig, axes = plt.subplots(4, 4, figsize=(7, 7))
for i in range(16):
    ax = axes[i // 4][i % 4]
    f  = filters[:, :, 0, i]
    f  = (f - f.min()) / (f.max() - f.min() + 1e-8)
    ax.imshow(f, cmap='viridis')
    ax.set_title(f'F{i+1}', fontsize=8)
    ax.axis('off')
plt.suptitle('D3 — First Conv2D: 16 Learned Filters (4×4 grid)', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/d3_filters.png', dpi=150)
plt.close()
print("Saved: outputs/d3_filters.png")

# Feature maps for one image per digit — first 8 channels in a 2×4 grid
inp = tf.keras.Input(shape=(28, 28, 1))
x   = cnn.layers[0](inp)
feat_model = tf.keras.Model(inputs=inp, outputs=x)

fig_fm, axes_fm = plt.subplots(10, 8, figsize=(18, 22))
for digit in range(10):
    idx   = np.where(y_test_d == digit)[0][0]
    img   = X_test_cnn[idx:idx+1]
    fmaps = feat_model.predict(img, verbose=0)  # (1, 28, 28, 16)
    for ch in range(8):
        ax = axes_fm[digit][ch]
        ax.imshow(fmaps[0, :, :, ch], cmap='viridis')
        if ch == 0:
            ax.set_ylabel(f'Digit {digit}', fontsize=9, fontweight='bold')
        ax.set_title(f'Ch{ch+1}', fontsize=7)
        ax.axis('off')

plt.suptitle('D3 — Feature Maps: First Conv2D Layer (8 channels, one row per digit)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/d3_feature_maps.png', dpi=150)
plt.close()
print("Saved: outputs/d3_feature_maps.png")

# ============================================================
# Done — verify all outputs
# ============================================================
print("\n" + "=" * 60)
print("Day 3 Part D complete. All outputs:")
print("=" * 60)
for f in ['d1_samples.png', 'd2_cnn_history.png', 'd2_cnn_confusion.png',
          'd3_filters.png', 'd3_feature_maps.png', 'cnn_mnist.h5',
          'mlp_baseline_acc.pkl']:
    p = f"outputs/{f}"
    print(f"  {'✓' if os.path.exists(p) else '✗ MISSING'}  {p}")
