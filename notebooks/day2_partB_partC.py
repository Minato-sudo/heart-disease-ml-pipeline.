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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, log_loss
import xgboost as xgb
import shap
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping

import joblib

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
plt.style.use('seaborn-v0_8-whitegrid')

os.makedirs('outputs', exist_ok=True)

print("Loading splits...")
X_train, X_test, y_train, y_test = joblib.load('outputs/splits.pkl')
X_train_res, y_train_res = joblib.load('outputs/train_resampled.pkl')
scaler = joblib.load('outputs/scaler.pkl')

feature_names = X_train.columns.tolist()

# Helper for evaluation
def evaluate_model(model_name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"--- {model_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1:  {f1:.4f}")
    print(f"Macro Prec:{prec:.4f}")
    print(f"Macro Rec: {rec:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"Confusion Matrix:\\n{cm}\\n")
    return {'acc': acc, 'f1': f1, 'auc': auc, 'rec_1': recall_score(y_true, y_pred, pos_label=1), 'time': 0}

# ============================================================
# PART B1: Random Forest
# ============================================================
print("\\n" + "="*60)
print("PART B1: Random Forest")
print("="*60)

rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10]
}

rf = RandomForestClassifier(random_state=SEED)
grid_rf = GridSearchCV(rf, rf_param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
grid_rf.fit(X_train_res, y_train_res)

print("Best RF Params:", grid_rf.best_params_)
print(f"Best CV F1: {grid_rf.best_score_:.4f}")

best_rf = grid_rf.best_estimator_
best_rf.fit(X_train_res, y_train_res)
y_pred_rf = best_rf.predict(X_test)
y_prob_rf = best_rf.predict_proba(X_test)[:, 1]

rf_metrics = evaluate_model("Random Forest", y_test, y_pred_rf, y_prob_rf)

# OOB Error plot
n_estimators_range = range(1, 201)
oob_errors = []
rf_oob = RandomForestClassifier(warm_start=True, oob_score=True, random_state=SEED)

for i in n_estimators_range:
    rf_oob.set_params(n_estimators=i)
    rf_oob.fit(X_train_res, y_train_res)
    oob_error = 1 - rf_oob.oob_score_
    oob_errors.append(oob_error)

plt.figure(figsize=(8, 5))
plt.plot(n_estimators_range, oob_errors, label='OOB Error', color='#4C72B0')
plt.axvline(grid_rf.best_params_['n_estimators'], color='red', linestyle='--', label=f"Chosen n_trees={grid_rf.best_params_['n_estimators']}")
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('OOB Error')
plt.title('B1 - Random Forest: OOB Error vs Number of Trees', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/b1_rf_oob.png', dpi=150)
plt.close()

# Feature Importances
importances = best_rf.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8, 8))
plt.title('B1 - Random Forest: Feature Importances', fontweight='bold')
plt.barh(range(len(indices)), importances[indices], align='center', color='#55A868')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Mean Decrease in Impurity')
plt.tight_layout()
plt.savefig('outputs/b1_rf_feat_imp.png', dpi=150)
plt.close()

top_5_idx = indices[-5:][::-1]
print("Top 5 Features:")
for i in top_5_idx:
    print(f" - {feature_names[i]}: {importances[i]:.4f}")


# ============================================================
# PART B2: XGBoost
# ============================================================
print("\\n" + "="*60)
print("PART B2: XGBoost")
print("="*60)

xgb_param_grid = {
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7]
}

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=SEED)
grid_xgb = GridSearchCV(xgb_model, xgb_param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
grid_xgb.fit(X_train_res, y_train_res)

print("Best XGB Params:", grid_xgb.best_params_)
print(f"Best CV F1: {grid_xgb.best_score_:.4f}")

# Train final with early stopping
best_xgb = xgb.XGBClassifier(**grid_xgb.best_params_, random_state=SEED, n_estimators=500, eval_metric='logloss', early_stopping_rounds=50)
eval_set = [(X_train_res, y_train_res), (X_test, y_test)]
best_xgb.fit(X_train_res, y_train_res, eval_set=eval_set, verbose=False)

results = best_xgb.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

plt.figure(figsize=(8, 5))
plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
plt.plot(x_axis, results['validation_1']['logloss'], label='Validation')
plt.axvline(best_xgb.best_iteration, color='red', linestyle='--', label=f'Optimal Round ({best_xgb.best_iteration})')
plt.legend()
plt.xlabel('Boosting Rounds')
plt.ylabel('Log Loss')
plt.title('B2 - XGBoost: Train vs Validation Log Loss', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/b2_xgb_logloss.png', dpi=150)
plt.close()

y_pred_xgb = best_xgb.predict(X_test)
y_prob_xgb = best_xgb.predict_proba(X_test)[:, 1]

xgb_metrics = evaluate_model("XGBoost", y_test, y_pred_xgb, y_prob_xgb)

# SHAP values
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title('B2 - XGBoost: SHAP Values Summary', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/b2_xgb_shap.png', dpi=150, bbox_inches='tight')
plt.close()

joblib.dump(best_xgb, 'outputs/best_xgb.pkl')


# ============================================================
# PART C: Neural Networks
# ============================================================
print("\\n" + "="*60)
print("PART C: Neural Networks")
print("="*60)

# C1: Single-Layer Perceptron (SLP)
print("\\nC1 — Single-Layer Perceptron (SLP)")

slp = Sequential([
    Dense(1, input_dim=X_train_res.shape[1], activation='sigmoid')
])
slp.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])

history_slp = slp.fit(X_train_res, y_train_res, epochs=100, verbose=0)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history_slp.history['loss'], label='Train Loss')
plt.title('C1 - SLP: Training Loss', fontweight='bold')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_slp.history['accuracy'], label='Train Accuracy')
plt.title('C1 - SLP: Training Accuracy', fontweight='bold')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/c1_slp_history.png', dpi=150)
plt.close()

weights = slp.layers[0].get_weights()[0].flatten()
abs_weights = np.abs(weights)
top3_slp_idx = np.argsort(abs_weights)[-3:][::-1]

print("Top 3 SLP features (highest absolute weights):")
for i in top3_slp_idx:
    print(f" - {feature_names[i]}: {weights[i]:.4f} (abs: {abs_weights[i]:.4f})")

y_prob_slp = slp.predict(X_test, verbose=0).flatten()
y_pred_slp = (y_prob_slp > 0.5).astype(int)

slp_metrics = evaluate_model("SLP", y_test, y_pred_slp, y_prob_slp)

# C2: Multi-Layer Perceptron (MLP)
print("\\nC2 — Multi-Layer Perceptron (MLP)")

def create_mlp(architecture):
    model = Sequential()
    model.add(Dense(architecture[0], input_dim=X_train_res.shape[1], activation='relu'))
    model.add(Dropout(0.3))
    for units in architecture[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

architectures = {
    'Small': [32],
    'Medium': [64, 32],
    'Large': [128, 64, 32]
}

best_val_f1 = 0
best_mlp_name = ""
mlp_results = {}

import time

for name, arch in architectures.items():
    start_time = time.time()
    model = create_mlp(arch)
    
    # Simple validation split for architecture search
    X_t, X_v, y_t, y_v = train_test_split(X_train_res, y_train_res, test_size=0.2, random_state=SEED)
    model.fit(X_t, y_t, epochs=50, verbose=0)
    
    y_p = (model.predict(X_v, verbose=0).flatten() > 0.5).astype(int)
    f1 = f1_score(y_v, y_p, average='macro')
    train_time = time.time() - start_time
    mlp_results[name] = {'val_f1': f1, 'time': train_time}
    
    print(f"Arch: {name} | Val F1: {f1:.4f} | Time: {train_time:.2f}s")
    
    if f1 > best_val_f1:
        best_val_f1 = f1
        best_mlp_name = name

print(f"Best MLP Architecture: {best_mlp_name}")

# Final best MLP training with early stopping
final_mlp = create_mlp(architectures[best_mlp_name])
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

start_time = time.time()
history_mlp = final_mlp.fit(
    X_train_res, y_train_res, 
    validation_data=(X_test, y_test), 
    epochs=150, 
    callbacks=[early_stop], 
    verbose=0
)
final_mlp_time = time.time() - start_time

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history_mlp.history['loss'], label='Train Loss')
plt.plot(history_mlp.history['val_loss'], label='Val Loss')
plt.axvline(early_stop.best_epoch, color='red', linestyle='--', label=f'Early Stop ({early_stop.best_epoch})')
plt.title('C2 - Best MLP: Loss', fontweight='bold')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_mlp.history['accuracy'], label='Train Acc')
plt.plot(history_mlp.history['val_accuracy'], label='Val Acc')
plt.axvline(early_stop.best_epoch, color='red', linestyle='--')
plt.title('C2 - Best MLP: Accuracy', fontweight='bold')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/c2_mlp_history.png', dpi=150)
plt.close()

y_prob_mlp = final_mlp.predict(X_test, verbose=0).flatten()
y_pred_mlp = (y_prob_mlp > 0.5).astype(int)

mlp_metrics = evaluate_model("Best MLP", y_test, y_pred_mlp, y_prob_mlp)
mlp_metrics['time'] = final_mlp_time
mlp_metrics['name'] = best_mlp_name

# 5-fold CV for best MLP architecture
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv_acc, cv_f1 = [], []

for train_idx, val_idx in kf.split(X_train_res):
    model = create_mlp(architectures[best_mlp_name])
    model.fit(X_train_res.iloc[train_idx], y_train_res.iloc[train_idx], epochs=early_stop.best_epoch, verbose=0)
    y_p = (model.predict(X_train_res.iloc[val_idx], verbose=0).flatten() > 0.5).astype(int)
    cv_acc.append(accuracy_score(y_train_res.iloc[val_idx], y_p))
    cv_f1.append(f1_score(y_train_res.iloc[val_idx], y_p, average='macro'))

print(f"5-Fold CV MLP Acc: {np.mean(cv_acc):.4f} ± {np.std(cv_acc):.4f}")
print(f"5-Fold CV MLP F1:  {np.mean(cv_f1):.4f} ± {np.std(cv_f1):.4f}")

# Save MLP model
final_mlp.save('outputs/best_mlp.h5')

# C3: Ablation Study
print("\\nC3 — Ablation Study")

def train_ablation(variant_name, remove_dropout=False, replace_relu=False, remove_es=False):
    model = Sequential()
    act = 'sigmoid' if replace_relu else 'relu'
    arch = architectures[best_mlp_name]
    
    model.add(Dense(arch[0], input_dim=X_train_res.shape[1], activation=act))
    if not remove_dropout: model.add(Dropout(0.3))
    
    for units in arch[1:]:
        model.add(Dense(units, activation=act))
        if not remove_dropout: model.add(Dropout(0.3))
        
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    
    epochs = 150 if remove_es else 150
    callbacks = [] if remove_es else [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    
    hist = model.fit(X_train_res, y_train_res, validation_data=(X_test, y_test), epochs=epochs, callbacks=callbacks, verbose=0)
    y_p = (model.predict(X_test, verbose=0).flatten() > 0.5).astype(int)
    f1 = f1_score(y_test, y_p, average='macro')
    
    return f1, hist.history['val_loss']

f1_A, loss_A = train_ablation("A: No Dropout", remove_dropout=True)
f1_B, loss_B = train_ablation("B: Sigmoid Activations", replace_relu=True)
f1_C, loss_C = train_ablation("C: No Early Stopping", remove_es=True)

print(f"Baseline (Best MLP): {mlp_metrics['f1']:.4f}")
print(f"Variant A (No Dropout): {f1_A:.4f}")
print(f"Variant B (Sigmoid): {f1_B:.4f}")
print(f"Variant C (No Early Stop): {f1_C:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(history_mlp.history['val_loss'], label='Baseline (Best MLP)', lw=2)
plt.plot(loss_A, label='A: No Dropout')
plt.plot(loss_B, label='B: Sigmoid')
plt.plot(loss_C, label='C: No Early Stop')
plt.title('C3 - Ablation Study: Validation Loss', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Val Loss')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/c3_ablation.png', dpi=150)
plt.close()

# ============================================================
# PART B3: Comparison & ROC
# ============================================================
print("\\n" + "="*60)
print("PART B3: Comparison & ROC")
print("="*60)

table_data = [
    ["Best MLP", f"{mlp_metrics['acc']:.3f}", f"{mlp_metrics['f1']:.3f}", f"{mlp_metrics['auc']:.3f}", f"{mlp_metrics['rec_1']:.3f}", f"{mlp_metrics['time']:.2f}s"],
    ["Random Forest", f"{rf_metrics['acc']:.3f}", f"{rf_metrics['f1']:.3f}", f"{rf_metrics['auc']:.3f}", f"{rf_metrics['rec_1']:.3f}", "-"],
    ["XGBoost", f"{xgb_metrics['acc']:.3f}", f"{xgb_metrics['f1']:.3f}", f"{xgb_metrics['auc']:.3f}", f"{xgb_metrics['rec_1']:.3f}", "-"]
]

df_comp = pd.DataFrame(table_data, columns=["Classifier", "Accuracy", "Macro F1", "AUC-ROC", "Recall (Disease)", "Train Time"])
print(df_comp.to_string(index=False))

plt.figure(figsize=(8, 6))
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_prob_mlp)

plt.plot(fpr_mlp, tpr_mlp, label=f"Best MLP (AUC = {mlp_metrics['auc']:.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {rf_metrics['auc']:.3f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {xgb_metrics['auc']:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.title('B3 - ROC Curve Comparison', fontweight='bold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/b3_roc_comparison.png', dpi=150)
plt.close()

print("\\nRecommendation (B3):")
print("We compare Random Forest, XGBoost, and the Best MLP.")
print("In a clinical setting, recall for the disease class (minimizing false negatives) is paramount.")
print("We should choose the model that maintains a high AUC while achieving the highest recall for the positive class.")
