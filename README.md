# DS-3002 Data Mining — Assignment #4: Heartbeat to Heatmap

**Spring 2026 · BSDS · FAST-NUCES**

A complete machine learning pipeline covering Unsupervised Learning, Ensemble Methods, Neural Networks, and CNNs for heart disease prediction and handwritten digit recognition.

---

## Project Structure

```
heart-disease-ml-pipeline/
├── notebooks/
│   ├── assignment4.ipynb         ← Main notebook (all parts: Pre, A, B, C, D)
│   └── processed.cleveland.data  ← Heart Disease dataset (UCI Cleveland)
├── app/
│   ├── app.py                    ← Streamlit dashboard (Part E)
│   ├── model.pkl                 ← Saved best model
│   └── requirements.txt          ← App-specific dependencies
├── report/                       ← PDF/DOCX report goes here
├── requirements.txt              ← Full environment dependencies
└── README.md
```

---

## Dataset Download (Heart Disease — UCI Cleveland)

1. Go to: <https://archive.ics.uci.edu/dataset/45/heart+disease>
2. Click the **Download (125.0 kB)** button (top-right of the page).
3. Unzip the downloaded file. You will see 4 `.data` files.
4. **Use only** `processed.cleveland.data` — place it inside the `notebooks/` folder.
5. MNIST loads automatically via `from tensorflow.keras.datasets import mnist` — no download needed.

---

## Environment Setup

```bash
# Clone the repository
git clone https://github.com/Minato-sudo/heart-disease-ml-pipeline.git
cd heart-disease-ml-pipeline

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

---

## Running the Notebook

```bash
source venv/bin/activate
cd notebooks
jupyter notebook assignment4.ipynb
```

- Restart the kernel and **Run All** cells in order to reproduce all results.
- All random seeds are fixed to `random_state=42`.

---

## Running the Streamlit App (Part E — Local Dashboard)

```bash
source venv/bin/activate
cd app
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**.

---

## Parts Covered

| Part | Topic | Marks |
|------|-------|-------|
| Pre | Preprocessing & Data Setup | 12 |
| A | Unsupervised Learning (K-Means, Hierarchical, PCA, t-SNE) | 20 |
| B | Bagging & Boosting (Random Forest, XGBoost) | 22 |
| C | ANN / SLP / MLP on Tabular Data + Ablation | 20 |
| D | CNN on MNIST Digits | 16 |
| E | Local Front-End Dashboard (Streamlit) | 10 |

---

## Key References

- Detrano, R. et al. (1989). UCI Heart Disease Dataset.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Chen, T. & Guestrin, C. (2016). XGBoost. *KDD '16*.
- Lundberg, S.M. & Lee, S.I. (2017). SHAP. *NeurIPS*.
- LeCun, Y. et al. (1998). Gradient-Based Learning Applied to Document Recognition. *Proceedings of the IEEE*.