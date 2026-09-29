# 🎗️ Breast Cancer Diagnostic Risk Predictor

An end-to-end clinical decision support tool powered by a **Gradient Boosting Classifier** and deployed via an interactive **Streamlit** web application. The model predicts the likelihood of breast cancer (Benign vs. Malignant) based on patient vitals, clinical indicators, and cumulative lifestyle risk factors.

---

## 📌 Key Highlights
* **Dataset:** 10,000 patient records encompassing clinical, demographic, genetic, and lifestyle features.
* **Champion Model:** **Gradient Boosting Classifier** achieving **99.40% Accuracy**, **1.000 Precision**, and **0.9976 ROC-AUC**.
* **Pre-processing & Pipeline:** Scikit-Learn `ColumnTransformer` with median/mode imputation, `StandardScaler`, and `OneHotEncoder`.
* **Feature Engineering:** Implemented a compounding `Risk_Factor_Count` index capturing cumulative genetic and lifestyle risk burden.
* **Leakage-Free Architecture:** Excluded post-biopsy ground truths and non-informative noise (`Annual_Income_USD`, `Patient_ID`).
* **Deployment:** Real-time web UI using **Streamlit** with risk badges, confidence scoring, and health factor analytics.

---

## 📊 Model Benchmark

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** ⭐ | **99.40%** | **1.000** | **0.9690** | **0.9843** | **0.9976** |
| Decision Tree | 97.75% | 0.9407 | 0.9432 | 0.9419 | 0.9644 |
| Random Forest | 97.35% | 0.9744 | 0.8863 | 0.9283 | 0.9940 |
| Logistic Regression | 93.70% | 0.8462 | 0.8243 | 0.8351 | 0.9761 |

---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn
* **App Framework:** Streamlit
* **Model Serialization:** Joblib

---

## 📂 Project Structure
```text
├── app.py                         # Streamlit web application
├── cancer_prediction_pipeline.pkl # Serialized trained ML pipeline
├── model.ipynb                    # EDA, feature engineering & model training notebook
├── breast_cancer_prediction.csv   # Dataset
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
