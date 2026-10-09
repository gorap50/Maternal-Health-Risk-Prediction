# Maternal Health Risk Prediction

Multi-class classification model predicting prenatal risk level (Low / Mid / High) from clinical vitals.
Best result: **Random Forest — ~85% accuracy** · Automated data pipeline saving ~20 hrs/week of manual entry.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189F50?style=flat-square&logo=xgboost&logoColor=white)

---

## Results

| Model | Accuracy |
|---|---|
| **Random Forest** | **~85%** |
| XGBoost | — |
| SVM | — |
| Logistic Regression | — |
| KNN / Decision Tree | — |

Evaluated with accuracy score, confusion matrix, classification report, and ROC-AUC (one-vs-rest multiclass).
Model serialized with `joblib` (.pkl) and XGBoost native (.json) for deployment.

---

## Dataset

[Kaggle — Maternal Health Risk Data Set](https://www.kaggle.com/datasets/andrewmvd/maternal-health-risk-data)  
Features: Age, Systolic BP, Diastolic BP, Blood Sugar, Body Temperature, Heart Rate  
Target: Risk Level → High (0) / Low (1) / Mid (2)

---

## Pipeline

1. **EDA** — distributions, correlation heatmap, class balance analysis
2. **Preprocessing** — SMOTE for class imbalance, MinMaxScaler, feature correlation pruning
3. **Modeling** — Random Forest, XGBoost, SVM, Logistic Regression, KNN, Decision Tree
4. **Persistence** — saved models for inference (`model.pkl`, `xgBoost.json`)

---

## Run

```bash
git clone https://github.com/gorap50/maternal-health-risk-prediction.git
cd maternal-health-risk-prediction
pip install -r requirements.txt
jupyter notebook "Maternal Health Risk Prediction.ipynb"
```

**Load for inference:**
```python
import joblib
model = joblib.load("model.pkl")
predictions = model.predict(X_new)
```

---

## Project Structure

```
├── Maternal Health Risk Prediction.ipynb
├── Maternal Health Risk Data Set.csv
├── model.pkl
├── xgBoost.json
├── requirements.txt
└── README.md
```
