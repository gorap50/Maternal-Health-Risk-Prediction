# Maternal Health Risk Prediction 🚼🩺

A comprehensive machine learning project focused on predicting maternal health risk levels — **low**, **mid**, or **high** — using clinical data. This model helps in early detection of risks to improve maternal care outcomes.

---

## 📖 Project Overview

Maternal health complications are a leading cause of mortality worldwide. This project leverages advanced machine learning algorithms to classify pregnant women into risk categories based on vital health parameters, supporting timely interventions and better healthcare decisions.

---

## ⚙️ Technologies & Tools

- **Programming Language:** Python  
- **Libraries:**  
  - Data manipulation & visualization: `pandas`, `numpy`, `matplotlib`, `seaborn`  
  - Machine Learning: `scikit-learn`, `xgboost`, `imblearn`  
  - Model persistence: `joblib`  
- **Environment:** Google Colab / Jupyter Notebook

---

## 🗂 Dataset Description

- **Source:** [Kaggle - Maternal Health Risk Data Set](https://www.kaggle.com/datasets/andrewmvd/maternal-health-risk-data)  
- **Features:** Age, Systolic BP, Diastolic BP, Blood Sugar, Body Temperature, Heart Rate  
- **Target Variable:** Risk Level (`High risk`, `Mid risk`, `Low risk` mapped to 0, 2, 1 respectively)  
- **Size:** Approximate number of records and features (mention if known)

---

## 🔍 Exploratory Data Analysis (EDA)

- Checked for null and duplicate values  
- Visualized data distributions with histograms and boxplots  
- Correlation heatmap to identify feature relationships  
- Class distribution shown through pie and bar charts  
- Outlier removal and data cleaning performed

---

## 🛠 Data Preprocessing

- Label encoding for categorical target  
- Handling imbalanced classes using **SMOTE**  
- Feature scaling using **MinMaxScaler**  
- Removal of highly correlated features to reduce redundancy

---

## 🤖 Machine Learning Models Used

- Random Forest Classifier  
- XGBoost Classifier  
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Decision Tree Classifier

### Best Model:  
Random Forest achieved highest accuracy (~85%) and strong ROC AUC scores for multiclass classification.

---

## 📈 Model Evaluation

- Accuracy Score  
- Confusion Matrix  
- Classification Report (Precision, Recall, F1-Score)  
- ROC AUC (One-vs-Rest for multi-class)

---

## 💾 Model Persistence

- Saved models in `.pkl` and `.json` formats for deployment and reuse  
- Used `joblib` for loading and inference in Python environments

---

## 🚀 How to Run This Project

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/maternal-health-risk-prediction.git
    cd maternal-health-risk-prediction
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Jupyter Notebook:**
    - Open `Maternal Health Risk Prediction.ipynb` and execute all cells  
    - Or upload the notebook to Google Colab and run online

4. **Load the trained model for predictions:**
    ```python
    import joblib
    model = joblib.load("model.pkl")
    # Use model.predict() for inference
    ```

---

## 📁 Project Structure

├── Maternal Health Risk Prediction.ipynb # Main notebook with code & analysis
├── Maternal Health Risk Data Set.csv # Dataset used for training and evaluation
├── model.pkl # Serialized Random Forest model
├── xgBoost.json # Serialized XGBoost model
├── requirements.txt # Python dependencies
└── README.md # This file


---

## 🔮 Future Enhancements

- Deploy model with a web app interface (e.g., Flask, Streamlit)  
- Add real-time prediction API for healthcare providers  
- Expand dataset with more health parameters  
- Experiment with deep learning models for better accuracy

---

## 🤝 Acknowledgements

- Dataset by Andrew Mvd on Kaggle  
- Python ML ecosystem: scikit-learn, xgboost, imblearn  
- Open source community contributors

---

## 📬 Contact

**Gorap Kumar**  
- Email: [gorapkumar11@gmail.com](mailto:gorapkumar11@gmail.com)  
- GitHub: [github.com/yourusername](https://github.com/yourusername)

---

⭐ If you found this project useful, please give it a star!

---

*Thank you for visiting!*

