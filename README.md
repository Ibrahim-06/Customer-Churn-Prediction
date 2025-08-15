
# 📊 Telco Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-pandas%2C%20scikit--learn%2C%20xgboost%2C%20imbalanced--learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Overview
This project focuses on predicting **customer churn** for a telecom company using various machine learning algorithms. The goal is to identify customers who are likely to leave the service, enabling the business to take proactive measures to retain them.

The dataset used is **Telco-Customer-Churn.csv**, containing customer demographics, account details, and service usage patterns.

---

## 📂 Project Structure
```
├── Data/
│   └── Telco-Customer-Churn.csv
├── Model/
│   └── Logistic_model.pkl
├── Scaler/
│   └── scaler.pkl
├── Features/
│   └── features.pkl
├── App/
│   └── app.py          # Streamlit App for predictions
├── Notebook/
│   └── churn_analysis.ipynb
├── README.md
├── Stteamlit.png
└── BY_ME.bat
```

---

## 🛠️ Features
- **Data Cleaning** (handling missing values, removing duplicates, converting data types)
- **Exploratory Data Analysis (EDA)** with Seaborn and Matplotlib
- **Feature Engineering** (One-Hot Encoding for categorical variables)
- **Model Training** with:
  - Logistic Regression
  - Decision Tree (with GridSearchCV tuning)
  - Random Forest
  - SVM
  - KNN
  - XGBoost
  - Soft Voting Classifier (Ensemble Learning)
- **Imbalanced Data Handling** with SMOTE
- **Model Saving** using Joblib
- **Interactive Streamlit App** for prediction

---

## 📊 Data Preprocessing
- Dropped **customerID**
- Converted `TotalCharges` to numeric and imputed missing values with median
- Removed duplicate entries
- Split into train/test with stratification
- Standardized numerical features using `StandardScaler`

---

## 📈 Model Evaluation
Metrics used:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Threshold tuning was applied to improve recall for churned customers.

---

## 🚀 Installation & Usage
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ibrahim-06/Customer-Churn-Prediction
.git
cd Customer-Churn-Prediction
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run Jupyter Notebook (for training and analysis)
```bash
jupyter notebook Notebook/churn_analysis.ipynb
```

### 4️⃣ Run Streamlit App
```bash
streamlit run App/app.py
```

---

## 📹 Demo 
>  <img width="3770" height="2122" alt="Streamlit" src="https://github.com/user-attachments/assets/76042d3d-63dc-4e83-9485-b5d395534509" />


---

## 📊 Results Snapshot
| Model                 | Accuracy | Precision | Recall | F1-score |
|-----------------------|----------|-----------|--------|----------|
| Logistic Regression   | 0.82     | 0.78      | 0.93   | 0.84     |
| Random Forest         | 0.85     | 0.80      | 0.88   | 0.84     |
| XGBoost               | 0.86     | 0.81      | 0.89   | 0.85     |
| Voting Classifier     | 0.87     | 0.83      | 0.88   | 0.85     |

---

## 📦 Saved Artifacts
- **Logistic_model.pkl** → Trained logistic regression model
- **scaler.pkl** → StandardScaler fitted on training data
- **features.pkl** → List of feature names after preprocessing

---

## 📚 Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
joblib
streamlit
```

---

## 👤 Author
**Ibrahim Mohamed**  
📧 Email: ibrahim.06.dev@gmail.com  
🔗 LinkedIn: [[Ibrahim Mohamed](https://www.linkedin.com/in/ibrahim-mohamed-211-)](#)

---

## 📜 License
This project is licensed under the MIT License.
