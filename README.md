# 📊 Customer Churn Prediction with Streamlit

This project includes an interactive, beautifully designed **Streamlit web app** with:

- 🎯 Animated headers  
- 🌫️ Blur background  
- 🖱️ Hover effects on inputs  
- ✨ Stylish buttons and layout  

---

## 🚀 How to Run the App

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run App/app.py
```

---

## 📁 Project Structure

```
your-repo-name/
│
├── App/
│   └── app.py                 # Streamlit app
│
├── Model/
│   └── Logistic_model.pkl     # Trained Logistic Regression model
│
├── Scaler/
│   └── scaler.pkl             # Fitted StandardScaler
│
├── Features/
│   └── features.pkl           # List of features used for training
│
├── Data/
│   └── Telco-Customer-Churn.csv
│
├── requirements.txt
└── README.md
```

---

## 🎥 Demo Video

👉 [Watch the App Demo](https://your-video-link.com)

---

## ✅ Model Performance

- **Recall**: 95% after applying SMOTE and threshold tuning  
- Focused on minimizing false negatives for churn prediction
