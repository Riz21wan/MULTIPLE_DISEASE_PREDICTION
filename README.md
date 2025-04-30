# 🩺 Multi-Disease Prediction App 🏥

This Streamlit-based web application allows users to upload health report files and predict the likelihood of three major diseases:
- **Parkinson's Disease**
- **Heart Disease**
- **Diabetes**

The app utilizes pre-trained machine learning models to provide instant predictions based on user-uploaded data.

---

## 🚀 Features

- 📂 Upload medical reports in **CSV**, **Excel**, **JSON**, or **TXT** formats.
- 🧠 Predict using multiple models simultaneously.
- 🧪 Displays prediction results in an intuitive and user-friendly format.
- 🎈 Visual feedback for healthy results (e.g., balloons).
- ⚠️ Warnings and recommendations for high-risk cases.

---

## 🧰 Technologies Used

- Python 🐍
- [Streamlit](https://streamlit.io/)
- Pandas, NumPy
- Scikit-learn
- Pickle (for loading pre-trained ML models)

---
## 🎯 Purpose
The main objective of this application is to provide users—patients, health professionals, and researchers—with a fast, accessible, and intelligent tool to:

Analyze uploaded medical reports

Predict multiple diseases simultaneously

Get immediate visual feedback and health recommendations

This tool is especially useful in remote settings, for quick self-assessments, or as an aid in clinical workflows.
## 📄 Input File Requirements
streamlit
pandas
numpy
scikit-learn
openpyxl

✅ Parkinson's Disease Columns:

'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1',
'spread2', 'D2', 'PPE'

✅ Heart Disease Columns:

'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'

✅ Diabetes Columns:

'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'

---

## 📦 Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/multi-disease-predictor.git
cd multi-disease-predictor
