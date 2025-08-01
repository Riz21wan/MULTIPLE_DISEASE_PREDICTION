import streamlit as st


st.set_page_config(page_title="Multi-Disease Prediction App 🏥", page_icon="🩺")

# --- Imports ---
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# --- Load models ---
parkinsons_model = pickle.load(open("dataset for multiple disease prediction/parkinsons_model.sav", 'rb'))
heart_model = pickle.load(open("dataset for multiple disease prediction/heart_disease_model.sav", 'rb'))
diabetes_model = pickle.load(open("dataset for multiple disease prediction/diabetes_model.sav", 'rb'))


# --- Define expected features ---
FEATURES = {
    "Parkinson's Disease": [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
        'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1',
        'spread2', 'D2', 'PPE'
    ],
    "Heart Disease": [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ],
    "Diabetes": [ 
        'Pregnancies',	'Glucose',	'BloodPressure',	'SkinThickness',	'Insulin',	'BMI',	'DiabetesPedigreeFunction',	'Age'

    ]
}


# --- Functions ---
def load_file(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file)
        elif file.name.endswith('.json'):
            return pd.read_json(file)
        elif file.name.endswith('.txt'):
            return pd.read_csv(file, delimiter=r'[\t,;|]', engine='python')
        else:
            st.error("Unsupported format. Upload CSV, Excel, JSON, or TXT.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None

def process_bp(data):
    if "Blood_Pressure (mmHg)" in data.columns:
        try:
            bp = data["Blood_Pressure (mmHg)"].str.split("/", expand=True)
            data["Systolic_BP"] = pd.to_numeric(bp[0], errors='coerce')
            data["Diastolic_BP"] = pd.to_numeric(bp[1], errors='coerce')
            data.drop(columns=["Blood_Pressure (mmHg)"], inplace=True)
        except:
            st.warning("⚠️ Could not parse Blood Pressure column correctly.")
    return data

def predict_with_available_features(model, model_features, df):
    available = [f for f in model_features if f in df.columns]
    missing = [f for f in model_features if f not in df.columns]

    # Ensure columns are of type string
    df.columns = df.columns.astype(str)

    filtered_df = df.copy()
    filtered_df = filtered_df[available]

    for col in missing:
        filtered_df[col] = 0

    filtered_df = filtered_df[model_features]
    input_data = filtered_df.astype(float)

    return model.predict(input_data)


# --- Main App ---
def main():
    st.title("🩺 Smart Health Disease Predictor")
    st.markdown("""
    Upload your medical report, and predict **Parkinson's**, **Heart Disease**, or **Diabetes** instantly.  
    Supported file types: **CSV**, **Excel**, **JSON**, **TXT**.
    """)

    st.sidebar.header("Settings ⚙️")
    selected_models = st.sidebar.multiselect(
        "Select Models to Predict:",
        ["Parkinson's Disease", "Heart Disease", "Diabetes"],
        default=["Parkinson's Disease", "Heart Disease", "Diabetes"]
    )

    uploaded_file = st.file_uploader("📤 Upload your Health Report", type=["csv", "xlsx", "xls", "json", "txt"])

    if uploaded_file:
        df = load_file(uploaded_file)
        if df is not None:
            df = process_bp(df)

            st.subheader("📄 Uploaded Report Preview")
            st.dataframe(df.head())

            st.subheader("🧪 Prediction Results")

            models = {
                "Parkinson's Disease": parkinsons_model,
                "Heart Disease": heart_model,
                "Diabetes": diabetes_model
            }

            try:
                for model_name in selected_models:
                    pred = predict_with_available_features(models[model_name], FEATURES[model_name], df)

                    if pred[0] == 1:
                        st.error(f"🔴 {model_name}: Positive")
                        st.info(f"⚠️ Recommendation: Consult a specialist for {model_name.lower()} check-up.")
                    else:
                        st.success(f"🟢 {model_name}: Negative")
                        st.balloons()

            except Exception as e:
                st.error(f"❌ Prediction Error: {e}")

# --- Run App ---
if __name__ == "__main__":
    main()
