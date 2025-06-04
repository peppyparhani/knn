import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Load dan Siapkan Data
# ---------------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("data_balita.csv")
    
    # Encode jenis kelamin
    df['Jenis Kelamin'] = df['Jenis Kelamin'].map({'laki-laki': 1, 'perempuan': 0})
    
    # Fitur dan Label
    X = df[['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)']]
    y = df['Status Gizi']
    
    # Encode label target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Model
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y_encoded)
    
    return model, le

model, label_encoder = load_model()

# ---------------------------
# UI Streamlit
# ---------------------------
st.title("📊 Deteksi Stunting Balita dengan KNN")

umur = st.number_input("Umur Balita (bulan)", min_value=0, max_value=60, value=24)
jenis_kelamin = st.selectbox("Jenis Kelamin", ["laki-laki", "perempuan"])
tinggi = st.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=120.0, value=80.0)

if st.button("Prediksi"):
    jk = 1 if jenis_kelamin == "laki-laki" else 0
    input_data = [[umur, jk, tinggi]]
    pred = model.predict(input_data)[0]
    hasil = label_encoder.inverse_transform([pred])[0]
    
    st.success(f"✅ Status Gizi Balita: **{hasil.upper()}**")
