import streamlit as st
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------
# Load Model dan Dataset
# ---------------------------
@st.cache_resource
def load_model():
    # Model KNN dilatih ulang di sini agar aplikasi tetap berdiri sendiri
    data = pd.read_csv("dataset_balita.csv")
    X = data[['Jenis_Kelamin', 'Umur', 'Tinggi_Badan']]
    y = data['Status_Gizi']
    
    # Encoding jenis kelamin (Laki-laki: 1, Perempuan: 0)
    X['Jenis_Kelamin'] = X['Jenis_Kelamin'].map({'Laki-laki': 1, 'Perempuan': 0})
    
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    return model

model = load_model()

# ---------------------------
# UI Streamlit
# ---------------------------
st.title("Deteksi Stunting pada Balita")

# Input
umur = st.number_input("Umur (bulan)", min_value=0, max_value=60, value=24)
jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
tinggi_badan = st.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=120.0, value=80.0)

# Prediksi
if st.button("Prediksi"):
    jk_encoded = 1 if jenis_kelamin == "Laki-laki" else 0
    input_data = pd.DataFrame([[jk_encoded, umur, tinggi_badan]],
                              columns=['Jenis_Kelamin', 'Umur', 'Tinggi_Badan'])
    
    hasil = model.predict(input_data)[0]
    st.success(f"Status Gizi Balita: **{hasil}**")
