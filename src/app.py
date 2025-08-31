
import streamlit as st
import pandas as pd
import joblib
import os


st.set_page_config(page_title="Predicción de Clúster con K-Means", layout="centered")


try:
    kmeans_model = joblib.load("src/kmeans_model.joblib")
    scaler_clustering = joblib.load("src/scaler_clustering.joblib")

except FileNotFoundError:
    st.error("Error: Archivos de modelo o escalador no encontrados. Asegúrese de que 'kmeans_model.joblib' y 'scaler_clustering.joblib' estén en el mismo directorio que 'app.py'.")
    st.stop() 



st.markdown("""
<style>
.main-title {
    color: #2E86C1;
    text-align: center;
    font-size: 2.5em;
    margin-bottom: 20px;
    text-shadow: 2px 2px 5px #cccccc;
}
.stNumberInput label {
    font-weight: bold;
    color: #34495E;
    font-size: 1.1em;
}
.stButton button {
    background-color: #2ECC71;
    color: white;
    font-weight: bold;
    padding: 10px 20px;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
.stButton button:hover {
    background-color: #28B463;
}
.prediction-output {
    margin-top: 30px;
    padding: 20px;
    background-color: #EAECEE;
    border-left: 6px solid #3498DB;
    border-radius: 8px;
    box-shadow: 2px 2px 10px #cccccc;
}
.prediction-output h3 {
    color: #3498DB;
    margin-top: 0;
}
.prediction-output p {
    font-size: 1.2em;
    color: #2C3E50;
</style>
""", unsafe_allow_html=True)

cluster_descriptions = {
    0: "Área de vivienda de alto valor y densidad media",
    1: "Área de vivienda de muy alto valor y baja densidad",
    2: "Área de vivienda de valor medio y alta densidad",
    3: "Área de vivienda de bajo valor y baja densidad",
    4: "Área de vivienda de valor medio y densidad muy alta",
    5: "Área de vivienda de valor bajo y densidad media"
}

#  Cargar datos originales y calcular rangos 

try:

    original_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv")
    data_ranges = {
        "MedInc": (original_data["MedInc"].min(), original_data["MedInc"].max()),
        "Latitude": (original_data["Latitude"].min(), original_data["Latitude"].max()),
        "Longitude": (original_data["Longitude"].min(), original_data["Longitude"].max())
    }
    data_load_success = True

except Exception as e:
    st.warning(f"No se pudieron cargar los datos originales para mostrar los rangos. Error: {e}")
    data_load_success = False
    data_ranges = {} 



# Título principal con clase personalizada

st.markdown('<h1 class="main-title">Predicción de Clúster de Vivienda (K-Means)</h1>', unsafe_allow_html=True)

st.write("Ingrese las características de la vivienda para predecir a qué clúster pertenece.")



# Rangos de Datos de Entrenamiento

if data_load_success:
    st.header("Rangos de Datos:")
    st.write(f"- **Ingreso Medio (MedInc):** {data_ranges['MedInc'][0]:.4f} a {data_ranges['MedInc'][1]:.4f}")
    st.write(f"- **Latitud:** {data_ranges['Latitude'][0]:.4f} a {data_ranges['Latitude'][1]:.4f}")
    st.write(f"- **Longitud:** {data_ranges['Longitude'][0]:.4f} a {data_ranges['Longitude'][1]:.4f}")
    st.markdown("---")



# Campos de entrada para el usuario

st.header("Datos de la Vivienda:")
medinc = st.number_input("Ingreso Medio (MedInc)", min_value=0.0, value=float(data_ranges.get("MedInc", (0.0, 10.0))[0]), format="%.4f", help="Ingrese el ingreso medio del bloque en decenas de miles de dólares.")
latitude = st.number_input("Latitud", value=float(data_ranges.get("Latitude", (0.0, 10.0))[0]), format="%.4f", help="Ingrese la latitud de la ubicación de la vivienda.")
longitude = st.number_input("Longitud", value=float(data_ranges.get("Longitude", (0.0, 10.0))[0]), format="%.4f", help="Ingrese la longitud de la ubicación de la ubicación de la vivienda.")



# Boton para activar la predicción
if st.button("Predecir Clúster"):

    input_data = pd.DataFrame([[medinc, latitude, longitude]], columns=["MedInc", "Latitude", "Longitude"])

    input_data_scaled = scaler_clustering.transform(input_data)

    predicted_cluster = kmeans_model.predict(input_data_scaled)[0]

    predicted_description = cluster_descriptions.get(predicted_cluster, f"Clúster desconocido: {predicted_cluster}")

    # Resultado de la prediccion
    st.markdown('<div class="prediction-output">', unsafe_allow_html=True)
    st.subheader("Resultado de la Predicción:")
    st.write(f"Según las características ingresadas, la vivienda probablemente pertenece al clúster: **{predicted_description}**")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("Aplicación web para demostración de K-Means.")
