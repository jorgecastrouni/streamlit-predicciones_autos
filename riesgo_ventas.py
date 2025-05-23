import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from PIL import Image

# --- Custom Styling ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #111827; /* Dark background */
    }
    .st-container {
        background-color: #374151; /* Darker container */
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 0 15px rgba(255, 255, 255, 0.1);
        color: white;
    }
    h1, h2, h3, h4, h5, h6, p, div, .st-header, .st-subheader, .streamlit-image-caption, .st-info {
        color: white !important;
    }
    .logo-container {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        overflow: hidden;
        margin-right: 15px;
        display: inline-block;
        vertical-align: middle;
    }
    .logo-container img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    .st-selectbox > div > div > div > div {
        background-color: #4b5563; /* Dark selectbox */
        border-radius: 8px;
        color: white;
    }
    .st-slider > div > div > div > div[data-testid="stThumb"] {
        background-color: #64748b;
    }
    .st-slider > div > div > div > div[data-testid="stTrack"] {
        background-color: #71717a;
    }
    .st-button > button {
        background-color: #4CAF50; /* Green button */
        color: white;
        border-radius: 8px;
        font-weight: bold;
        padding: 10px 20px;
    }
    .st-button > button:hover {
        background-color: #45a049;
    }
    .st-info {
        background-color: #1d4ed8;
        color: white !important;
        border: 1px solid #3b82f6;
        border-radius: 0.25rem;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
    }
    .prediction-text {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Load the Model ---
try:
    with open('modelo-clas-tree-knn-nn.pkl', 'rb') as file:
        model_Knn, model_Tree, model_NN, labelencoder, model_variables, min_max_scaler = pickle.load(file)
except FileNotFoundError:
    st.error("El archivo del modelo 'modelo-clas-tree-knn-nn.pkl' no se encontró. Asegúrate de que esté en la misma carpeta que este script.")
    st.stop()
except Exception as e:
    st.error(f"Ocurrió un error al cargar el modelo: {e}")
    st.stop()

# --- Main Content ---
st.container()

# --- Logo and Title ---
col1, col2 = st.columns([1, 5])
with col1:
    try:
        image = Image.open("seguro1.jpg")
        st.image(image, width=60)
    except FileNotFoundError:
        st.info("No se encontró la imagen 'seguro1.jpg' en la ruta especificada.")

with col2:
    st.header("Predicción de Riesgo")

st.subheader("Clasificador")

# --- Parameters ---
age = st.number_input("Seleccione la edad", min_value=18, max_value=100, value=33, step=1)
vehicle_type = st.selectbox("Seleccione el tipo de vehiculo", ["combi", "family", "sport", "minivan"])
selected_model_name = st.selectbox("Modelo", ["Nn", "Knn", "Dt"]) # NN first as per last interaction
st.write(f"Modelo Seleccionado: {selected_model_name}")

if st.button("Realizar Predicción"):
    # Crear DataFrame con los datos del usuario
    user_data = pd.DataFrame({'age': [age], 'cartype': [vehicle_type]}) # Cambiamos 'vehicle_type' a 'cartype'

    # Preprocesamiento (One-Hot Encoding para el tipo de vehículo)
    user_data = pd.get_dummies(user_data, columns=['cartype'], drop_first=False) # Usamos 'cartype' aquí

    # Asegurarse de que las columnas estén en el mismo orden que durante el entrenamiento
    if 'model_variables' in locals():
        processed_data = pd.DataFrame(columns=model_variables)
        for col in user_data.columns:
            if col in processed_data.columns:
                processed_data[col] = user_data[col]
        processed_data = processed_data.fillna(0) # Llenar las columnas faltantes con 0
        st.write("Datos preprocesados para la predicción:")
        st.dataframe(processed_data)
    else:
        st.error("Las variables del modelo no se cargaron correctamente.")
        st.stop()


    # Seleccionar el modelo basado en la elección del usuario
    if selected_model_name == "Knn":
        selected_model = model_Knn
    elif selected_model_name == "Dt":
        selected_model = model_Tree
    elif selected_model_name == "Nn":
        selected_model = model_NN
    else:
        st.error("Modelo no reconocido")
        st.stop()

    try:
        # Realizar la predicción
        prediction = selected_model.predict(processed_data)

        # Mostrar la predicción
        st.subheader("La predicción es:")
        if prediction[0] == 0:  # 0 represents high risk
            st.markdown("<p class='prediction-text'>Alto Riesgo</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='prediction-text'>Bajo Riesgo</p>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Ocurrió un error durante la predicción: {e}")

st.subheader("Clasificador")
st.write("Valores seleccionados:")
st.write(f"Edad: {age}")
st.write(f"Tipo de Vehículo: {vehicle_type}")
