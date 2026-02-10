# App Streamlit - Predictor de Resistencia del Concreto

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Cargar modelo
ruta_script = os.path.dirname(os.path.abspath(__file__))
ruta_modelo = os.path.join(ruta_script, 'modelo_concreto.pkl')

@st.cache_resource
def cargar_modelo():
    return joblib.load(ruta_modelo)

try:
    datos = cargar_modelo()
except FileNotFoundError:
    st.error("No se encontró 'modelo_concreto.pkl'. Ejecuta primero `python concrete.py`.")
    st.stop()

modelo = datos['modelo']
scaler = datos['scaler']
feature_cols = datos['feature_cols']
nombre_modelo = datos['mejor_nombre']

# --- UI ---
st.title("Predictor de Resistencia del Concreto")
st.caption(f"Modelo: {nombre_modelo}")

st.markdown("Ingresa los componentes de la mezcla para estimar la resistencia (MPa).")

# Inputs organizados en dos columnas
col1, col2 = st.columns(2)

with col1:
    cement = st.number_input("Cemento (kg/m\u00b3)", min_value=1.0, value=300.0, step=10.0)
    slag = st.number_input("Escoria de alto horno (kg/m\u00b3)", min_value=0.0, value=50.0, step=10.0)
    ash = st.number_input("Ceniza volante (kg/m\u00b3)", min_value=0.0, value=100.0, step=10.0)
    water = st.number_input("Agua (kg/m\u00b3)", min_value=1.0, value=180.0, step=5.0)

with col2:
    superplastic = st.number_input("Superplastificante (kg/m\u00b3)", min_value=0.0, value=6.0, step=1.0)
    coarse_agg = st.number_input("Agregado grueso (kg/m\u00b3)", min_value=1.0, value=1000.0, step=10.0)
    fine_agg = st.number_input("Agregado fino (kg/m\u00b3)", min_value=1.0, value=750.0, step=10.0)
    age = st.number_input("Edad (d\u00edas)", min_value=1, value=28, step=1)

if st.button("Predecir resistencia", type="primary"):
    # Features de ingeniería (misma lógica que Fase 2 de concrete.py)
    feature_map = {
        'cement': cement,
        'slag': slag,
        'ash': ash,
        'water': water,
        'superplastic': superplastic,
        'coarse_agg': coarse_agg,
        'fine_agg': fine_agg,
        'age': float(age),
        'water_cement_ratio': water / cement,
        'fine_coarse_ratio': fine_agg / coarse_agg,
        'supp_cement_ratio': (slag + ash) / cement,
        'cement_superplastic': cement * superplastic,
        'cement_age': cement * float(age),
        'water_cement_product': water * cement,
        'age_log': np.log1p(float(age)),
    }

    X_input = pd.DataFrame([[feature_map[col] for col in feature_cols]], columns=feature_cols)
    X_scaled = pd.DataFrame(scaler.transform(X_input), columns=feature_cols)
    prediccion = modelo.predict(X_scaled)[0]

    st.markdown("---")
    st.metric("Resistencia estimada", f"{prediccion:.2f} MPa")

    # Resumen de la mezcla
    with st.expander("Detalle de la mezcla"):
        resumen = pd.DataFrame({
            'Componente': ['Cemento', 'Escoria', 'Ceniza', 'Agua',
                           'Superplastificante', 'Agregado grueso', 'Agregado fino', 'Edad'],
            'Valor': [f"{cement:.1f} kg/m\u00b3", f"{slag:.1f} kg/m\u00b3", f"{ash:.1f} kg/m\u00b3",
                      f"{water:.1f} kg/m\u00b3", f"{superplastic:.1f} kg/m\u00b3",
                      f"{coarse_agg:.1f} kg/m\u00b3", f"{fine_agg:.1f} kg/m\u00b3", f"{age} d\u00edas"],
            'Ratio clave': ['-', '-', '-', f"w/c = {water/cement:.3f}",
                            '-', '-', f"f/c = {fine_agg/coarse_agg:.3f}", '-']
        })
        st.table(resumen)
