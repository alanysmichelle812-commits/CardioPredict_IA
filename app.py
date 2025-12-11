# app.py (Versión FINAL con 10 Features y Precisión 81.97%)
import streamlit as st
import pandas as pd
import joblib
import os 
import numpy as np

# --- 1. CONFIGURACIÓN INICIAL DE LA EMPRESA ---
st.set_page_config(
    page_title="CardioPredict S.A. - Plataforma de IA para Riesgo Cardíaco",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar el modelo entrenado con la ruta robusta
MODEL_ACCURACY = 0.8197
# Usamos la ruta más robusta para el servidor
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'cardio_model.joblib') 

try:
    model = joblib.load(MODEL_PATH) 
except FileNotFoundError:
    st.error(f"Error: El archivo del modelo '{MODEL_PATH}' no se encontró. Asegúrate de que la carpeta 'model' y el archivo 'cardio_model.joblib' existen.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

st.title("🩺 CardioPredict S.A. - Análisis de Riesgo Cardiovascular IA")
st.markdown(f"""
**CardioPredict S.A. ofrece un servicio digital avanzado para la evaluación predictiva de enfermedades cardíacas.**
Nuestro modelo **XGBoost optimizado**, entrenado con datos clínicos reales, proporciona un análisis rápido y fiable para la toma de decisiones clínicas (**Precisión: {MODEL_ACCURACY:.2%}**).
---
""")

# --- 2. ENTRADA DE DATOS DEL PACIENTE (10 FEATURES) ---
st.subheader("👤 Datos Clínicos del Paciente (10 Parámetros Clave)")
col1, col2, col3 = st.columns(3)

sex_options = {"Femenino (0)": 0, "Masculino (1)": 1}
angina_options = {"No (0)": 0, "Sí (1)": 1}

with col1:
    st.markdown("##### Información Básica")
    age = st.slider("1. Edad (años)", 30, 75, 50, key="age")
    sex_label = st.selectbox("2. Sexo", list(sex_options.keys()), key="sex_label")
    sex = sex_options[sex_label.split(" ")[0]] 
    
    st.markdown("##### Tipo de Dolor de Pecho (CP)")
    chest_pain_type = st.select_slider("3. Tipo de Dolor en el Pecho (0-3)", options=[0, 1, 2, 3], key="cp_type")

with col2:
    st.markdown("##### Parámetros Metabólicos")
    resting_bp = st.number_input("4. Presión Arterial en Reposo (trestbps)", 90, 200, 120, key="resting_bp")
    cholesterol = st.number_input("5. Colesterol Sérico (chol)", 120, 564, 200, key="chol")
    
    fasting_bs_label = st.selectbox("6. Azúcar en Sangre en Ayunas (>120 mg/dl)", ["No (0)", "Sí (1)"], key="fbs_label")
    fasting_bs = 1 if "Sí" in fasting_bs_label else 0

with col3:
    st.markdown("##### Índices Cardíacos y Riesgo")
    max_hr = st.number_input("7. Frecuencia Cardíaca Máxima Alcanzada (thalach)", 70, 202, 150, key="max_hr")
    exercise_angina_label = st.selectbox("8. Angina Inducida por Ejercicio (exang)", list(angina_options.keys()), key="angina_label")
    exercise_angina = angina_options[exercise_angina_label.split(" ")[0]] 
    
    st_depression = st.number_input("9. Depresión del Segmento ST (oldpeak)", 0.0, 6.5, 1.0, step=0.1, key="st_dep")
    num_major_vessels = st.slider("10. Vasos Principales Coloreados (ca, 0-3)", 0, 3, 0, key="num_vessels")


# --- 3. PREDICCIÓN Y RESULTADOS ---
st.markdown("---")
if st.button("💰 Generar Reporte de Riesgo (Servicio Premium)", type="primary"):
    
    features = pd.DataFrame([[
        age, sex, chest_pain_type, resting_bp, cholesterol, 
        fasting_bs, max_hr, exercise_angina, st_depression, 
        num_major_vessels
    ]], columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                 'FastingBS', 'MaxHR', 'ExerciseAngina', 'ST_Depression', 
                 'NumMajorVessels'])
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[:, 1][0]
    
    if prediction == 1:
        st.error(f"### 🚨 ALTO RIESGO DETECTADO (Probabilidad de Enfermedad: {probability:.2%})")
        st.write("---")
        st.markdown(
            f"""
            **REPORTE DE NEGOCIO PREMIUM - CardioPredict S.A.:** El paciente muestra un patrón de características clínicas **altamente correlacionadas con la enfermedad cardíaca**. 
            
            **Recomendación de la IA:** Se sugiere una **consulta urgente con un cardiólogo** para pruebas confirmatorias y la iniciación de un plan de tratamiento preventivo agresivo. (Confianza del Modelo: {MODEL_ACCURACY:.2%}).
            """
        )
        
    else:
        st.success(f"### ✅ RIESGO BAJO (Probabilidad de Enfermedad: {probability:.2%})")
        st.write("---")
        st.markdown(
            f"""
            **REPORTE DE NEGOCIO PREMIUM - CardioPredict S.A.:** El perfil clínico actual del paciente indica un **Riesgo Cardiovascular Bajo** según nuestros modelos predictivos. 
            
            **Recomendación de la IA:** Mantener un seguimiento periódico. (Confianza del Modelo: {MODEL_ACCURACY:.2%}).
            """
        )

# --- 4. MODELO DE NEGOCIO ---
st.sidebar.title("💳 Modelo de Ingreso")
st.sidebar.markdown(
    """
    **CardioPredict S.A. opera bajo un modelo SaaS B2B:**
    
    * **Valor de Mercado:** La precisión de **81.97%** es la clave de nuestro servicio premium.
    * **Tarifas:** Básico ($99 USD/mes) o Empresarial ($399 USD/mes).
    """
)
st.sidebar.markdown("---")
st.sidebar.info("© 2025 CardioPredict S.A. | Innovación en Ingeniería Biomédica")
