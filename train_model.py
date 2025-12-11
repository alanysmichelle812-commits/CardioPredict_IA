# train_model.py (Versión Final con XGBoost Optimizado)
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier  # <--- CAMBIO CLAVE
from sklearn.metrics import accuracy_score
import joblib
import os

# --- 1. Carga del Dataset Real (Heart Disease UCI) ---
def load_and_prepare_data(file_path='heart.csv'):
    """Carga el dataset real y selecciona las 10 características."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo '{file_path}'. Asegúrate de que esté en la carpeta.")
        return None

    # Seleccionar las 10 características que coinciden con app.py
    # Estas columnas coinciden con el dataset UCI estándar:
    df.columns = [
        'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
        'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
        'ST_Depression', 'Slope', 'NumMajorVessels', 'Thal', 'HeartDisease'
    ]
    
    X = df[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
            'FastingBS', 'MaxHR', 'ExerciseAngina', 'ST_Depression', 
            'NumMajorVessels']].copy()
    
    y = df['HeartDisease']
    
    return X, y

# --- 2. Entrenamiento y Optimización con XGBoost ---
def train_and_save_optimized_model():
    X, y = load_and_prepare_data()
    if X is None:
        return

    print("Iniciando la OPTIMIZACIÓN del modelo (XGBoost) para MÁXIMA precisión...")
    print("Esto puede tardar un poco más debido a la optimización Grid Search.")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Definir los hiperparámetros de XGBoost a probar
    param_grid = {
        'n_estimators': [50, 100, 200], # Número de árboles
        'learning_rate': [0.05, 0.1, 0.2], # Tasa de aprendizaje
        'max_depth': [3, 5] # Profundidad de cada árbol
    }
    
    # Usar XGBClassifier
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42) 
    
    # Inicializar GridSearchCV
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Evaluación
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n--- RESULTADOS DE OPTIMIZACIÓN XGBoost ---")
    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Precisión del Modelo Optimizado (XGBoost): {accuracy:.4f}")
    
    # --- 3. Guardar el Mejor Modelo ---
    model_dir = 'model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_filename = os.path.join(model_dir, 'cardio_model.joblib')
    joblib.dump(best_model, model_filename) 
    
    print(f"Modelo de ALTA PRECISIÓN guardado en: {model_filename}")

if __name__ == "__main__":
    train_and_save_optimized_model()