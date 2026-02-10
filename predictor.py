# Predictor CLI de Resistencia del Concreto
# Carga el modelo entrenado en concrete.py y permite predecir
# la resistencia de una mezcla ingresando sus 8 componentes.

import numpy as np
import pandas as pd
import joblib
import os

# Cargar modelo, scaler y lista de features
ruta_script = os.path.dirname(os.path.abspath(__file__))
ruta_modelo = os.path.join(ruta_script, 'modelo_concreto.pkl')

try:
    datos = joblib.load(ruta_modelo)
except FileNotFoundError:
    print("Error: No se encontró 'modelo_concreto.pkl'.")
    print("Ejecuta primero 'python concrete.py' para generar el modelo.")
    exit(1)

modelo = datos['modelo']
scaler = datos['scaler']
feature_cols = datos['feature_cols']
nombre_modelo = datos['mejor_nombre']

print("=" * 60)
print("PREDICTOR DE RESISTENCIA DEL CONCRETO")
print(f"Modelo: {nombre_modelo}")
print("=" * 60)
print("Ingresa los valores de la mezcla para obtener la predicción.")
print("Escribe 'salir' para terminar.\n")

# Los 8 componentes originales que el usuario debe ingresar
componentes = [
    ('cement', 'Cemento (kg/m³)'),
    ('slag', 'Escoria de alto horno (kg/m³)'),
    ('ash', 'Ceniza volante (kg/m³)'),
    ('water', 'Agua (kg/m³)'),
    ('superplastic', 'Superplastificante (kg/m³)'),
    ('coarse_agg', 'Agregado grueso (kg/m³)'),
    ('fine_agg', 'Agregado fino (kg/m³)'),
    ('age', 'Edad (días)')
]

while True:
    print("-" * 60)
    valores = {}
    salir = False

    for key, label in componentes:
        while True:
            entrada = input(f"  {label}: ")
            if entrada.strip().lower() == 'salir':
                salir = True
                break
            try:
                valor = float(entrada)
                if key == 'age' and valor <= 0:
                    print("    La edad debe ser mayor a 0.")
                    continue
                if key == 'cement' and valor <= 0:
                    print("    El cemento debe ser mayor a 0.")
                    continue
                if valor < 0:
                    print("    El valor no puede ser negativo.")
                    continue
                valores[key] = valor
                break
            except ValueError:
                print("    Ingresa un número válido.")

    if salir:
        print("\nHasta luego.")
        break

    # Recrear features de ingeniería (misma lógica que Fase 2 de concrete.py)
    cement = valores['cement']
    slag = valores['slag']
    ash = valores['ash']
    water = valores['water']
    superplastic = valores['superplastic']
    coarse_agg = valores['coarse_agg']
    fine_agg = valores['fine_agg']
    age = valores['age']

    # Ratios
    water_cement_ratio = water / cement
    fine_coarse_ratio = fine_agg / coarse_agg
    supp_cement_ratio = (slag + ash) / cement

    # Interacciones
    cement_superplastic = cement * superplastic
    cement_age = cement * age
    water_cement_product = water * cement

    # Transformación log
    age_log = np.log1p(age)

    # Construir vector de features en el mismo orden que feature_cols
    feature_map = {
        'cement': cement,
        'slag': slag,
        'ash': ash,
        'water': water,
        'superplastic': superplastic,
        'coarse_agg': coarse_agg,
        'fine_agg': fine_agg,
        'age': age,
        'water_cement_ratio': water_cement_ratio,
        'fine_coarse_ratio': fine_coarse_ratio,
        'supp_cement_ratio': supp_cement_ratio,
        'cement_superplastic': cement_superplastic,
        'cement_age': cement_age,
        'water_cement_product': water_cement_product,
        'age_log': age_log,
    }

    # Armar DataFrame en el orden exacto del modelo
    try:
        X_input = pd.DataFrame([[feature_map[col] for col in feature_cols]], columns=feature_cols)
    except KeyError as e:
        print(f"\nError: feature no reconocida: {e}")
        print(f"Features esperadas: {feature_cols}")
        continue

    # Escalar y predecir
    X_scaled = pd.DataFrame(scaler.transform(X_input), columns=feature_cols)
    prediccion = modelo.predict(X_scaled)[0]

    print(f"\n  >>> Resistencia estimada: {prediccion:.2f} MPa <<<\n")
