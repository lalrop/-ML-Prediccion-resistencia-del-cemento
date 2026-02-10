# Analisis de la resistencia del concreto.
# Este es un trabajo de análisis exploratorio de datos (EDA) sobre un dataset de Sina Mehdinia dentro de uno de los proyectos de Feature Engineering. en Kaggle.
# Para este aplicaremos todas las herramientas de ML aplicables a este caso.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuramos la visualizacion
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================
# FASE 1: CARGA Y PREPARACIÓN DE DATOS
# ============================================================


# Construir ruta al archivo Excel relativa al script
ruta_script = os.path.dirname(os.path.abspath(__file__))
df = pd.read_excel(os.path.join(ruta_script, 'Concrete_Data.xls'))

# Los nombres de las columnas son excesivamente largos, los renombramos para facilitar su uso
df.columns = ['cement', 'slag', 'ash', 'water', 'superplastic',
              'coarse_agg', 'fine_agg', 'age', 'strength']

print("=" * 60)
print("FASE 1: ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("=" * 60)

# 1.1 PRIMER VISTAZO A LOS DATOS

print("DIMENSIONES DEL DATASET")
print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

print("PRIMERAS FILAS")
print(df.head())

print("TIPOS DE DATOS Y VALORES NULOS")
print(df.info())

print("VALORES NULOS POR COLUMNA")
print(df.isnull().sum())

# # 1.2 ESTADÍSTICAS DESCRIPTIVAS
# Seguimos con una visualizacion rapida sobre las estadisticas descriptivas del dataset

print("ESTADÍSTICAS DESCRIPTIVAS")
print(df.describe().T.round(2))

# # 1.3 ANÁLISIS DE LA VARIABLE OBJETIVO (strength)
# ok, identificamos nuestra variable objetivo que en este caso es 'strength' (resistencia del concreto en MPa) y obtenemos sus estadisticas descriptivas

print("[1.6] ANÁLISIS DE LA VARIABLE OBJETIVO: strength")
print(f"Media:    {df['strength'].mean():.2f} MPa")
print(f"Mediana:  {df['strength'].median():.2f} MPa")
print(f"Std:      {df['strength'].std():.2f} MPa")
print(f"Min:      {df['strength'].min():.2f} MPa")
print(f"Max:      {df['strength'].max():.2f} MPa")
print(f"Rango:    {df['strength'].max() - df['strength'].min():.2f} MPa")

# # Asimetría y curtosis
# Estas metricas nos ayudan a entender la forma de la distribucion de la variable objetivo en la linea de la simetria y la presencia de colas pesadas o ligeras
# Por que hacemos esto? Porque nos ayuda a decidir si necesitamos aplicar transformaciones para normalizar la variable objetivo antes de modelar
# Una distribucion simetrica tiene un valor 0 de asimetria, mientras que una distribucion con colas pesadas tiene una curtosis alta
print(f"Asimetría (skewness): {df['strength'].skew():.3f}")
print(f"Curtosis:             {df['strength'].kurtosis():.3f}")

# Una asimetria positiva indica que la cola derecha es mas larga y tiene valores mas extremos. Luego, un curtosis de -0.314 indica que la distribucion es mas plana
# con colas mas ligeras y menos valores extremos de lo esperado.

# 1.4 MATRIZ DE CORRELACIONES
# La matriz de correlaciones es una herramienta que nos permite conocer la relacion entre las variables del dataset y la variable objetivo.
# En este caso el objetivo es saber que variable tiene una mayor correlacion con la variable "strength".


print("[1.7] CORRELACIONES CON LA VARIABLE OBJETIVO")

correlaciones = df.corr()['strength'].drop('strength').sort_values(ascending=False)
print(correlaciones.round(3))

# Al observar la matriz de correlaciones se destaca "cement" y "superplastic" como las variables con mayor correlacion positiva sobre "strength". Luego,
# "water" y "fine_agg" se muestran como las variables con mayor correlacion negativa pero en menor impacto que las positivas.
# 1.5 VISUALIZACIONES
# Preparamos el espacio donde se mostrara la visualizacion de datos, designaremos un area de 2x2 donde se mostraran cada uno de los graficos a continuacion.

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Distribución de la variable objetivo
# Veamos como se distribuye la variable objetivo "strength"

ax1 = axes[0, 0]
sns.histplot(df['strength'], kde=True, ax=ax1, color='steelblue')
ax1.axvline(df['strength'].mean(), color='red', linestyle='--', label=f"Media: {df['strength'].mean():.1f}")
ax1.axvline(df['strength'].median(), color='green', linestyle='--', label=f"Mediana: {df['strength'].median():.1f}")
ax1.set_title('Distribución de Resistencia del Concreto')
ax1.set_xlabel('Resistencia (MPa)')
ax1.legend()

# 2. Boxplot de todas las variables (normalizadas para comparar)
# Aqui buscamos conocer como se distribuyen las variables del dataset, veremos si existen valores atipicos y como se comparan entre si. A fin de poder compararlas, las normalizamos. y luego generamos el boxplot.

ax2 = axes[0, 1]
df_normalized = (df - df.min()) / (df.max() - df.min())
df_normalized.boxplot(ax=ax2, rot=45)
ax2.set_title('Boxplot de Variables (Normalizadas)')
ax2.set_ylabel('Valor normalizado [0-1]')

# 3. Heatmap de correlaciones
# El heatmap permite ver graficamente lo que ya identificamos en la matriz de correlaciones; es decir, que variables tienen mayor o menor correlacion con la variable objetivo y que sentido.

ax3 = axes[1, 0]
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax3)
ax3.set_title('Matriz de Correlaciones')

# 4. Top correlaciones con strength
# Aqui basicamente buscamos identificar visualmente las variables mas correlacionadas con nuestra variable objetivo.

ax4 = axes[1, 1]
correlaciones.plot(kind='barh', ax=ax4, color=['green' if x > 0 else 'red' for x in correlaciones])
ax4.set_title('Correlación de Features con Strength')
ax4.set_xlabel('Coeficiente de Correlación')
ax4.axvline(x=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(ruta_script, 'eda_visualizaciones.png'), dpi=150)
plt.show()

print("\n[INFO] Visualizaciones guardadas en 'eda_visualizaciones.png'")

# ============================================================
# FASE 2: INGENIERÍA DE FEATURES Y PREPROCESAMIENTO
# ============================================================

print("\n" + "=" * 60)
print("FASE 2: INGENIERÍA DE FEATURES Y PREPROCESAMIENTO")
print("=" * 60)

# 2.1 DETECCIÓN Y TRATAMIENTO DE OUTLIERS
# Los outliers son valores atípicos que se alejan significativamente del resto de los datos.
# Usamos el método IQR (Rango Intercuartílico) donde un valor es outlier si cae por debajo de
# Q1 - 1.5*IQR o por encima de Q3 + 1.5*IQR.

print("\n[2.1] DETECCIÓN Y ELIMINACIÓN DE OUTLIERS (Método IQR)")

features = df.columns.drop('strength')
outliers_count = {}

for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outliers_count[col] = n_outliers

print("Outliers detectados por variable:")
for col, count in outliers_count.items():
    print(f"  {col:15s}: {count:3d} outliers")

# Guardamos las dimensiones originales para comparar después de eliminar outliers
filas_antes = df.shape[0]

# Eliminamos filas que contengan outliers en cualquier feature
# Construimos una máscara booleana para identificar con True = fila sin outliers (se mantiene)
mask = pd.Series(True, index=df.index)
for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = mask & (df[col] >= lower) & (df[col] <= upper)

df = df[mask].reset_index(drop=True)

print(f"\nFilas antes:   {filas_antes}")
print(f"Filas después: {df.shape[0]}")
print(f"Filas eliminadas: {filas_antes - df.shape[0]} ({(filas_antes - df.shape[0]) / filas_antes * 100:.1f}%)")

# 2.2 ANÁLISIS DE MULTICOLINEALIDAD (VIF)
# El VIF (Variance Inflation Factor) nos ayuda a dimensionar cuanto se correlacionan dos variables independientes entre si.
# Regla general: VIF > 5 indica multicolinealidad moderada, VIF > 10 indica severa.
# Si dos variables están muy correlacionadas entre sí, el modelo no puede distinguir
# cuál de las dos está causando el efecto sobre la variable objetivo.

print("\n[2.2] ANÁLISIS DE MULTICOLINEALIDAD (VIF)")
# Por que hacer este analisis? Una vez que generamos la matriz de correlacion vemos ademas de la correlacion con la variable objetivo, la correlacion entre las variables independientes. En este caso, vemos que "water" y "superplastic" tienen una correlacion de -0.66, lo que puede indicar multicolinealidad.
# Por lo mismo es necesario dimensionar cuan fuerte es esta correlacion para luego decidir si eliminar alguna de las dos o fusionarlas.

X_vif = df[features]
vif_data = pd.DataFrame({
    'Feature': features,
    'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(len(features))]
})
vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)

print(vif_data.to_string(index=False))
print("\nInterpretación: VIF > 5 = multicolinealidad moderada | VIF > 10 = severa")

# En base a lo observado en los valores VIF, existe una fuerte impacto de la multicolinealidad de variables como water. Esto tiene que ver sobre todo porque que el agua tiene directa relacion con la cantidad de cemento y todos los otros adivitivos mencionados. Si agregamos menos agua, implica menos cemento, y los otros aditivos.
# En este caso, decidimos mantener todas las variables por ahora, pero en un analisis mas profundo podriamos considerar eliminar o combinar algunas variables altamente correlacionadas.


# 2.3 RATIOS ENTRE COMPONENTES
# Para este caso, generamos los ratios entre los componentes dado que en la industria del concreto es comun analizar las proporciones entre los materiales en vez de los valores absolutos por si mismos..

print("\n[2.3] RATIOS ENTRE COMPONENTES")

df['water_cement_ratio'] = df['water'] / df['cement']

# Este ratio es fundamental en la industria del concreto, ya que afecta directamente la resistencia y durabilidad del concreto. De acuerdo con la ley de Abrams, a menor relación agua/cemento, mayor será la resistencia del concreto.

df['fine_coarse_ratio'] = df['fine_agg'] / df['coarse_agg']
# Este ratio afecta sobre la trabajabilidad y resistencia del concreto.

df['supp_cement_ratio'] = (df['slag'] + df['ash']) / df['cement']
# Este tiene que ver sobre si hablamos de un cemento convencional o con adiciones.

nuevos_ratios = ['water_cement_ratio', 'fine_coarse_ratio', 'supp_cement_ratio']

print("Correlación de nuevos ratios con strength:")
for ratio in nuevos_ratios:
    corr = df[ratio].corr(df['strength'])
    print(f"  {ratio:25s}: {corr:.3f}")

# Observamos entonces ahora una relacion negativa de todos los ratios con la variable "strength", siendo el relacionado al agua/cemento el de mayor impacto negativo.

# 2.4 INTERACCIONES ENTRE VARIABLES

# Las interacciones capturan efectos combinados que las variables individuales no pueden.
# Por ejemplo, el efecto del cemento sobre la resistencia puede depender de la edad del concreto.

print("\n[2.4] INTERACCIONES ENTRE VARIABLES")

df['cement_superplastic'] = df['cement'] * df['superplastic']
# Esta interaccion captura el efecto combinado de ambos, si tenemos mucho cemento con mucho superplastic, deberia impactar en una fortaleza mayor.
df['cement_age'] = df['cement'] * df['age']
# El cemento demanda tiempo para hidratarse y secarse. Por lo cual, un cemento con mayor edad deberia impactar en mayor resistencia.
df['water_cement_product'] = df['water'] * df['cement']
# Aqui aplicamos una relacion en terminos de volumen, a mayor volumen de ambos vamos a tener una mayor masa que implique una mayor resistencia, o no ...

nuevas_interacciones = ['cement_superplastic', 'cement_age', 'water_cement_product']
print("Correlación de interacciones con strength:")
for inter in nuevas_interacciones:
    corr = df[inter].corr(df['strength'])
    print(f"  {inter:25s}: {corr:.3f}")

# Ahora bien, observamos que las intereacciones trabajadas si tienen un impacto positivo en la resistencia del concreto. siendo la de mayor peso la intereaccion edad/cemento.

# 2.5 TRANSFORMACIONES (LOG, POLYNOMIAL)
# Aplicamos log a variables con asimetría alta (skewness > 1) para reducir el sesgo.
# Log solo funciona con valores positivos, así que sumamos 1 para evitar log(0).
# También exploramos si las relaciones no lineales mejoran con transformaciones polinómicas.

print("\n[2.5] TRANSFORMACIONES")

print("Asimetría (skewness) de las features originales:")
skewness = df[features].skew().sort_values(ascending=False)
print(skewness.round(3).to_string())

# Como se observa, age es la variable con mayor asimetria. Esto se debe a que tenemos pocas observaciones para edades altas y muchas observaciones en edades bajas.
# Aplicamos log1p (log(1+x)) a variables con skewness > 1
vars_sesgadas = skewness[skewness.abs() > 1].index.tolist()
print(f"\nVariables con |skewness| > 1: {vars_sesgadas}")

for col in vars_sesgadas:
    df[f'{col}_log'] = np.log1p(df[col])
    corr_original = df[col].corr(df['strength'])
    corr_log = df[f'{col}_log'].corr(df['strength'])
    print(f"  {col}: corr original={corr_original:.3f} -> corr log={corr_log:.3f}")

# Como se observa , la variable age mantiene la misma correlacion con strength luego de aplicar el log e incluso aumenta levemente su correlacion. Esto se debe a que el log ayuda a reducir la asimetria y hace que la relacion con strength sea mas lineal, lo que puede mejorar el rendimiento de modelos lineales.

# 2.6 ESCALADO / NORMALIZACIÓN
# StandardScaler transforma cada variable para que tenga media=0 y desviación estándar=1.
# Esto es importante para modelos sensibles a la escala (regresión lineal, SVM, KNN, redes neuronales).
# Modelos basados en árboles (Random Forest, XGBoost) NO necesitan escalado.

print("\n[2.6] ESCALADO (StandardScaler)")

# Separamos features y variable objetivo antes de escalar
# No escalamos la variable objetivo (strength)
feature_cols = [col for col in df.columns if col != 'strength']
X = df[feature_cols]
y = df['strength']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

print(f"Features escaladas: {len(feature_cols)}")
print(f"Media después de escalar (debe ser ~0):\n{X_scaled.mean().round(4).to_string()}")
print(f"\nStd después de escalar (debe ser ~1):\n{X_scaled.std().round(4).to_string()}")

# 2.7 DIVISIÓN TRAIN / TEST
# Dividimos los datos en entrenamiento (80%) y prueba (20%).
# random_state=42 asegura que la división sea reproducible (siempre el mismo resultado). EL numero 42 es una convencion clasica en la comunidad de ciencia de datos, podria ser cualquier otro valor.
# El modelo se SE ENTRENA CON TRAIN y SE EVALUA CON TEST (datos que nunca vio).

print("\n[2.7] DIVISIÓN TRAIN / TEST")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Conjunto de entrenamiento: {X_train.shape[0]} filas ({X_train.shape[0]/len(y)*100:.0f}%)")
print(f"Conjunto de prueba:        {X_test.shape[0]} filas ({X_test.shape[0]/len(y)*100:.0f}%)")
print(f"Features totales:          {X_train.shape[1]}")
print(f"\nMedia de strength (train): {y_train.mean():.2f} MPa")
print(f"Media de strength (test):  {y_test.mean():.2f} MPa")

# si bien estamos aumentando el numero de variables independientes sumando las interacciones y ratios, lo que buscamos es tener candidatos de features que tengan una mayor correlacion con la variable objetivo . Luego en el modelo mismo entendermos que es lo que realmente aporta valor y que no.

print("\n" + "=" * 60)
print("FASE 2 COMPLETADA")
print("=" * 60)
print(f"Dataset final: {X_train.shape[1]} features, {len(y)} observaciones")
print(f"Listo para Fase 3: Modelado")

# ============================================================
# FASE 3: MODELADO
# ============================================================
# Entrenamos múltiples modelos de regresión para predecir la resistencia del concreto.
# Comparamos desde un modelo simple (regresión lineal) hasta modelos más complejos
# (Random Forest, Gradient Boosting) para entender qué enfoque funciona mejor.

print("\n" + "=" * 60)
print("FASE 3: MODELADO")
print("=" * 60)

# Diccionario para almacenar los resultados de cada modelo
resultados = {}

# Función auxiliar para evaluar un modelo e imprimir sus métricas
def evaluar_modelo(nombre, modelo, X_train, X_test, y_train, y_test):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    resultados[nombre] = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'modelo': modelo, 'y_pred': y_pred}
    print(f"  R²:   {r2:.4f}")
    print(f"  RMSE: {rmse:.2f} MPa")
    print(f"  MAE:  {mae:.2f} MPa")
    return modelo

# 3.1 REGRESIÓN LINEAL
# La regresión lineal busca una relación lineal entre las features y la variable objetivo.
# Es el modelo más simple y sirve como punto de referencia.
# Si los demás modelos no superan a este, algo está mal.

print("\n[3.1] REGRESIÓN LINEAL (Baseline)")
evaluar_modelo('Regresión Lineal', LinearRegression(), X_train, X_test, y_train, y_test)

# 3.2 MODELOS REGULARIZADOS — RIDGE Y LASSO
# Ridge (L2) penaliza los coeficientes grandes, reduciendo el impacto de la multicolinealidad.
# Lasso (L1) puede reducir coeficientes a exactamente cero, eliminando features irrelevantes.
# Ambos son variantes de la regresión lineal con un término de penalización.

# el valor de alpha por defecto suele ser 1 tanto para Ridge como para Lasso. Para un analisis mas profundo se suele aplicar una validacion cruzada para encontrar el mejor valor de alpha que minimice el error del modelo.

print("\n[3.2a] RIDGE (Regularización L2)")
evaluar_modelo('Ridge', Ridge(alpha=1.0, random_state=42), X_train, X_test, y_train, y_test)

print("\n[3.2b] LASSO (Regularización L1)")
lasso = evaluar_modelo('Lasso', Lasso(alpha=0.1, random_state=42), X_train, X_test, y_train, y_test)

# Veamos qué features eliminó Lasso (coeficiente = 0)
coefs_lasso = pd.Series(lasso.coef_, index=feature_cols)
eliminadas = coefs_lasso[coefs_lasso == 0].index.tolist()
print(f"\n  Features eliminadas por Lasso: {len(eliminadas)}")
if eliminadas:
    for f in eliminadas:
        print(f"    - {f}")
else:
    print("    Ninguna (todas aportan al modelo)")

# 3.3 RANDOM FOREST
# Random Forest construye múltiples árboles de decisión y promedia sus predicciones.
# Ventajas: maneja relaciones no lineales, no necesita escalado, robusto ante outliers.
# No se ve afectado por la multicolinealidad.

print("\n[3.3] RANDOM FOREST")
rf = evaluar_modelo('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42),
                    X_train, X_test, y_train, y_test)

# Feature importance: cuánto contribuye cada variable a las predicciones del modelo
print("\n  Top 10 features más importantes:")
importances_rf = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
for feat, imp in importances_rf.head(10).items():
    print(f"    {feat:25s}: {imp:.4f}")

# 3.4 GRADIENT BOOSTING
# Gradient Boosting construye árboles secuencialmente, donde cada árbol corrige
# los errores del anterior. Suele ser más preciso que Random Forest pero más lento.

print("\n[3.4] GRADIENT BOOSTING")
gb = evaluar_modelo('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42),
                    X_train, X_test, y_train, y_test)

print("\n  Top 10 features más importantes:")
importances_gb = pd.Series(gb.feature_importances_, index=feature_cols).sort_values(ascending=False)
for feat, imp in importances_gb.head(10).items():
    print(f"    {feat:25s}: {imp:.4f}")

# 3.5 TABLA COMPARATIVA DE MODELOS
print("\n[3.5] COMPARACIÓN DE MODELOS")
print(f"{'Modelo':<25s} {'R²':>8s} {'RMSE':>10s} {'MAE':>10s}")
print("-" * 55)
for nombre, metricas in resultados.items():
    print(f"{nombre:<25s} {metricas['R2']:>8.4f} {metricas['RMSE']:>10.2f} {metricas['MAE']:>10.2f}")

# Identificar el mejor modelo por R²
mejor = max(resultados, key=lambda x: resultados[x]['R2'])
print(f"\nMejor modelo: {mejor} (R² = {resultados[mejor]['R2']:.4f})")

# 3.6 VISUALIZACIONES
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico 1: Comparación de R² entre modelos
ax1 = axes[0]
nombres = list(resultados.keys())
r2_valores = [resultados[n]['R2'] for n in nombres]
colores = ['green' if n == mejor else 'steelblue' for n in nombres]
ax1.barh(nombres, r2_valores, color=colores)
ax1.set_xlabel('R²')
ax1.set_title('Comparación de Modelos (R²)')
ax1.set_xlim(0, 1)

# Gráfico 2: Valores reales vs predichos del mejor modelo
ax2 = axes[1]
y_pred_mejor = resultados[mejor]['y_pred']
ax2.scatter(y_test, y_pred_mejor, alpha=0.5, color='steelblue', edgecolors='navy', s=30)
# Línea diagonal: si las predicciones fueran perfectas, todos los puntos caerían sobre esta línea
min_val = min(y_test.min(), y_pred_mejor.min())
max_val = max(y_test.max(), y_pred_mejor.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Predicción perfecta')
ax2.set_xlabel('Valor Real (MPa)')
ax2.set_ylabel('Valor Predicho (MPa)')
ax2.set_title(f'Real vs Predicho — {mejor}')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(ruta_script, 'modelos_comparacion.png'), dpi=150)
plt.show()

print("\n[INFO] Visualizaciones guardadas en 'modelos_comparacion.png'")

# print("\n" + "=" * 60)
# print("FASE 3 COMPLETADA")
# print("=" * 60)

# ============================================================
# FASE 4: EVALUACIÓN
# ============================================================
# Profundizamos en la evaluación de los modelos entrenados en Fase 3.
# Las métricas de un solo split train/test pueden ser engañosas (dependen de qué datos
# cayeron en cada conjunto). La validación cruzada y el análisis de residuos nos dan
# una imagen más completa de qué tan confiable es cada modelo.

print("\n" + "=" * 60)
print("FASE 4: EVALUACIÓN")
print("=" * 60)

# 4.1 VALIDACIÓN CRUZADA (Cross Validation)
# La validación cruzada divide los datos en k partes (folds), entrena k veces usando k-1 partes y evalúa
# con la parte restante. Así obtenemos k métricas y podemos calcular media y
# desviación estándar, lo que nos indica qué tan estable es el modelo.
# Un modelo con baja desviación estándar es más confiable.

print("\n[4.1] VALIDACIÓN CRUZADA (5-Fold)")

# Reconstruimos los modelos frescos para la validación cruzada
# (cross_val_score entrena internamente, así que necesitamos modelos sin entrenar)
modelos_cv = {
    'Regresión Lineal': LinearRegression(),
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'Lasso': Lasso(alpha=0.1, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

cv_resultados = {}

print(f"\n{'Modelo':<25s} {'R² medio':>10s} {'± Std':>10s} {'R² Test':>10s}")
print("-" * 57)

for nombre, modelo in modelos_cv.items():
    # scoring='r2' calcula R² en cada fold. cv=5 divide en 5 partes.
    # Usamos X_scaled e y (todo el dataset escalado) para que CV haga sus propios splits
    scores = cross_val_score(modelo, X_scaled, y, cv=5, scoring='r2')
    cv_resultados[nombre] = {'media': scores.mean(), 'std': scores.std(), 'scores': scores}
    r2_test = resultados[nombre]['R2']
    print(f"{nombre:<25s} {scores.mean():>10.4f} {scores.std():>10.4f} {r2_test:>10.4f}")

# Identificar el modelo más estable (menor std) y el mejor en promedio
mejor_cv = max(cv_resultados, key=lambda x: cv_resultados[x]['media'])
mas_estable = min(cv_resultados, key=lambda x: cv_resultados[x]['std'])

print(f"\nMejor modelo por CV:     {mejor_cv} (R² medio = {cv_resultados[mejor_cv]['media']:.4f})")
print(f"Modelo más estable:      {mas_estable} (std = {cv_resultados[mas_estable]['std']:.4f})")

# Si el mejor modelo en el split simple coincide con el mejor en CV, tenemos más confianza.
# Si no coinciden, puede indicar que el split simple fue favorable para un modelo particular.
if mejor == mejor_cv:
    print(f"\n  El mejor modelo coincide en evaluación simple y validación cruzada.")
else:
    print(f"\n  Atención: el mejor modelo difiere entre evaluación simple ({mejor})")
    print(f"  y validación cruzada ({mejor_cv}). CV es más confiable.")

# 4.2 ANÁLISIS DE RESIDUOS DEL MEJOR MODELO
# Los residuos (error = valor real - valor predicho) revelan patrones que las métricas
# globales no capturan. Un buen modelo debería tener:
# - Residuos distribuidos normalmente, centrados en 0 (sin sesgo sistemático)
# - Sin patrones visibles al graficar residuos vs predichos (homocedasticidad)

print("\n[4.2] ANÁLISIS DE RESIDUOS — " + mejor)

y_pred_mejor = resultados[mejor]['y_pred']
residuos = y_test.values - y_pred_mejor

print(f"  Media de residuos:     {residuos.mean():.4f} MPa (ideal: 0)")
print(f"  Std de residuos:       {residuos.std():.2f} MPa")
print(f"  Residuo máximo:        {residuos.max():.2f} MPa")
print(f"  Residuo mínimo:        {residuos.min():.2f} MPa")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico 1: Histograma de residuos
# Si los residuos siguen una distribución normal centrada en 0, el modelo no tiene sesgo.
# Un sesgo hacia positivo o negativo indica que subestima o sobreestima sistemáticamente.
ax1 = axes[0]
sns.histplot(residuos, kde=True, ax=ax1, color='steelblue', bins=25)
ax1.axvline(0, color='red', linestyle='--', label='Cero (ideal)')
ax1.axvline(residuos.mean(), color='orange', linestyle='--', label=f'Media: {residuos.mean():.2f}')
ax1.set_xlabel('Residuo (MPa)')
ax1.set_ylabel('Frecuencia')
ax1.set_title(f'Distribución de Residuos — {mejor}')
ax1.legend()

# Gráfico 2: Residuos vs Valores Predichos
# Si no hay patrones (nube aleatoria alrededor de 0), el modelo es consistente en todo el rango.
# Si aparece un embudo (más dispersión a la derecha), hay heterocedasticidad.
# Si aparece una curva, el modelo no captura una relación no lineal.
ax2 = axes[1]
ax2.scatter(y_pred_mejor, residuos, alpha=0.5, color='steelblue', edgecolors='navy', s=30)
ax2.axhline(0, color='red', linestyle='--')
ax2.set_xlabel('Valor Predicho (MPa)')
ax2.set_ylabel('Residuo (MPa)')
ax2.set_title(f'Residuos vs Predichos — {mejor}')

plt.tight_layout()
plt.savefig(os.path.join(ruta_script, 'evaluacion_residuos.png'), dpi=150)
plt.show()

print("\n[INFO] Visualizaciones guardadas en 'evaluacion_residuos.png'")

# 4.3 CONCLUSIONES Y RESUMEN FINAL
# Resumimos todo el pipeline: desde los datos crudos hasta el modelo final.

print("\n[4.3] CONCLUSIONES Y RESUMEN FINAL")
print("=" * 60)

print("\n--- Pipeline completo ---")
print(f"  1. Dataset original:     1030 observaciones, 8 features + 1 objetivo")
print(f"  2. Tras eliminar outliers: {df.shape[0]} observaciones")
print(f"  3. Features generadas:   {len(feature_cols)} (originales + ratios + interacciones + log)")
print(f"  4. Modelos evaluados:    {len(resultados)}")

print(f"\n--- Mejor modelo: {mejor} ---")
print(f"  R² (test):           {resultados[mejor]['R2']:.4f}")
print(f"  RMSE (test):         {resultados[mejor]['RMSE']:.2f} MPa")
print(f"  MAE (test):          {resultados[mejor]['MAE']:.2f} MPa")
print(f"  R² medio (5-fold CV): {cv_resultados[mejor]['media']:.4f} ± {cv_resultados[mejor]['std']:.4f}")

# Features más importantes según el mejor modelo (si es basado en árboles)
mejor_modelo = resultados[mejor]['modelo']
if hasattr(mejor_modelo, 'feature_importances_'):
    print(f"\n--- Top 5 features más importantes ({mejor}) ---")
    importances = pd.Series(mejor_modelo.feature_importances_, index=feature_cols).sort_values(ascending=False)
    for i, (feat, imp) in enumerate(importances.head(5).items(), 1):
        print(f"  {i}. {feat:25s}: {imp:.4f}")

print(f"\n--- Posibles mejoras futuras ---")
print(f"  - Optimización de hiperparámetros (GridSearchCV / RandomizedSearchCV)")
print(f"  - Probar XGBoost / LightGBM para potencialmente mejorar rendimiento")
print(f"  - Stacking o blending de los mejores modelos")
print(f"  - Selección automática de features (RFE, SelectFromModel)")

print("\n" + "=" * 60)
print("FASE 4 COMPLETADA — Proyecto finalizado")
print("=" * 60)

# ============================================================
# EXPORTACIÓN DEL MODELO PARA PREDICTOR CLI
# ============================================================

# ruta_modelo = os.path.join(ruta_script, 'modelo_concreto.pkl')
# joblib.dump({
#     'modelo': resultados[mejor]['modelo'],
#     'scaler': scaler,
#     'feature_cols': feature_cols,
#     'mejor_nombre': mejor
# }, ruta_modelo)

# print(f"\n[INFO] Modelo exportado en '{ruta_modelo}'")
