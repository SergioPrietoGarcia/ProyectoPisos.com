import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, RepeatedKFold, cross_validate
from sklearn.compose import ColumnTransformer, make_column_transformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
from xgboost import XGBRegressor
import xgboost as xgb

pd.set_option('display.max_rows', None) # Ver los dataframes completos

## CARGAMOS LOS DATOS
df_pisos = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Universidad\\Python\\Proyecto Inmobiliaria\\Proyecto Pisos.com\\Analisis Exploratorio de Datos\\pisos-pontevedra-machine-learning.csv')
df_pisos = df_pisos.drop(["titulo"], axis = 1)
df_pisos.head()
df_pisos.columns

# Variable "precio"
sns.pairplot(df_pisos[["precio"]])
plt.show()

# Variable "precio" con transformacion logaritmica
sns.pairplot(np.log10(df_pisos[["precio"]]))
plt.show()

# Variable ubicacion
df_pisos.ubicacion.value_counts()
# COMENTARIO: Al haber pocos casos en algunas ubicaciones (por ejemplo, As Neves o Cruces), cuando se vaya a ajustar
# el modelo y predecir, puede ser que en la muestra test aparezcan estas ubicaciones con las que el modelo no ha sido
# entrenado, causando un error. Por lo que, vamos a crear una nueva variable llamada ubicacion_nueva que agrupe las
# ubicaciones con menos de 10 casos en una categoria que sea "OtrasUbicaciones"

ciudades_muchos_pisos = [
    'Vigo', 'Pontevedra', 'Sanxenxo', 'Vilagarcía', 'Ponteareas', 'Cangas',
    'Poio', 'Marín', 'Guarda', 'Porriño', 'Salceda', 'Estrada', 'Salvaterra',
    'Tui', 'Redondela', 'Tomiño', 'Baiona', 'Bueu', 'Mondariz', 'Silleda',
    'Rosal', 'Moaña'
]
    
df_pisos["ubicacion_nueva"] = df_pisos['ubicacion'].apply(lambda x: 'OtrasUbicaciones' if x not in ciudades_muchos_pisos else x)
df_pisos.ubicacion_nueva.value_counts()
df_pisos.info() # Hay NAs, vamos a eliminarlos
df_pisos = df_pisos.dropna()

# TRANSFORMACION DE LAS VARIABLES CATEGORICAS
df_pisos_num = pd.get_dummies(df_pisos, columns = ["ubicacion_nueva","Planta", "Conservacion", "Calefaccion"], dummy_na=True)

df_pisos_num.head()
# COMENTARIO: Sin conocer el motivo se han creado columnas para los valores perdidos, que en principio son todo 0
df_pisos_num.Calefaccion_nan.sum()
df_pisos_num.Calefaccion_nan.sum()
df_pisos_num.Conservacion_nan.sum()
df_pisos_num.Planta_nan.sum()

df_pisos_num = df_pisos_num.drop(["ubicacion_nueva_nan", "Calefaccion_nan", "Conservacion_nan", "Planta_nan"], axis = 1)
df_pisos_num.columns

# Eliminamos las filas que presentan NAs
df_pisos_num = df_pisos_num.dropna()
df_pisos_num.isna().sum()


"""
MODELO 1. TRANSFORMACIÓN DE LA VARIABLE PRECIO (LOGARITMO EN BASE 10)
          XGBOOST SIN MODIFICAR SUS PARÁMETROS, POR DEFECTO
"""

## SELECCION DE LAS VARIABLES X e Y
x, y = df_pisos_num.drop(["precio", "ubicacion"], axis = 1), np.log10(df_pisos["precio"])
x.info()
x = x.astype(int)

## DIVISION EN MUESTRA DE ENTRENAMIENTO Y MUESTRA DE TEST
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2, random_state=99)
X_train.shape, X_test.shape


## MODELO DE REGRESION
model = XGBRegressor()


# Ajustamos el modelo
model.fit(X_train, Y_train)


# Representamos la importancia de las variables
xgb.plot_importance(model, ax = plt.gca())
plt.show()
# COMENTARIO: El orden de importancia de las variables es "Superficie", "Gastos de Comunidad"
# y "Habitaciones"

# Realizamos predicciones. Las predicciones se realizan prediciendo el 
# logaritmo en base 10 del precio. Aplicaremos una funcion exponencial
# para obtener el precio real y el precio estimado

    # Prediccion sobre los datos de entrenamiento
pred_train = model.predict(X_train)
print(r2_score(sp.special.exp10(Y_train),sp.special.exp10(pred_train)))
print(median_absolute_error(sp.special.exp10(Y_train),sp.special.exp10(pred_train)))


    # Prediccion sobre los datos de test
pred_test = model.predict(X_test)
print(r2_score(sp.special.exp10(Y_test),sp.special.exp10(pred_test)))
print(median_absolute_error(sp.special.exp10(Y_test),sp.special.exp10(pred_test)))
mae = median_absolute_error(sp.special.exp10(Y_test),sp.special.exp10(pred_test))

# Representacion de los resultados
fig, ax = plt.subplots(figsize = (5,5))
plt.scatter(sp.special.exp10(Y_test), sp.special.exp10(pred_test))
ax.plot(sp.special.exp10(Y_test), sp.special.exp10(Y_test), linestyle="--", color="red")
plt.text(500000, 200000, f"MAE on training set: {mae: .2f} euros")
plt.title("XGBoost Model")
plt.xlabel("Test Data")
plt.ylabel("Predict Data")
plt.show()

# COMENTARIO: El modelo ajustado mediante XGBoost ha logrado un valor del R^2 de 0.69
# aproximadamente para la muestra de test. Esto indica que el 69% de la varianza en la variable respuesta es 
# explicada por el modelo. Por otro lado, para la muestra de entrenamiento se ha alcanzado un r^2 de 0.96,
# aproximadamente

# COMENTARIO: Por otro lado, si estudiamos el MAE, observamos que se logra un valor de error
# de 27000 aproximadamente en la muestra de test. Se ha conseguido reducir el error que habiamos alcanzado al ajustar
# el modelo de Ridge Regression. Por otro lado, la prediccion en la muestra de entrenamiento es bastante acertada
# con un valor del MAE de 6279 aproximadamente, lo que claramente indica un sobreajuste.


"""
MODELO 2. TRANSFORMACIÓN DE LA VARIABLE PRECIO (LOGARITMO EN BASE 10)
          XGBOOST CON USO DE PIPELINES
"""

# En este caso, como se emplean los PIPELINES no es necesario
# crear directamente variables dummies, si no que lo que se hace
# es crear un preprocesador que realiza ese proceso al introducirlo
# en el modelo


df_pisos.head()
df_pisos = df_pisos.drop("ubicacion", axis = 1) # Nos quedamos con "ubicacion_nueva"

## SELECCION DE LAS VARIABLES X e Y
x, y = df_pisos.drop(["precio"], axis = 1), df_pisos["precio"]
x.info()

## DIVISION EN MUESTRA DE ENTRENAMIENTO Y MUESTRA DE TEST
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2, random_state=99)
X_train.shape, X_test.shape

# Convertir variables categoricas en nuevas variables con 1 y 0
variables_categoricas = ["ubicacion_nueva", "Planta", "Conservacion", "Calefaccion"]
variables_numericas = ["Superficie", "Habitaciones", "Baños", "Gastos de Comunidad",
                       "Amueblado", "Balcon", 'Ascensor', 'Calefaccion', 'Garaje', 
                       'Terraza', 'Trastero']
# Preprocesador
preprocesador = make_column_transformer(
    (OneHotEncoder(drop = "if_binary"), variables_categoricas), # Crea variables binarias y elimina las antiguas
    remainder = "passthrough", # Indica que no se deben cambiar las variables que no sean categoricas
    verbose_feature_names_out=False # Evita que se generen nombres de caracteristicas adicionales
)

# Creamos el modelo
model = make_pipeline(
    preprocesador,
    TransformedTargetRegressor(
        regressor = XGBRegressor(), func = np.log10, inverse_func=sp.special.exp10
    ),
)

# Ajustamos el modelo
model.fit(X_train, Y_train)

# Predicciones

    # Predicciones sobre la muestra de entrenamiento
y_pred = model.predict(X_train)
r2 = r2_score(Y_train, y_pred)
mae = median_absolute_error(Y_train, y_pred)
print(f"MAE on training set: {mae: .2f} euros")
print(f"R2 on training set: {r2: .2f}")

    # Predicciones sobre la muestra de test
y_pred = model.predict(X_test)
r2 = r2_score(Y_test, y_pred)
mae = median_absolute_error(Y_test, y_pred)
print(f"MAE on training set: {mae: .2f} euros")
print(f"R2 on training set: {r2: .2f}")

# Representacion de los resultados
fig, ax = plt.subplots(figsize = (5,5))
plt.scatter(Y_test, y_pred)
ax.plot(Y_test, Y_test, linestyle="--", color="red")
plt.text(500000, 200000, f"MAE on training set: {mae: .2f} euros")
plt.title("XGBoost Model")
plt.xlabel("Test Data")
plt.ylabel("Predict Data")
plt.show()

# COMENTARIO: Este modelo ha logrado un valor del R^2 de 0.64 en la muestra de test
# aproximadamente. Esto indica que el 64% de la varianza en la variable respuesta es 
# explicada por el modelo. En cambio, podemos observar que el valor del R^2 para
# la muestra de entrenamiento es de 0.96, lo que puede suponer un claro sobreajuste a 
# los datos de entrenamiento.

# COMENTARIO: Por otro lado, si estudiamos el MAE, observamos que se logra un valor de error
# de 29000 aproximadamente en la muestra de test. Ha aumentado ligeramente el error que habiamos alcanzado al ajustar
# el primer modelo XGBoost sin Pipelines. Por otro lado, la prediccion en la muestra de entrenamiento es tambien bastante acertada
# con un valor del MAE de 6500 aproximadamente, lo que claramente indica un sobreajuste, como se supuso anteriormente.

# COMENTARIO: Para tratar de corregir este sobreajuste, vamos emplear el uso de pipelines, la validación
# cruzada y la selección de hiperparámetros.




"""
MODELO 2. TRANSFORMACIÓN DE LA VARIABLE PRECIO (LOGARITMO EN BASE 10)
          XGBOOST CON USO DE PIPELINES Y SELECCION DE HIPERPARAMETROS
          USO DE LA VALIDACIÓN CRUZADA
"""

cv = RepeatedKFold(n_splits = 5, n_repeats=5, random_state = 99)
cv_model = cross_validate(
    model,
    x,
    y,
    cv = cv, 
    return_estimator=True,
    n_jobs = -1
)

feature_names = model[:-1].get_feature_names_out() # Nombre de las variables

# Inicializar una lista para almacenar la importancia de las características
coefs = []

for est, (train_idx, _) in zip(cv_model["estimator"], cv.split(x, y)):
    # Extraer el modelo de regresión XGBoost del estimador
    regressor = est[-1].regressor_
    
    # Obtener la importancia de las características
    feature_importance = regressor.feature_importances_
    
    # Crear una lista de importancia de características en el orden de feature_names
    importancia_caracteristicas = [feature_importance[i] for i in range(len(feature_names))]
    
    # Agregar la importancia de características a la lista
    coefs.append(importancia_caracteristicas)

# Asegurarse de que importancia_caracteristicas_lista tenga la forma adecuada
importancia_caracteristicas_lista = np.squeeze(coefs)

# Crear un DataFrame con la importancia de las características y los nombres de las características
coefs_df = pd.DataFrame(importancia_caracteristicas_lista, columns=feature_names)


# Representacion de las importancia de las variables y su variabilidad
plt.figure(figsize = (9,7))
sns.stripplot(data = coefs_df, orient = "h", alpha = 0.5)
sns.boxplot(data = coefs_df, orient = "h", color = "cyan", saturation=0.5, whis = 10)
plt.axvline(x = 0, color = ".5")
plt.xlabel("Coefficient importance")
plt.title("Coefficient importance and its variability")
plt.suptitle("XGBoost Model | Regularization")
plt.subplots_adjust(left = 0.3)
plt.show()


# COMENTARIO: En el grafico anterior se puede observar que la variabilidad de la 
# importancia de algunas variables como las de ubicacion, o incluso
# la variabilidad de la importancia de la variable baño, es demasiado
# grande. A continuacion, vamos a proceder a ajustar el modelo y
# observar sus metricas.

# Definir el espacio de búsqueda de hiperparámetros
param_grid = {
    'transformedtargetregressor__regressor__n_estimators': [100, 200, 300],
    'transformedtargetregressor__regressor__learning_rate': [0.01, 0.1, 0.2],
    'transformedtargetregressor__regressor__max_depth': [3, 4, 5],
    'transformedtargetregressor__regressor__min_child_weight': [1, 3, 5],
    'transformedtargetregressor__regressor__subsample': [0.8, 1.0],
    'transformedtargetregressor__regressor__colsample_bytree': [0.8, 1.0],
    'transformedtargetregressor__regressor__gamma': [0, 0.1],
    'transformedtargetregressor__regressor__reg_alpha': [0, 0.1],
    'transformedtargetregressor__regressor__reg_lambda': [0, 0.1],
    'transformedtargetregressor__regressor__objective': ['reg:squarederror']
}

# Crear el objeto GridSearchCV. Introducimos ya los mejores hiperparametros (se alcanzaron al ajustar el modelo con el param_grid de antes)
grid_search = GridSearchCV(cv_model, param_grid = param_grid
, scoring='neg_mean_absolute_error', cv=5)

# Ajustar el modelo con búsqueda en cuadrícula
# grid_search.fit(X_train, Y_train) 


# COMENTARIO:  Alto tiempo de computacion. Ajuste de 2916 modelos. Los mejores hiperparametros se
# han comentado a continuación.

# Obtener los mejores hiperparámetros
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params) 
# Best Hyperparameters: {'transformedtargetregressor__regressor__colsample_bytree': 0.8, 'transformedtargetregressor__regressor__gamma': 0, 
# 'transformedtargetregressor__regressor__learning_rate': 0.1, 'transformedtargetregressor__regressor__max_depth': 4, 
# 'transformedtargetregressor__regressor__min_child_weight': 1, 'transformedtargetregressor__regressor__n_estimators': 200, 
# 'transformedtargetregressor__regressor__objective': 'reg:squarederror', 'transformedtargetregressor__regressor__reg_alpha': 0.1, 
# 'transformedtargetregressor__regressor__reg_lambda': 0, 'transformedtargetregressor__regressor__subsample': 1.0}

# Obtener el mejor modelo
best_model = grid_search.best_estimator_


## PREDICCIONES

    # Predicciones sobre la muestra de entrenamiento

# Realizar predicciones en el conjunto de entrenamiento
y_pred = best_model.predict(X_train)
mae = median_absolute_error(Y_train, y_pred)
r2 = r2_score(Y_train, y_pred)

# Mostrar el mejor modelo y resultados
print("Best Model:", best_model)
print(f"MAE on training set: {mae:.2f} euros")
print(f"R2 on training set: {r2:.2f}")

    # Predicciones sobre la muestra de test

# Realizar predicciones en el conjunto de prueba
y_pred = best_model.predict(X_test)
mae = median_absolute_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

# Mostrar el mejor modelo y resultados
print("Best Model:", best_model)
print(f"MAE on test set: {mae:.2f} euros")
print(f"R2 on test set: {r2:.2f}")

# Representacion de los resultados
fig, ax = plt.subplots(figsize=(5, 5))
plt.scatter(Y_test, y_pred)
ax.plot(Y_test, Y_test, linestyle="--", color="red")
plt.text(500000, 200000, f"MAE on test set: {mae: .2f} euros")
plt.title("XGBoost Model with GridSearchCV")
plt.xlabel("Test Data")
plt.ylabel("Predict Data")
plt.show()


# COMENTARIO: Este modelo con los hiperparametros seleccionados empleando la validacion cruzada
# ha obtenido para la muestra de entrenamiento un valor del R^2 de 0.84 y un MAE de 
# 17700€ aproximadamente. El valor de la proporcion de la varianza explicada se ha incrementado
# considerablemente. Por otro lado, también ha aumentado el MAE, lo que puede indicar que hemos
# reducido el sobreajuste del modelo sobre los datos del entrenamiento. Esto son muy buenas
# noticias.

# COMENTARIO: Por otro lado, se ha obtenido para la muestra de test un valor del R^2 de 0.71 y
# un MAE de 23800€ aproximadamente. 




"""
MODELO 3. AJUSTE DEL MODELO FINAL
          PIPELINES
          MEJORES HIPERPARAMETROS
"""

## SELECCION DE LAS VARIABLES X e Y
x, y = df_pisos.drop(["precio"], axis = 1), df_pisos["precio"]
x.info()

## DIVISION EN MUESTRA DE ENTRENAMIENTO Y MUESTRA DE TEST
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2, random_state=99)
X_train.shape, X_test.shape

# Convertir variables categoricas en nuevas variables con 1 y 0
variables_categoricas = ["ubicacion_nueva", "Planta", "Conservacion", "Calefaccion"]
variables_numericas = ["Superficie", "Habitaciones", "Baños", "Gastos de Comunidad",
                       "Amueblado", "Balcon", 'Ascensor', 'Calefaccion', 'Garaje', 
                       'Terraza', 'Trastero']
# Preprocesador
preprocesador = make_column_transformer(
    (OneHotEncoder(drop = "if_binary"), variables_categoricas), # Crea variables binarias y elimina las antiguas
    remainder = "passthrough", # Indica que no se deben cambiar las variables que no sean categoricas
    verbose_feature_names_out=False # Evita que se generen nombres de caracteristicas adicionales
)

# Creamos el modelo
model = make_pipeline(
    preprocesador,
    TransformedTargetRegressor(
        regressor = XGBRegressor(colsample_bytree = 0.8,
                                 gamma = 0,
                                 learning_rate = 0.1,
                                 max_depth = 4,
                                 min_child_weight = 1,
                                 n_estimators = 200,
                                 objective = "reg:squarederror",
                                 reg_alpha = 0.1,
                                 reg_lambda = 0,
                                 subsample = 1.0), func = np.log10, inverse_func=sp.special.exp10
    ),
)


# Ajustamos el modelo
model.fit(X_train, Y_train)

# Predicciones

    # Predicciones sobre la muestra de entrenamiento
y_pred_train = model.predict(X_train)
r2 = r2_score(Y_train, y_pred_train)
mae_train = median_absolute_error(Y_train, y_pred_train)
print(f"MAE on training set: {mae_train: .2f} euros") # 17746.28
print(f"R2 on training set: {r2: .2f}") # 0.84

    # Predicciones sobre la muestra de test
y_pred_test = model.predict(X_test)
r2 = r2_score(Y_test, y_pred_test)
mae_test = median_absolute_error(Y_test, y_pred_test)
print(f"MAE on training set: {mae_test: .2f} euros") # 23806.49
print(f"R2 on training set: {r2: .2f}") # 0.71

# Representacion de los resultados

    # Resultados representando la muestra de entrenamiento y sus predicciones
fig, ax = plt.subplots(figsize = (5,5))
plt.scatter(Y_train, y_pred_train)
ax.plot(Y_train, Y_train, linestyle="--", color="red")
plt.text(500000, 200000, f"MAE on training set: {mae_train: .2f} euros")
plt.title("XGBoost Model | Training set - Predictions")
plt.xlabel("Training Data")
plt.ylabel("Predict Data")
plt.show()

    # Resultados representando la muestra de test y sus predicciones
fig, ax = plt.subplots(figsize = (5,5))
plt.scatter(Y_test, y_pred_test)
ax.plot(Y_test, Y_test, linestyle="--", color="red")
plt.text(500000, 200000, f"MAE on testing set: {mae_test: .2f} euros")
plt.title("XGBoost Model | Testing set - Predictions")
plt.xlabel("Test Data")
plt.ylabel("Predict Data")
plt.show()


# COMENTARIO: Observando el grafico que representa los datos de test junto con sus predicciones, 
# se puede ver que a partir de un precio del piso de 200000€ las predicciones comienzan a predecir por 
# debajo del valor real. Entiendo que poner precio a un piso también tiene una gran parte de 
# subjetividad. Ademas, haber alcanzado un error de unicamente 23000€ en la muestra de test,
# indica que estamos ante un buen modelo.

# COMENTARIO: A continuacion, vamos representar los datos de test junto con sus predicciones,
# señalando la ubicacion de cada uno de los pisos.

# Crear una paleta de colores única para cada ubicación
colores_ubicaciones = sns.color_palette("husl", n_colors=len(df_pisos['ubicacion_nueva'].unique()))

# Crear un diccionario que asocie cada ubicación con su respectivo color
colores_dict = dict(zip(df_pisos['ubicacion_nueva'].unique(), colores_ubicaciones))

# Asignar colores a cada punto en función de la ubicación
colores_test = df_pisos.loc[Y_test.index, 'ubicacion_nueva'].map(colores_dict)

# Crear el gráfico de dispersión con colores distintos para cada ubicación
fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(Y_test, y_pred_test, c=colores_test)

# Añadir línea de referencia y texto
ax.plot(Y_test, Y_test, linestyle="--", color="red")
ax.text(500000, 200000, f"MAE on testing set: {mae_test: .2f} euros")

# Configurar leyenda con etiquetas de ubicación y colores correspondientes
legend_labels = [Line2D([0], [0], marker='o', color='w', label=ubicacion, 
                        markerfacecolor=colores_dict[ubicacion], markersize=10) 
                 for ubicacion in df_pisos['ubicacion_nueva'].unique()]

ax.legend(handles=legend_labels, title='Ubicación', loc='upper left', bbox_to_anchor=(1, 1))

ax.set_title("XGBoost Model | Testing set - Predictions")
ax.set_xlabel("Test Data")
ax.set_ylabel("Predict Data")

plt.show()



# CONCLUSIÓN FINAL:

# Hay que reconocer que la predicción de un precio de un piso cuando se generalizan tanto los datos,
# como por ejemplo, al nivel de una provincia, es una tarea complicada.

# En el gráfico anterior, se puede observar, que el modelo predice con un error grande, en su mayoría,
# aquellos pisos que pertenecen a grandes ciudades como Vigo o Pontevedra, o que pertenecen a espacios
# que abarcan distintos pueblos como los pisos pertenecientes a la categoría "OtrasUbicaciones".

# Lo idóneo, sería crear un modelo para cada una de estas grandes ciudades, haciendo gran hincapié al
# barrio en el que se encuentra ubicado el piso. Dentro de Vigo, por ejemplo, existen pisos con 
# las mismas características que presentan precios muy distantes en función de donde se encuentren
# ubicados dentro de la ciudad. Y yo considero que eso es lo que mas hace errar al modelo de machone
# learning.

# A pesar de todo eso, haber alcanzado un MAE de aproximadamente 23000€ me parece un éxito. Incluso,
# estaría interesante estudiar los pisos que el modelo de machine learning predice por debajo de su
# precio real como oportunidades de inversión inmobiliaria.



