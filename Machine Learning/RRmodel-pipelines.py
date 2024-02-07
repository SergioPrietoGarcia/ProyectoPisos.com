import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

import joblib # Guardar el modelo

pd.set_option('display.max_rows', None) # Ver los dataframes completos

# Cargamos los datos
df_pisos = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Universidad\\Python\\Proyecto Inmobiliaria\\Proyecto Pisos.com\\Analisis Exploratorio de Datos\\pisos-pontevedra-machine-learning.csv')

# Visualizamos
df_pisos.head()
df_pisos.columns
df_pisos.info()

# Eliminamos valores perdidos
df_pisos = df_pisos.dropna()
df_pisos.reset_index(inplace=True)
df_pisos = df_pisos.drop("index", axis = 1)
df_pisos.head()

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

def process_ubicacion(ubicaciones):
    if ubicaciones not in ciudades_muchos_pisos:
        return ubicaciones
    else:
        return("OtrasUbicaciones")
    
df_pisos["ubicacion_nueva"] = df_pisos['ubicacion'].apply(lambda x: 'OtrasUbicaciones' if x not in ciudades_muchos_pisos else x)


# Variable precio
sns.pairplot(df_pisos[["precio"]])
plt.show()
# COMENTARIO: Se puede observar que la distribucion de los precios no es homogenea, cosa obvia.
# Que existan precios desplazados de la media de los datos puede darnos problemas a la hora de crear
# un modelo. Por lo que vamos a aplicar una transformacion logaritmica para corregirlo. Esto va a ayudar
# al futuro modelo a comprender mejor los datos

sns.pairplot(np.log10(df_pisos[["precio"]]))
plt.show()


"""
VARIABLES EXPLICATIVAS X | VARIABLE INDEPENDIENTE Y
"""
x, y = df_pisos.drop(["precio", "titulo", "ubicacion"], axis = 1), df_pisos["precio"]
x.info()
# Muestra de entrenamiento y muestra de test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=99, test_size=0.2)

# Convertir variables categoricas en nuevas variables con 1 y 0
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

variables_categoricas = ["ubicacion_nueva", "Planta", "Conservacion", "Calefaccion"]
variables_numericas = ["Superficie", "Habitaciones", "Baños", "Gastos de Comunidad",
                       "Amueblado", "Balcon", 'Ascensor', 'Calefaccion', 'Garaje', 
                       'Terraza', 'Trastero']

preprocesador = make_column_transformer(
    (OneHotEncoder(drop = "if_binary"), variables_categoricas), # Crea variables binarias y elimina las antiguas
    remainder = "passthrough", # Indica que no se deben cambiar las variables que no sean categoricas
    verbose_feature_names_out=False # Evita que se generen nombres de caracteristicas adicionales
)

"""
PIPELINES

un pipeline es una secuencia de procesos de datos encadenados, donde cada paso se realiza de manera ordenada. 
Los pipelines son especialmente útiles para organizar y estructurar el flujo de trabajo de preprocesamiento y
modelado, facilitando la reproducción y la gestión del código.
"""

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
import scipy as sp

# Creamos el modelo
model = make_pipeline(
    preprocesador,
    TransformedTargetRegressor(
        regressor = Ridge(alpha = 1e-10), func = np.log10, inverse_func=sp.special.exp10
    ),
)

# Ajustamos el modelo
model.fit(X_train, Y_train)

# Predicciones
from sklearn.metrics import median_absolute_error

y_pred = model.predict(X_train)
mae = median_absolute_error(Y_train, y_pred)
print(f"MAE on training set: {mae: .2f} euros")

y_pred = model.predict(X_test)
mae = median_absolute_error(Y_test, y_pred)
print(f"MAE on training set: {mae: .2f} euros")

# COMENTARIO: Con un alpha con valor 1e-10, realizando una transformación logaritmica sobre la variable
# precio y empleando los pipelines se ha obtenido:

    # MAE para la muestra de entrenamiento: 28668.19€
    # MAE para la muestra de test: 28305.86€

# Se ha reducido considerablemente el MAE en aproximadamente unos 20000€. Una reducción impresionante.
# A continuacion, vamos a ir cambiando el valor del alpha y comprobar si conseguimos reducir algo mas 
# este error

# Reajuste de los modelos en base a un grid de alphas
l_alpha = [2**k for k in range(-6,10)]

resultados = []

# Para cada valor de alpha, ajusta el modelo y evalúa su rendimiento
for alpha in l_alpha:
    model = make_pipeline(
        preprocesador,
        TransformedTargetRegressor(
            regressor=Ridge(alpha=alpha), func=np.log10, inverse_func=sp.special.exp10
        ),
    )
    # Ajuste del modelo
    model.fit(X_train, Y_train)
    
    # Calcula el MAE en el conjunto de prueba
    y_pred_test = model.predict(X_test)
    mae_test = median_absolute_error(Y_test, y_pred_test)
    
    print(f"Alpha: {alpha}, MAE on testing set: {mae_test:.2f} euros")
    
    # Almacenamos el resultado en la lista
    resultados.append({"Alpha": alpha, "MAE":mae_test})

print(pd.DataFrame(resultados))

# COMENTARIO: El mejor resultado se ha obtenido para alpha = 0.25 con un MAE
# de 28145.77 aproximadamente. Esto se aproxima mucho al resultado con el alpha = 1e-10,
# de hecho, obtenemos un MAE algo mas pequeño, por lo que nos quedaremos con este alpha



    # Representacion de las predicciones para el modelo con alpha = 0.25
# Creamos el modelo
model = make_pipeline(
    preprocesador,
    TransformedTargetRegressor(
        regressor = Ridge(alpha = 0.25), func = np.log10, inverse_func=sp.special.exp10
    ),
)

# Ajustamos el modelo
model.fit(X_train, Y_train)

# Predicciones
from sklearn.metrics import median_absolute_error

y_pred = model.predict(X_train)
mae = median_absolute_error(Y_train, y_pred)
print(f"MAE on training set: {mae: .2f} euros")

y_pred = model.predict(X_test)
mae = median_absolute_error(Y_test, y_pred)
print(f"MAE on training set: {mae: .2f} euros")


# Representacion de los resultados
fig, ax = plt.subplots(figsize = (5,5))
plt.scatter(Y_test, y_pred)
ax.plot([0,1], [0,1], transform = ax.transAxes, ls = "--", c = "red")
plt.text(500000, 200000, f"MAE on training set: {mae: .2f} euros")
plt.title("Ridge Regression Modelo | Regularization")
plt.xlabel("Test Data")
plt.ylabel("Predict Data")
plt.show()

"""
PRIMERAS CONCLUSIONES:

El mejor modelo hasta el momento se ha obtenido ajustando la regresion Ridge para
un valor de alpha de 0.25. Se ha obtenido un MAE para la muestra de test de 28145.77€.

De momento, este es nuestro mejor modelo.

A continuación, vamos a estudiar la aportación de cada variable. Tal vez existan variables 
correlacionadas que dificulten la comprensión de los datos al modelo de machine learning.
"""



# Aportación de cada variable para explicar la variable "precio"
feature_names = model[:-1].get_feature_names_out() # Nombre de las variables
coeficientes = model[-1].regressor_.coef_ # Valor de los coeficientes de cada variable

coefs = pd.DataFrame(coeficientes, columns = ["Coefficients"], index = feature_names)
coefs

# Representacion de los coeficientes
coefs.plot.barh(figsize = (9,7))
plt.title("Coefficients Model")
plt.axvline(x = 0, color = ".5")
plt.xlabel("Raw coefficient values")
plt.subplots_adjust(left = 0.3)
plt.show()

# COMENTARIO: Estos coeficientes no se encuentran representados en la misma escala,
# por lo tanto, no obtenemos una verdadera representacion de la importancia de las variables.
# Esto es debido principalmente a que las variables no tienen, ni mucho menos, varianzas similares

"""
Gracias ChatGpt por ayudarme a resolver como calcular la varianza de las variables
después de una extensa conversacion.

Es necesario convertir el dataframe a formato denso para poder calcular la std.
"""

X_train_preprocessed = pd.DataFrame.sparse.from_spmatrix(model[:-1].transform(X_train))
X_train_preprocessed.columns = feature_names

# Convertir todo el DataFrame a formato denso
X_train_densified = X_train_preprocessed.sparse.to_dense()

# Calcular la desviación estándar para cada columna
desviacion_estandar_por_columna = X_train_densified.std(axis=0)

"""
En el código anterior, cuando convertimos el DataFrame disperso 
X_train_preprocessed a un formato denso, estamos simplemente 
llenando todas las celdas, incluso las que contienen valores nulos, 
para que el DataFrame resultante tenga una representación densa. 
Esto puede ocupar más memoria, pero a veces es necesario si deseas 
realizar ciertas operaciones que no son compatibles con matrices 
dispersas. En este caso, se utilizó para calcular la desviación 
estándar, ya que algunos métodos estadísticos pueden no ser compatibles 
con matrices dispersas.
"""


desviacion_estandar_por_columna.plot.barh(figsize=(9,7))
plt.title("Feature range")
plt.xlabel("Std of feature values")
plt.subplots_adjust(left = 0.5)
plt.show()

# COMENTARIO: Se puede observar que la varianza de los Gastos de Comunidad y
# Superficie tienen mucha mas varianza en comparacion con el resto de variables. 
# Esto supone serias dificultades a la hora de calcular la importancia de las variables
# Por lo que representaremos los coeficientes calculados antes multiplicados por sus
# varianzas

coefs = pd.DataFrame(
    model[-1].regressor_.coef_ * desviacion_estandar_por_columna,
    columns = ["Coefficient importance"], index = feature_names
)

coefs.plot.barh(figsize = (9,7))
plt.xlabel("Coefficient values corrected by the featured's std")
plt.title("Ridge model | Regularization")
plt.axvline(x = 0, color = ".5")
plt.subplots_adjust(left = 0.5)
plt.show()

# COMENTARIO: Lo que mas impactante me parece es que el coeficiente de la variable
# "habitaciones" sea negativo. Tambien "balcon" y "amueblado". En cuanto a las ubicaciones,
# se puede ver que las que tienen coeficientes positivos mas elevados son Vigo, Sanxenxo,
# y Pontevedra; ciudades grandes (centros urbanos) y turísticas.



"""
AJUSTE DEL MODELO POR VALIDACIÓN CRUZADA

A continuacion, emplearemos la validacion cruzada para ajustar modelos con diferentes
datos de entrenamiento y de test
"""

from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold

cv = RepeatedKFold(n_splits = 5, n_repeats=5, random_state = 99)
cv_model = cross_validate(
    model,
    x,
    y,
    cv = cv, 
    return_estimator=True,
    n_jobs = -1
)

# coefs = pd.DataFrame(
#     [
#         est[-1].regressor_.coef_ * est[:-1].transform(x.iloc[train_idx]).std(axis = 0)
#         for est, (train_idx, _) in zip(cv_model["estimator"], cv.split(x,y))
#     ]
# )

"""
Queria aplicar el codigo que se encuentra comentado pero me da un error a la hora
de calcular la desviacion tipica. He buscado la siguiente opcion para calcularlo:
"""

# Inicializar una lista para almacenar los coeficientes ajustados
coefs_list = []

for est, (train_idx, _) in zip(cv_model["estimator"], cv.split(x, y)):
    # Extraer el modelo de regresión lineal del estimador
    regressor = est[-1].regressor_

    # Obtener las transformaciones realizadas por el modelo en el conjunto de entrenamiento
    X_train_transformed = est[:-1].transform(x.iloc[train_idx])

    # Calcular la desviación estándar por columna para matrices dispersas
    std_devs = np.sqrt(np.array(X_train_transformed.power(2).mean(axis=0) - np.power(X_train_transformed.mean(axis=0), 2)))

    # Obtener los coeficientes del modelo ajustado y multiplicar por las desviaciones estándar
    coefficients = regressor.coef_ * std_devs

    # Agregar los coeficientes a la lista
    coefs_list.append(coefficients)

# Asegurarse de que coefs_list tenga la forma adecuada
coefs_list = np.squeeze(coefs_list)
coefs = pd.DataFrame(coefs_list, columns=feature_names)
coefs

# Representacion de las importancia de las variables y su variabilidad
plt.figure(figsize = (9,7))
sns.stripplot(data = coefs, orient = "h", alpha = 0.5)
sns.boxplot(data = coefs, orient = "h", color = "cyan", saturation=0.5, whis = 10)
plt.axvline(x = 0, color = ".5")
plt.xlabel("Coefficient importance")
plt.title("Coefficient importance and its variability")
plt.suptitle("Ridge model | Regularization")
plt.subplots_adjust(left = 0.3)
plt.show() 


"""
Como se puede observar en el grafico la variabilidad de la importancia de las variables
no es muy grande para el conjunto de las variables. La variable que mas variabilidad presenta
en cuanto a su importancia es "Habitaciones", "Baños" y "Metros reales". 

Atendiendo al significado de esas variables, puede deducirse que se encuentran correlacionadas
entre si. Cuantos mas baños y habitaciones es obligatorio que haya mas metros cuadrados.

Esta correlacion puede suponer que el modelo se "lie". Que pasará si eliminamos las variables
"Baños" y "Habitaciones" del modelo?
"""

column_to_drop = ["Habitaciones", "Baños"]

cv_model = cross_validate(
    model,
    x.drop(columns = column_to_drop),
    y,
    cv = cv, 
    return_estimator=True,
    n_jobs = -1
)

# Inicializar una lista para almacenar los coeficientes ajustados
coefs_list = []

for est, (train_idx, _) in zip(cv_model["estimator"], cv.split(x, y)):
    # Extraer el modelo de regresión lineal del estimador
    regressor = est[-1].regressor_

    # Obtener las transformaciones realizadas por el modelo en el conjunto de entrenamiento
    X_train_transformed = est[:-1].transform(x.drop(columns = column_to_drop).iloc[train_idx])

    # Calcular la desviación estándar por columna para matrices dispersas
    std_devs = np.sqrt(np.array(X_train_transformed.power(2).mean(axis=0) - np.power(X_train_transformed.mean(axis=0), 2)))

    # Obtener los coeficientes del modelo ajustado y multiplicar por las desviaciones estándar
    coefficients = regressor.coef_ * std_devs

    # Agregar los coeficientes a la lista
    coefs_list.append(coefficients)

# Asegurarse de que coefs_list tenga la forma adecuada
coefs_list = np.squeeze(coefs_list)
feature_names.tolist().index("Habitaciones")
feature_names.tolist().index("Baños")
coefs_modificado = pd.DataFrame(coefs_list, columns=np.delete(feature_names, [38,39]))


plt.figure(figsize = (9,7))
sns.stripplot(data = coefs_modificado, orient = "h", alpha = 0.5)
sns.boxplot(data = coefs_modificado, orient = "h", color = "cyan", saturation=0.5, whis = 10)
plt.axvline(x = 0, color = ".5")
plt.xlabel("Coefficient importance")
plt.title("Coefficient importance and its variability")
plt.suptitle("Ridge model | Regularization")
plt.subplots_adjust(left = 0.3)
plt.show() 


# Representacion de ambos graficos juntos

# Crear una figura con dos subgráficos en una fila
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))

# Primer subgráfico para coefs_modificado
sns.stripplot(data=coefs_modificado, orient="h", alpha=0.5, ax=axes[0])
sns.boxplot(data=coefs_modificado, orient="h", color="cyan", saturation=0.5, whis=10, ax=axes[0])
axes[0].axvline(x=0, color=".5")
axes[0].set_xlabel("Coefficient importance")
axes[0].set_title("Coefficient importance and its variability")
axes[0].set_suptitle("Ridge model | Regularization")
axes[0].set_adjust(left=0.3)

# Segundo subgráfico para coefs
sns.stripplot(data=coefs, orient="h", alpha=0.5, ax=axes[1])
sns.boxplot(data=coefs, orient="h", color="cyan", saturation=0.5, whis=10, ax=axes[1])
axes[1].axvline(x=0, color=".5")
axes[1].set_xlabel("Coefficient importance")
axes[1].set_title("Coefficient importance and its variability")
axes[1].set_suptitle("Ridge model | Regularization")
axes[1].set_adjust(left=0.3)

plt.show()

"""
La variabilidad de la importancia de la variable "Superficie"
se reduce considerablemente cuando eliminamos las variables 
"Habitaciones" y "Baños" del modelo. Todo esto teniendo en cuenta
la transformación logaritmica realizada sobre la variable precio.
"""

"""
A continuacion, vamos a reajustar el modelo con RidgeCV, pasandole un grid de alphas,
y realizando la transformación logaritmica sobre la variable precio
"""

from sklearn.linear_model import RidgeCV

alphas = np.logspace(-10,10, 21)
model = make_pipeline(
    preprocesador,
    TransformedTargetRegressor(
        regressor = RidgeCV(alphas = alphas),
        func = np.log10,
        inverse_func=sp.special.exp10,
    ),
)

model.fit(X_train, Y_train)
model[-1].regressor_.alpha_ # EL mejor alpha es un 1

# Predicciones y metricas
y_pred = model.predict(X_train)
mae = median_absolute_error(Y_train, y_pred)
print(f"MAE on training set: {mae: .2f} euros")

y_pred = model.predict(X_test)
mae = median_absolute_error(Y_test, y_pred)
print(f"MAE on training set: {mae: .2f} euros")


# Representacion de los resultados
fig, ax = plt.subplots(figsize = (5,5))
plt.scatter(Y_test, y_pred)
ax.plot([0,1], [0,1], transform = ax.transAxes, ls = "--", c = "red")
plt.text(500000, 200000, f"MAE on training set: {mae: .2f} euros")
plt.title("Ridge Regression Modelo | Regularization")
plt.xlabel("Test Data")
plt.ylabel("Predict Data")
plt.show()



"""
PREDICCION FINAL

Trabajamos con el total de los datos
"""
from sklearn.metrics import r2_score

# Ajustamos el modelo
model.fit(x.drop(columns = ["Habitaciones", "Baños"]), y)

# Predicciones y metricas
y_pred = model.predict(x.drop(columns = ["Habitaciones", "Baños"]))
mae = median_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"MAE: {mae: .2f} euros")
print(f"R^2: {r2: .2f}")

# Representacion de los resultados
fig, ax = plt.subplots(figsize = (5,5))
plt.scatter(y, y_pred)
ax.plot([0,1], [0,1], transform = ax.transAxes, ls = "--", c = "red")
plt.text(500000, 200000, f"MAE on training set: {mae: .2f} euros")
plt.title("Ridge Regression Modelo | Regularization")
plt.xlabel("Test Data")
plt.ylabel("Predict Data")
plt.show()


"""
COMENTARIOS FINALES: El mejor modelo de Ridge Regression logrado ha obtenido
un valor del MAE de 29621.64 euros y un valor del R^2 de 0.58.
"""
