import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

import joblib

pd.set_option('display.max_rows', None) # Ver los dataframes completos

# Cargamos los datos
df_pisos = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Universidad\\Python\\Proyecto Inmobiliaria\\Proyecto Pisos.com\\Analisis Exploratorio de Datos\\pisos-pontevedra-machine-learning.csv')
df_pisos.head()
df_pisos.columns

"""
En primer lugar, para que un modelo de machine learning funcione correctamente, lo mejor,
es transformar las variables categóricas en dummies, es decir, en valores numéricos.

Estas variables serían la "ubicacion", la "Planta", la "Conservacion" y la "Calefaccion"
"""

# Transformacion de las variables categoricas
df_pisos_num = pd.get_dummies(df_pisos, columns = ["ubicacion","Planta", "Conservacion", "Calefaccion"], dummy_na=True)
#df_pisos_num = pd.get_dummies(df_pisos, columns = ["Planta", "Conservacion", "Calefaccion"], dummy_na=True)

df_pisos.info() # Hay NAs, vamos a eliminarlos
df_pisos = df_pisos.dropna()
df_pisos_num.head()
# COMENTARIO: Sin conocer el motivo se han creado columnas para los valores perdidos, que en principio son todo 0
df_pisos_num.Calefaccion_nan.sum()
df_pisos_num.Calefaccion_nan.sum()
df_pisos_num.Conservacion_nan.sum()
df_pisos_num.Planta_nan.sum()

df_pisos_num = df_pisos_num.drop(["Calefaccion_nan", "Calefaccion_nan", "Conservacion_nan", "Planta_nan"], axis = 1)
df_pisos_num.columns

# Eliminamos las filas que presentan NAs
df_pisos_num = df_pisos_num.dropna()
df_pisos_num.isna().sum()

# COMENTARIO: Las dummies estan compuestas por False y True en lugar de 0 y 1.
# Esto se cambiara mas adelante realizando una transformacion a tipo entero

"""
RIDGE REGRESSION
"""

# Creacion de objeto "x" (variables dependientes) y objeto "y" (variables independientes)
x, y = df_pisos_num.drop(["titulo", "precio"], axis = 1), df_pisos_num["precio"]
#x, y = df_pisos_num.drop(["titulo", "precio", "ubicacion"], axis = 1), df_pisos_num["precio"]

# Transformamos el objeto "x" en entero
x = x.astype(int)
x.info(); x.head()

# Almacenamos el conjunto de variables en una lista
vars_pisos = list(df_pisos_num.columns)
vars_pisos.remove("precio")
vars_pisos.remove("titulo")
#vars_pisos.remove("ubicacion")


    # MODELO SIMPLE DE RIDGE REGRESION POR VALIDACION CRUZADA

# COMENTARIO: Ajustando este modelo por validacion cruzada se lleva a cabo una division
# de los datos en 10 secciones y a continuacion se emplea como muestra de entrenamiento
# 9 de ellas y la restante como test. Este proceso se realiza para cada una de las secciones,
# tantas veces como secciones haya, en este caso 10

lr_m = Ridge()

from sklearn import model_selection

n_folds = 10
kf = KFold(n_folds, shuffle=True, random_state=99) # Argumento shuffle es para mezclar los datos antes de dividirlos

scores = cross_val_score(lr_m, x, y, scoring="neg_mean_absolute_error", cv = kf, n_jobs = -1) # USO DE TODOS LOS NUCLEOS n_jobs = -1
# COMENTARIO: el neg_mean_absolute_error se utiliza porque scikit-learn espera funciones 
# de puntuación que sean maximizadas, y al tomar el negativo del MAE, se convierte en una 
# función de maximización que puede ser utilizada en el contexto de la validación cruzada.

print("mae_mean: %.3f\t\tmae_std: %.3f" % (-scores.mean(), scores.std()))
print("scores: \n", -np.round(scores,3))
# COMENTARIO: El error medio de prediccion es de 49711.194€ y su desviación típica de 2594.346€



    # FINE TUNING. RIDGE REGRESSION POR VALIDACIÓN CRUZADA

lr_m = Ridge()

# Definimos el grid de alphas
l_alpha = [2**k for k in range(-6,10)]
param_grid = {'alpha':l_alpha}

# Ajustamos el modelo
ridge_alpha_search = GridSearchCV(lr_m,
                                  param_grid = param_grid,
                                  cv = kf,
                                  scoring="neg_mean_absolute_error", 
                                  n_jobs = -1,
                                  verbose = 1) # para imprimir por pantalla la informacion

ridge_alpha_search.fit(x,y)

# Imprimimos el grid, el mejor alpha y el mejor mae resultante del modelo
print("alpha range: %.2f - %.2f" % (np.array(l_alpha).min(), np.array(l_alpha).max()))
print("best alpha = %.2f" % (ridge_alpha_search.best_params_['alpha']))
print("best_cv_mae = %.2f" % (-ridge_alpha_search.best_score_))
# COMENTARIO: El mejor alpha recibe un valor de 2. El mejor MAE de 49704.33

# Representacion de los alphas y el mae
plt.xticks(range(len(l_alpha)), l_alpha, rotation = 45) # Label eje X
plt.plot(-ridge_alpha_search.cv_results_['mean_test_score'])
plt.show()
# COMENTARIO: Vamos a repetir el proceso pero esta vez para los alpha cercanos
# al mejor, a ver si encontramos algun alpha mejor que 4.


# Segunda iteracion: recentramos el alpha y agudizamos la busqueda
l_alpha = [2. * 2.**(k/2.) for k in range(-5,4)]
param_grid = {'alpha':l_alpha}

# Ajustamos el modelo
ridge_alpha_search = GridSearchCV(lr_m,
                                  param_grid = param_grid,
                                  cv = kf,
                                  scoring="neg_mean_absolute_error", 
                                  n_jobs = -1,
                                  verbose = 1) # para imprimir por pantalla la informacion

ridge_alpha_search.fit(x,y)

# Imprimimos el grid, el mejor alpha y el mejor mae resultante del modelo
print("alpha range: %.2f - %.2f" % (np.array(l_alpha).min(), np.array(l_alpha).max()))
print("best alpha = %.2f" % (ridge_alpha_search.best_params_['alpha']))
print("best_cv_mae = %.2f" % (-ridge_alpha_search.best_score_))
# COMENTARIO: El mejor alpha recibe un valor de 1.41. El mejor MAE de 49695.84

# Representacion de los alphas y el mae
plt.xticks(range(len(l_alpha)), np.round(l_alpha,2), rotation = 45) # Label eje X
plt.plot(-ridge_alpha_search.cv_results_['mean_test_score'])
plt.show()
# COMENTARIO: Hemos mejorado un poco el MAE y el mejor alpha ha cambiado a 5.66


"""
Una vez calculado el mejor alpha, vamos a proceder a ajustar el modelo con este parametro.
Una vez hecho esto, realizaremos la predicciones y calcularemos las correspondientes metricas
"""

best_alpha = ridge_alpha_search.best_params_['alpha']
lr_m = Ridge(alpha = best_alpha)
scores = cross_val_score(lr_m, x, y, scoring="neg_mean_absolute_error", cv = kf, n_jobs = -1) 
print("mae_mean: %.3f\t\tmae_std: %.3f" % (-scores.mean(), scores.std()))
# COMENTARIO: Obtenemos un MAE de 49695.836 y una desviacion tipica de 2577.822

# Realizamos las predicciones
y_pred = cross_val_predict(lr_m, x, y)

# Representamos graficamente
plt.figure(figsize = (8,6))
plt.title("Real vs CV Predicted Values")
plt.xlabel("Precio")
plt.ylabel("Predicciones_precio")
plt.plot(y, y_pred, ".", y, y, "-")
plt.show()

# COMENTARIO: Se observa claramente que el modelo predice con mas error los datos reales
# cuando el precio del piso supera aproximadamente los 400k euros. 


# Coeficientes del modelo
lr_m.fit(x,y)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
coeficientes_modelo = pd.DataFrame(lr_m.coef_, vars_pisos, columns = ["coef"]).sort_values(by = "coef", ascending=False)
coeficientes_modelo
# COMENTARIO: Investigar a ajustar el modelo quitando como variable dependiente la ubicacion. Hay ubicaciones que suman
# demasiado valor a los pisos, como es obvio, pero tengo curiosidad de observar que ocurre al retirarla del modelo

# COMENTARIO: Quitar la ubicacion de los datos significa perder demasiada informacion y obtener un MAE mas elevado






"""
Hasta ahora hemos ajustado un modelo Ridge Regression por validación cruzada.
A continuación, lo que haremos será dividir los datos en una muestra de entrenamiento
y una de test y observar los resultados
"""
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
x_train.shape; y_train.shape
x_test.shape; y_test.shape

# Ajuste del modelo Ridge Regression 
lr_m = Ridge()
lr_m.fit(x_train, y_train)

# Realizamos las predicciones
predictions = lr_m.predict(x_test)
y_test[0:10];predictions[0:10]

# Representamos graficamente
plt.figure(figsize = (8,6))
plt.title("Real vs CV Predicted Values")
plt.xlabel("Precio")
plt.ylabel("Predicciones_precio")
plt.plot(y_test, predictions, ".", y_test, y_test, "-")
plt.show()

# Evaluacion del modelo
RidgeRegression_MAE = np.mean(np.absolute(predictions - y_test))
RidgeRegression_MSE = np.mean(predictions - y_test)** 2
from sklearn.metrics import r2_score
RidgeRegression_R2 = r2_score(y_test, predictions)

RidgeRegression_MAE.round(5)
RidgeRegression_MSE.round(5)
RidgeRegression_R2.round(5)

# COMENTARIO: Observamos que el MAE recibe un valor de 48927.047. El R^2 seria de 0.61 aproximadamente


"""
COMENTARIOS FINALES. Observan los resultados se ha alcanzado un MAE minimo de 48927.05 y un R^2 de 0.61
aproximadamente. El objetivo es mejorar el modelo. Las dificultades se encuentran a la hora de tratar de
predecir los pisos más caros. Para solucionar este error, se creara otro modelo mediante la aplicacion
de correción logaritmica sobre la variable "precio" y el uso de pipelines. En el script RRmodel-pipelines.py
se encuentra el codigo
"""
