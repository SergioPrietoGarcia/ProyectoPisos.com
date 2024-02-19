# MACHINE LEARNING

En esta carpeta se pueden encontrar 3 archivos .py. Cada uno de ellos recoge una serie de modelos creados con la finalidad de tratar de predecir la variable "precio" el función de las principales características de un piso cualquiera.

## Script 1. **'Ridge-Regression.py'**

En este script se crearon dos modelos. El primero de ellos una Ridge Regression con validación cruzada y el segundo de ellos una Ridge Regression con validación cruzada y fine tunning.

  - Modelo 1. Ridge Regression - Validación cruzada
En este caso, se han ajustado 10 modelos diferentes por validación cruzada con los parámetros por defecto logrando un MAE de 49711€ aproximadamente. Es decir, de media, el error a la hora de predecir en cada uno de los 10 modelos creados es de 49711€.

  - Modelo 2. Ridge Regression - Validación cruzada - Fine Tunning
En la búsqueda del mejor hiperparámetro, alpha, para el ajuste del modelo se ha alcanzado un MAE de 48908.85€ y un R^2 de 0.61. Las dificultades se encentran principalmente a la hora de predecir los pisos mas caros. Para solucionar este error, se empleará la aplicación de una corrección logarítmica sobre la variable "precio" y el uso de pipelines.


## Script 2. **'RR-model-pipelines.py'**

En este script, se ha partido de la base de transformar logarítmicamente la variable 'precio'.

  - Modelo 1. Ridge Regression - Pipelines
Con un alpha con valor 1e-10, realizando una transformación logarítmica sobre la variable precio y empleando los pipelines se han obtenido los siguientes resultados después de realizar las correspondientes predicciones. Para la muestra de entrenamiento, el MAE ha tomado un valor de 28668.19€, mientras que para la muestra de test ha tomado un valor de 28305.86€. Por otro lado, el R^2 obtenido para la muestra de test y muestra de entrenamiento ha sido de 0.57.

  - Modelo 2. Ridge Regression - Pipelines - Fine Tunning
El mejor modelo hasta el momento se ha obtenido ajustando la regresión Ridge para un valor de alpha de 0.25. Se ha obtenido un MAE para la muestra de test de 28145.77€ y un valor del R^2 de 0.57.

  - Modelo 3. Ridge Regression - Pipelines - Validación Cruzada
El valor del MAE para estos datos y este modelo es de 28705.08€ y el valor del R^2 es de 0.57. No se ha conseguido mejorar el modelo que ya teniamos con un MAE de aproximadamente 28100 y un valor del R^2 de 0.57.

Esto es lo máximo que he podido alcanzar mediante la Ridge Regression. En el siguiente script llevare a cabo un ajuste mediante XGBoost con el objetivo de mejorar el resultado final y obtener un mejor modelo.


## Script 3. **'model-xgboost.py'**

Este script utiliza XGBoost para mejorar los resultados obtenidos con Ridge Regression. También se ha mantenido la transformación logarítmica de la variable 'precio'.

  - Modelo 1. XGBOOST CON PARÁMETROS POR DEFECTO
El modelo ajustado mediante XGBoost ha logrado un valor del R^2 de aproximadamente 0.69 para la muestra de test, indicando que el 69% de la varianza en la variable respuesta es explicada por el modelo. Para la muestra de entrenamiento, se alcanzó un R^2 de aproximadamente 0.96.

Por otro lado, al estudiar el MAE, se observa un valor de error de aproximadamente 27,000 en la muestra de test. Se logró reducir el error comparado con el modelo de Ridge Regression. Además, las predicciones en la muestra de entrenamiento son bastante acertadas, con un valor del MAE de aproximadamente 6,279, indicando claramente un sobreajuste.

  - Modelo 2. XGBOOST - PIPELINES
Este modelo ha obtenido un valor del R^2 de aproximadamente 0.64 en la muestra de test, indicando que alrededor del 64% de la varianza en la variable respuesta es explicada por el modelo. Sin embargo, se observa un claro sobreajuste a los datos de entrenamiento, ya que el R^2 para esa muestra es de aproximadamente 0.96.

En cuanto al MAE, se ha alcanzado un valor de aproximadamente 29,000 en la muestra de test, mostrando un ligero aumento en el error en comparación con el primer modelo XGBoost sin Pipelines. Además, la predicción en la muestra de entrenamiento sigue siendo bastante acertada, con un valor del MAE de aproximadamente 6,500, lo que confirma la presencia de sobreajuste.

Con el objetivo de corregir este sobreajuste, se va a emplear el uso de pipelines, la validación cruzada y la selección de hiperparámetros.

  - Modelo 3. XGBOOTS - PIPELINES - SELECCIÓN DE HIPERPARÁMETROS
En este modelo ajustado con pipelines, selección de hiperparámetros y validación cruzada, se logró mejorar el R^2 en la muestra de test a aproximadamente 0.71. El MAE en la muestra de test se redujo a 23,806.49€, mientras que en la muestra de entrenamiento fue de 17,746.28€, indicando una mejora en la capacidad de generalización del modelo.


# Conclusión final
Hay que reconocer que la predicción de un precio de un piso cuando se generalizan tanto los datos, como por ejemplo, al nivel de una provincia, es una tarea complicada.

En el gráfico anterior, se puede observar que el modelo predice con un error grande, en su mayoría, aquellos pisos que pertenecen a grandes ciudades como Vigo o Pontevedra, o que pertenecen a espacios que abarcan distintos pueblos como los pisos pertenecientes a la categoría "OtrasUbicaciones".

Lo idóneo sería crear un modelo para cada una de estas grandes ciudades, haciendo gran hincapié al barrio en el que se encuentra ubicado el piso. Dentro de Vigo, por ejemplo, existen pisos con las mismas características que presentan precios muy distantes en función de donde se encuentren ubicados dentro de la ciudad. Y yo considero que eso es lo que más hace errar al modelo de machine learning.

A pesar de todo eso, haber alcanzado un MAE de aproximadamente 23,000€ me parece un éxito. Incluso, estaría interesante estudiar los pisos que el modelo de machine learning predice por debajo de su precio real como oportunidades de inversión inmobiliaria.
