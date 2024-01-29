# Preprocesamiento de Datos

Este script realiza el preprocesamiento de datos necesario para llevar a cabo un análisis exploratorio y la creación de modelos de machine learning en el contexto del proyecto Pisos.com.

## Objetivo
El objetivo principal es adaptar los datos recopilados de la plataforma pisos.com para su posterior análisis y modelado predictivo.

## Datos
Se utiliza el archivo `pisos.com-pontevedra.csv` que contiene información sobre anuncios de venta de pisos en la provincia de Pontevedra.

## Limpieza de Datos
Se eliminan las filas con precio igual a "A consultar" y se convierte el precio a tipo entero para facilitar su manipulación.

## Extracción de Características
Se definen funciones para extraer características específicas de las variables "caracteristicas_basicas" y "caracteristicas_extra". Estas características incluyen detalles sobre la superficie, habitaciones, baños, planta, conservación, gastos de comunidad, amueblado, balcón, ascensor, calefacción, garaje, terraza y trastero.

## Eliminación de Variables
Una vez creadas las nuevas variables, se eliminan las columnas que almacenaban listas.

## Resultados
El DataFrame resultante se guarda en el archivo `df_pisos_pontevedra.csv` para su posterior análisis y modelado.

Este proceso de preprocesamiento sienta las bases para análisis más avanzados y la creación de modelos predictivos.
