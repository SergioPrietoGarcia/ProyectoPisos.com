import pandas as pd
import numpy as np
from ast import literal_eval # convertir un string que almacena una lista de un csv en una lista
import re

# Cargamos los datos
archivo = "C:\\Users\\Lenovo\\Desktop\\Universidad\\Python\\Proyecto Inmobiliaria\\Proyecto Pisos.com\\Web Scraping\\pisos.com-pontevedra.csv"
df_pisos = pd.read_csv(archivo, encoding = "utf-16", sep = ";",  # necesario introducir encoding y el separador
                       converters = {'caracteristicas_basicas':literal_eval,
                                     'caracteristicas_extra':literal_eval}) # Esto es necesario para transformar el string que almacena ambas listas
                                                                            # en una lista.

# Aplanar la lista anidada de 'caracteristicas_extra' para crear una unica lista por fila en esta variable
df_pisos['caracteristicas_extra'] = df_pisos['caracteristicas_extra'].apply(lambda x: sum(x, []))

# Visualizamos los datos
df_pisos.head()
df_pisos.info()
df_pisos = df_pisos[df_pisos["precio"] != "A consultar"] # Eliminar filas con precio igual a "A consultar"
df_pisos["precio"] = df_pisos["precio"].astype(int)
df_pisos.shape # (1517, 5)

# Investigamos primera fila
df_pisos["titulo"][0]; df_pisos["precio"][0]; df_pisos["ubicacion"][0]; df_pisos["caracteristicas_basicas"][0]; df_pisos["caracteristicas_extra"][0]

# Numero total de ubicaciones (126)
df_pisos.ubicacion.unique()
df_pisos.ubicacion.nunique()

df_pisos.reset_index(inplace = True)
df_pisos = df_pisos.drop("index", axis = 1)

"""
El objetivo principal de este script es pre-procesar los datos para adaptarlos
a la realización de un análisis exploratorio y la creación de uno o varios
modelos de machine learning.

Se puede observar a continuación que las variables "caracteristicas_basicas"
y "caracteristicas_extra" vienen en formato lista. Dentro de estas listas
encontramos las caracteristicas de cada piso. El objetivo es desglosar cada
una de estas dos variables para formar otras variables que recojan cada una de
las caracteristicas del piso correspondiente.
"""
df_pisos.caracteristicas_basicas[34]
df_pisos.caracteristicas_extra[34]

"""
La primera funcion que vamos a definir devuelve un True o un False
en función de si encuentra un texto clave dentro de un string.
"""

def match_property(property, patterns):
    """
    patterns debe introducirse en formato lista. Para asi tener
    la opcion de introducir mas de una palabra
    """
    for pat in patterns:
        match_prop = re.search(pat, property)
        if match_prop:
            return True
        return False

match_property("con trastero", ["con"])

"""
La segunda funcion que vamos a definir devuelve un 1 o un 0
en función de si encuentra un texto clave dentro de un string.
"""

def check_property(property, patterns):
    """
    patterns debe introducirse en formato lista. Para asi tener
    la opcion de introducir mas de una palabra
    """
    for pat in patterns:
        check = re.search(pat, property)
        if check:
            return 1
        return 0

check_property("con trastero", ["con"])

"""
La tercera funcion que vamos a definir se le pasa un string
y su objetivo es extraer el número que esta dentro de ese string.
"""

def get_number(property):
    nums = re.findall(r'\d+', property)
    if len(nums) == 2:
        return(int(nums[0] + nums[1]))
    else:
        return int(nums[0])
    
re.findall(r'\d+', '3 baños') # Devuelve: ['3']
re.findall(r'\d+', '60.000') # ['60', '000'] por eso el condicionante if
get_number('3 baños')


"""
A continuacion, debemos crear una funcion para cada tipo de variable que deseemos extraer.
Por lo tanto, vamos a visualizar las caracteristicas para encontrar las que mas se repiten
y obviar algunas como la "Referencia".
"""

for i in df_pisos["caracteristicas_basicas"][0:200]:
    print(i)

# Variables a tener en cuenta de "caracteristicas_basicas": "Superficie (construida)", 
# "Habitaciones", "Baños", "Planta"(observar categorias), "Conservacion" (en principio "A estrenar", "En buen estado", "A reformar", "Reformado"),
# "Gastos de comunidad"
    
for i in df_pisos["caracteristicas_extra"][0:200]:
    print(i)

# Variables a tener en cuenta de "caracteristicas_extra": "Amueblado (con armario empotrado o amueblado)", "Trastero", "Ascensor", "Terraza", "Calefacción(tipo)",
# "Balcón" "Cocina amueblada-equipada (si o no)", "Garaje" 


    # Funciones para extraer las caracteristicas de las caracteristicas básicas

# Metros cuadrados construidos
def get_metros_construidos(features):
    for prop in features:
        if match_property(prop.lower().strip(), ["superficie construida"]):
            try:
                metros = get_number(prop.lower().strip().split(",")[0])
            except:
                metros = prop
            return(metros)

x = ['Superficie construida: 151 m²', 'Habitaciones: 4', 'Baños: 2', 'Planta: 4ª', 'Referencia: SA5110-PSV47']
get_metros_construidos(x)


# Numero de habitaciones
def get_habitaciones(features):
    for prop in features:
        if match_property(prop.lower().strip(), ["habitaci"]):
            try:
                habitaciones = int(get_number(prop.lower().strip()))
            except:
                habitaciones = None
            return habitaciones
        
x = ['Superficie construida: 151 m²', 'Habitaciones: 4', 'Baños: 2', 'Planta: 4ª', 'Referencia: SA5110-PSV47']
get_habitaciones(x)


# Numero de baños
def get_baños(features):
    for prop in features:
        if match_property(prop.lower().strip(), ["baño"]):
            try:
                baños = get_number(prop.lower().strip())
            except:
                baños = None
            return baños
        
x = ['Superficie construida: 151 m²', 'Habitaciones: 4', 'Baños: 2', 'Planta: 4ª', 'Referencia: SA5110-PSV47']
type(get_baños(x))


# Planta o tipo de piso(exterior, interior, bajo)
def get_planta(features):
    for prop in features:
        if match_property(prop.lower().strip(), ["planta", "interior", "exterior", "bajo"]):
            return(prop)

        
x = ['Superficie construida: 151 m²', 'Habitaciones: 4', 'Baños: 2', 'Planta: 4ª', 'Referencia: SA5110-PSV47']
get_planta(x)


# Estado de conservacion
def get_conservacion(features):
    for prop in features:
        if match_property(prop.lower().strip(), ["conserv"]):
            estado = prop.split(":")[1].strip()
            return(estado)
    return("En buen estado")
        
x = ['Superficie construida: 100 m²', 'Superficie útil: 91 m²', 'Habitaciones: 3', 'Baños: 1', 'Planta: 1ª', 'Antigüedad: Menos de 5 años', 'Conservación: A estrenar', 'Referencia: 4527440-000005']
x = ['Superficie construida: 100 m²', 'Superficie útil: 91 m²', 'Habitaciones: 3', 'Baños: 1', 'Planta: 1ª']
get_conservacion(x)




# Gasto comunidad
def get_gasto_comunidad(features):
    for prop in features:
        if match_property(prop.lower().strip(), ["gastos de comunidad:"]):
            gasto = prop.split(":")[1].strip()
            return(gasto)
    return 0

        
x = ['Superficie construida: 80 m²', 'Superficie útil: 75 m²', 'Habitaciones: 2', 'Baños: 2', 'Planta: 2ª', 'Antigüedad: Entre 5 y 10 años', 'Conservación: En buen estado', 'Referencia: 4527402-000002', 'Gastos de comunidad: 60€']
get_gasto_comunidad(x)


    # Funciones para extraer las caracteristicas de las caracteristicas extra

# Presencia de ascensor
def get_ascensor(features):
    for prop in features:
        if match_property(prop.lower().strip(), ["ascensor"]):
            return(check_property(prop.lower().strip(), ["ascensor"]))
    return 0

            

x = ['Armarios empotrados: 1', 'Tipo suelo: Tarima maciza', 'Carpintería exterior: Aluminio', 'Calefacción: Gas natural', 'Portero automático', 'Orientación: Este', 'Se aceptan mascotas', 'Clasificación: ', 'Ascensor']
get_ascensor(x)


# Presencia de terraza
def get_terraza(features):
    for prop in features:
        if match_property(prop.lower().strip(), ["terraza"]):
            return(check_property(prop.lower().strip(), ["terraza"]))
    return 0
            

x = ['Armarios empotrados: 1', 'Tipo suelo: Tarima maciza', 'Carpintería exterior: Aluminio', 'Calefacción: Gas natural', 'Portero automático', 'Ascensor', 'Orientación: Este', 'Se aceptan mascotas', 'Clasificación: ']
get_terraza(x)

# Presencia de Balcón
def get_balcon(features):
    for prop in features:
        if match_property(prop.lower().strip(), ["balcón"]):
            return(check_property(prop.lower().strip(), ["balcón"]))
    return 0
            

x = ['Armarios empotrados: 1', 'Tipo suelo: Tarima maciza', 'Carpintería exterior: Aluminio', 'Calefacción: Gas natural', 'Portero automático', 'Ascensor', 'Orientación: Este', 'Se aceptan mascotas', 'Clasificación: ', 'Terraza', 'Balcon']
get_balcon(x)

# Amueblado
def get_amueblado(features):
    for prop in features:
        if match_property(prop.lower().strip(), ["empotrado"]) | match_property(prop.lower().strip(), ["amueblado"]):
            return 1
    return 0

x = ['Carpintería interior: Roble', 'Tipo suelo: Parquet', 'Calefacción: Si, sin especificar', 'Cocina: Cocina amueblada. ', 'Garaje: Acensor hasta el garaje', 'Trastero: 6 metros cuadrados', 'Balcón: Tiene 1 balcon(es)', 'Clasificación: ']
get_amueblado(x)

# Presencia de Trastero
def get_trastero(features):
    for prop in features:
        if match_property(prop.lower().strip(), ["trastero"]):
            return(check_property(prop.lower().strip(), ["trastero"]))
    return 0
            

x = ['Armarios empotrados: 1', 'Trastero', 'Tipo suelo: Tarima maciza', 'Carpintería exterior: Aluminio', 'Calefacción: Gas natural', 'Portero automático', 'Ascensor', 'Orientación: Este', 'Se aceptan mascotas', 'Clasificación: ', 'Terraza', 'Balcon']
get_trastero(x)

# Calefaccion tipo
def get_calefaccion(features):
    for prop in features:
        if match_property(prop.lower().strip(), ["calefacción:"]):
            calefaccion = prop.split(":")[1].strip()
            return(calefaccion)
    return('No tiene')
    
x = ['Trastero','Calefacción: Gas Natural' ,'Armarios empotrados: 1', 'Tipo suelo: Tarima maciza', 'Carpintería exterior: Aluminio', 'Portero automático', 'Ascensor', 'Orientación: Este', 'Se aceptan mascotas', 'Clasificación: ', 'Terraza', 'Balcon']
get_calefaccion(x)

# Presencia de Garaje
def get_garaje(features):
    for prop in features:
        if match_property(prop.lower().strip(), ["garaje"]):
            return(check_property(prop.lower().strip(), ["garaje"]))
    return 0
        
x = ['Garaje', 'Trastero', 'Armarios empotrados: 1', 'Tipo suelo: Tarima maciza', 'Carpintería exterior: Aluminio', 'Portero automático', 'Ascensor', 'Orientación: Este', 'Se aceptan mascotas', 'Clasificación: ', 'Terraza', 'Balcon']
get_garaje(x)





"""
Una vez creadas todas las funciones se aplicará cada una de ellas con el objetivo de crear tantas
variables como funciones haya, introduciendolas en el dataframe df_pisos
"""

# Caracteristicas basicas
df_pisos["Superficie"] = df_pisos['caracteristicas_basicas'].apply(get_metros_construidos)
df_pisos["Habitaciones"] = df_pisos['caracteristicas_basicas'].apply(get_habitaciones)
df_pisos["Baños"] = df_pisos['caracteristicas_basicas'].apply(get_baños)
df_pisos["Planta"] = df_pisos['caracteristicas_basicas'].apply(get_planta)
df_pisos["Conservacion"] = df_pisos['caracteristicas_basicas'].apply(get_conservacion)
df_pisos["Gastos de Comunidad"] = df_pisos['caracteristicas_basicas'].apply(get_gasto_comunidad)

# Características Extra
df_pisos["Amueblado"] = df_pisos['caracteristicas_extra'].apply(get_amueblado)
df_pisos["Balcon"] = df_pisos['caracteristicas_extra'].apply(get_balcon)
df_pisos["Ascensor"] = df_pisos['caracteristicas_extra'].apply(get_ascensor)
df_pisos["Calefaccion"] = df_pisos['caracteristicas_extra'].apply(get_calefaccion)
df_pisos["Garaje"] = df_pisos['caracteristicas_extra'].apply(get_garaje)
df_pisos["Terraza"] = df_pisos['caracteristicas_extra'].apply(get_terraza)
df_pisos["Trastero"] = df_pisos['caracteristicas_extra'].apply(get_trastero)


"""
Una vez creada la nueva base de datos procedemos a eliminar las variables que almacenan listas
y comprobar que todo se encuentra correcto
"""

df_pisos = df_pisos.drop(["caracteristicas_basicas", "caracteristicas_extra"], axis = 1)

df_pisos.shape # 16 variables
df_pisos.head()
df_pisos.describe()
df_pisos.info() # La variable que presenta mas valores nulos es "Planta"

df_pisos["Planta"].unique()
df_pisos["Planta"].value_counts()
df_pisos["Calefaccion"].unique()
df_pisos["Conservacion"].value_counts()

df_pisos.to_csv('df_pisos_pontevedra.csv', header = True)

# df_pisos = pd.read_csv('df_pisos_pontevedra.csv')
# df_pisos = df_pisos.drop('Unnamed: 0', axis = 1)


"""
Despues de haber realizado un primer pre-procesado, en el análisis exploratorio de datos,
ademas de realizar la exploración, lo más seguro es que se continuen depurando y 
perfeccionando los datos.
"""
