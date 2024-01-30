import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

df_pisos = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Universidad\\Python\\Proyecto Inmobiliaria\\Proyecto Pisos.com\\Pre-Procesado\\df_pisos_pontevedra.csv')
df_pisos = df_pisos.drop('Unnamed: 0', axis = 1)
df_pisos.head()
df_pisos.describe()
df_pisos.columns
df_pisos.info()

# Variable Gastos de Comunidad
df_pisos["Gastos de Comunidad"].unique()


""" 
Como se puede observar en el tipo de variable, la variable "Gastos de Comunidad" no es tipo entero
si no que es un string. Esto es debido a que cuando se extrajeron los datos del web scraping
esa característica podía venir con un número en concreto, por ejemplo 30€ de gastos, o, podia venir
recogida en una horquilla de valores, por ejemplo, entre 30 y 60€.

Lo que vamos a hacer a continuacion antes de continuar con el analisis exploratorio es transformar
esa variable a tipo entero, y si aparece una horquilla de valores pues calcularemos la media.

Para ello, vamos a reciclar las funciones que creamos en el pre-procesado de datos
"""
# Funciones usadas en el Pre-procesado de datos
def match_property(property, patterns):
    for pat in patterns:
        match_prop = re.search(pat, property)
        if match_prop:
            return True
        return False

def check_property(property, patterns):
    for pat in patterns:
        check = re.search(pat, property)
        if check:
            return 1
        return 0

def get_number(property):
    nums = re.findall(r'\d+', property)
    if len(nums) == 2:
        return(int(nums[0] + nums[1]))
    else:
        return int(nums[0])
    
# Funcion para extraer el gasto de la comunidad
def gasto_comunidad(carac):
    # Buscar patrón "entre" en el string
    if match_property(carac.lower().strip(), ["entre"]):
        # Extraer todos los números presentes en el string
        lista = re.findall(r'\d+', carac)

        # Verificar que la lista tiene al menos dos elementos
        if len(lista) >= 2:
            # Calcular el promedio de los dos números
            precio = (int(lista[0]) + int(lista[1])) / len(lista)
            return int(precio)
    else:
        # Extraer el primer número encontrado en el string
        lista = re.findall(r'\d+', carac)
        if lista:
            return int(lista[0])

    # Si no se encuentra ningún número, retornar 0
    return 0

# Aplicamos la funcion a los datos de esa variable y sustituimos
df_pisos['Gastos de Comunidad'] = df_pisos["Gastos de Comunidad"].apply(gasto_comunidad)
df_pisos.info()
df_pisos.describe()



    ## PAIRPLOTS. Unicamente variables numericas.
vars1 = ["Superficie", "Habitaciones", "Baños", "Gastos de Comunidad"] + ["precio"]
vars2 = ["Amueblado", "Balcon", "Ascensor", "Garaje", "Terraza", "Trastero"] + ["precio"]


pair_plot = sns.pairplot(df_pisos[vars1])
plt.show()

# COMENTARIO: Existe una aparente relación lineal entre el precio de la vivienda y
# la superficie, el numero de baños y los gastos de la comunidad. En cuanto al numero
# de habitaciones parece que cuando se incrementa a 5 el numero de habitaciones el precio
# disminuye sustancialmente

pair_plot = sns.pairplot(df_pisos[vars2])
plt.show()

# COMENTARIO: Parece haber pequeñas diferencias en el precio en funcion de los distintos
# atributos de las variables


        ## ANÁLISIS DE LAS VARIABLES

"""
VARIABLE PRECIO
"""

    ### Variable "precio"
sns.histplot(data = df_pisos.precio)
plt.show()

df_pisos.loc[df_pisos.precio > 1000000][["precio", "Superficie"]]

# COMENTARIO: ¿Que hacemos con esas casas? Son unicamente 2 pisos de 764
# que superan el millón de euros. Tal vez la mejor idea es tratarlos como outliers
# y eliminarlas.
df_pisos = df_pisos[df_pisos.precio < 1000000]


"""
VARIABLE "HABITACIONES"
"""

    ### Variable "Habitaciones"
df_pisos.Habitaciones.value_counts()
sns.histplot(data = df_pisos.Habitaciones)
plt.show()

# COMENTARIO: Predominan los pisos con 3, 2 y 4 habitaciones, en ese orden

"""
VARIABLE "BAÑOS"
"""

    ### Variable "BAÑOS"
df_pisos["Baños"].value_counts().sum()
df_pisos.loc[df_pisos["Baños"].isna()][["titulo","Baños", "precio", "ubicacion", "Planta"]]

# COMENTARIO: Hay 4 pisos que reflejan un NaN en su valor de baños. Comprobando individualmente
# en la web se puede ver que algo debio pasar en el scraping por lo que los relleno manualmente
df_pisos = df_pisos.drop(3) # Se elimino el anuncio
df_pisos.loc[643, "Baños"] = 2.0
df_pisos.loc[308, "Baños"] = 1.0
df_pisos.loc[111, "Baños"] = 1.0


"""
VARIABLE "GASTOS DE COMUNIDAD"
"""

    ### Variable "Gastos de Comunidad"
sns.histplot(data = df_pisos["Gastos de Comunidad"])
plt.show()
# COMENTARIO: La mayoría de los gastos a la comunidad se encuentran entre 0 y 50 euros

sns.scatterplot(x = df_pisos["Gastos de Comunidad"], y = df_pisos.precio)
plt.show()
# COMENTARIO: Puede percibirse un pequeño incremento en el precio del piso a medida
# que se incrementan los gastos de comunidad
 

"""
VARIABLE "SUPERFICIE"
"""

    ### Variable "Superficie"
sns.histplot(data = df_pisos["Superficie"])
plt.show()


"""
VARIABLE "PLANTA"
"""

    ### Variable "Planta"
df_pisos.Planta.value_counts()
df_pisos.Planta.isnull().sum() 
df_pisos.Planta.unique()
# COMENTARIO: Hay 171 valores nulos. ¿Que podemos hacer con ello? Vamos a revisar las webs
# y observar si en la gran mayoría, que no aparezca la planta puede significar que el domicilio
# se encuentre en una planta alta o baja.
df_pisos.loc[df_pisos["Planta"].isnull()][["titulo", "ubicacion", "precio", "Planta"]]

# COMENTARIO: Existen bastantes casos en los que no pone nada de información sobre la planta
# y solo queda la opcion de guiarse por las imagenes del vendedor. 

# Division de la variable en: "Primeros_pisos"(1:3), "Ultimos_pisos"(4:7), "Muchos_pisos"(>8), "Desconocido"
def process_piso(serie):
    if pd.notna(serie):
        serie = str(serie)
        lista = serie.split()
        mapeo = ""
        if len(lista[1]) > 3:
            mapeo = "Bajo"
        elif int(lista[1].replace("ª", "")) < 3:
            mapeo = "Primeros_pisos"
        elif int(lista[1].replace("ª", "")) < 3 & int(lista[1].replace("ª", "")) > 8:
            mapeo = "Ultimos_pisos"
        else:
            mapeo = "Muchos_pisos"
        return(mapeo)
    else:
        return("Desconocido")

df_pisos.Planta = df_pisos.Planta.apply(process_piso)


# Representacion con la variable Precio
sns.catplot(x = df_pisos.Planta, y = df_pisos.precio)
plt.show()

# COMENTARIO: La gran diferencia entre las categorias radica en "Bajo". Como era de esperar
# los bajos son mas baratos que los pisos mas elevados.


"""
VARIABLE "CONSERVACION"
"""
    # Variable "Conservacion"
df_pisos.Conservacion.value_counts()
df_pisos.Conservacion.isnull().sum() # No existen valores nulos

# Representacion de las Conservacion junto al precio
sns.catplot(x = df_pisos.Conservacion, y = df_pisos.precio)
plt.show()

# COMENTARIO: Que el piso se encuentre " A reformar" o "A estrenar", aporta
# informacion valiosa a la variable precio.


"""
VARIABLE "AMUEBLADO"
"""
    ### Variable "Amueblado"
df_pisos.Amueblado.value_counts()
df_pisos.Amueblado.isnull().sum() # No presenta valores nulos

# Representacion de la variable "Amueblado" junto con el precio
sns.catplot(x = df_pisos.Amueblado, y = df_pisos.precio)
plt.show()



"""
VARIABLE "BALCON"
"""
    ### Variable "Balcon"
df_pisos.Balcon.value_counts()
df_pisos.Balcon.isnull().sum() # No presenta valores nulos

# Representacion de la variable "Balcon" junto con el precio
sns.catplot(x = df_pisos.Balcon, y = df_pisos.precio)
plt.show()


"""
VARIABLE "ASCENSOR"
"""
    ### Variable "Ascensor"
df_pisos.Ascensor.value_counts()
df_pisos.Ascensor.isnull().sum() # No presenta valores nulos

# Representacion de la variable "Ascensor" junto con el precio
sns.catplot(x = df_pisos.Ascensor, y = df_pisos.precio)
plt.show()



"""
VARIABLE "CALEFACCION"
"""
    ### Variable "Calefaccion"
df_pisos.Calefaccion = df_pisos.Calefaccion.replace("No tiene", "No especifica")
df_pisos.Calefaccion.value_counts()
df_pisos.Calefaccion.isnull().sum() # No presenta valores nulos
# COMENTARIO: Se observan demasiadas categorias de calefaccion, pero algunas se repiten.
# Por otro lado, algunas se marcan como colectivas, otras como individuales y otras no se marcan.
# Vamos a proceder a crear las categorias: "Gas", "Electrica", "No especifica", "Colectiva", "Renovable"

def process_calefaccion(serie):
    serie = str(serie)
    mapeo = ""
    if match_property(serie.lower().strip(), ["gas"]) or match_property(serie.lower().strip(), ["propan"]) or match_property(serie.lower().strip(), ["calefacción individual"]):
        mapeo = "Gas"
    if match_property(serie.lower().strip(), ["trica"]) or match_property(serie.lower().strip(), ["radiante"]):
        mapeo = "Electrica"
    if serie == "No especifica":
        mapeo = serie
    if match_property(serie.lower().strip(), ["central"]) or match_property(serie.lower().strip(), ["colect"]):
        mapeo = "Colectiva"
    if serie == "Individual" or serie == "Si, sin especificar":
        mapeo = "No especifica"
    if serie == "Aerotermia" or serie == "Calor azul" or serie == "Energía renovable" or serie == "Placas solares":
        mapeo = "Renovable"
    return(mapeo)

df_pisos.Calefaccion = df_pisos.Calefaccion.apply(process_calefaccion)
df_pisos.Calefaccion.value_counts()


# Representacion de la variable "Calefaccion" junto con el precio
sns.catplot(x = df_pisos.Calefaccion, y = df_pisos.precio)
plt.show()



"""
VARIABLE "GARAJE"
"""
    ### Variable "Garaje"
df_pisos.Garaje.value_counts()
df_pisos.Garaje.isnull().sum() # No presenta valores nulos

# Representacion de la variable "Garaje" junto con el precio
sns.catplot(x = df_pisos.Garaje, y = df_pisos.precio)
plt.show()


"""
VARIABLE "TERRAZA"
"""
    ### Variable "Terraza"
df_pisos.Terraza.value_counts()
df_pisos.Terraza.isnull().sum() # No presenta valores nulos

# Representacion de la variable "Terraza" junto con el precio
sns.catplot(x = df_pisos.Terraza, y = df_pisos.precio)
plt.show()



"""
VARIABLE "TRASTERO"
"""
    ### Variable "Trastero"
df_pisos.Trastero.value_counts()
df_pisos.Trastero.isnull().sum() # No presenta valores nulos

# Representacion de la variable "Trastero" junto con el precio
sns.catplot(x = df_pisos.Trastero, y = df_pisos.precio)
plt.show()


"""
VARIABLE "UBICACION"
"""

# Reseteamos los indices
df_pisos.reset_index(inplace = True)
df_pisos = df_pisos.drop("index", axis = 1)

    # Variable "Ubicacion"
df_pisos.ubicacion.value_counts() # 95 ubicaciones
df_pisos.ubicacion.unique()

"""
Explicación: Se puede observar que existen numerosas ubicaciones para os pisos.
Considero que son demasiadas e incluso la información puede verse afectada cuando
existen ciertas ubicaciones que pertenecen al mismo municipio. 

Para concentrar más la información de esta variable, lo que haré será clasificar cada una de las ubicaciones
en bruto, dentro de uno de los 61 municipios que tiene la provincia de Pontevedra. Y eso es lo que hago a continuación.

Destacar que he resumido cada uno de los 61 municipios con una palabra clave, mas que nada
para facilitar el desarrollo del codigo y llegar a un buen resultado. Por ejemplo, el municipio
de "A Illa de Arousa" lo he clasificado como "Arousa". "Vilagarcía de Arousa" como "Vilagarcía",
y así con las que me hizo falta abreviar.

Esto mas que nada es por la logica que he seguido al crear la función a que aparece después. Si por ejemplo
no sintetizo "Pazos de Borbén" en una unica clave como es "Pazos", si la ubicación real presentase
el string "de" pero perteneciese por ejemplo a "Municipio de Cangas", la función me clasificaría
el municipio como "Pazos", siendo este "Cangas".

"""

# Todas las ubicaciones de los pisos (monarcas y municipios)
municipios_pontevedra = [
    "Cañiza", "Estrada", "Guarda", "Arousa", "Lama", "Agolada", "Arbo", "Neves", 
    "Baiona", "Barro", "Bueu", "Caldas", "Cambados", "Lameiro", "Cangas", "Catoira", 
    "Cerdedo", "Cerdedo", "Cotobade", "Covelo", "Crecente", "Cuntis", "Dozón", "Forcarei", 
    "Fornelos", "Gondomar", "Lalín", "Marín", "Meaño", "Meis", "Moaña", "Mondariz", 
    "MondarizBalneario", "Moraña", "Mos", "Nigrán", "Grove", "Porriño", "Rosal", "Oia", 
    "Pazos", "Poio", "Caldelas", "Ponteareas", "Pontecesures", "Pontevedra", "Portas", 
    "Redondela", "Ribadumia", "Rodeiro", "Salceda", "Salvaterra", "Sanxenxo", 
    "Silleda", "Soutomaior", "Tomiño", "Tui", "Valga", "Vigo", "Cruces", "Vilaboa", 
    "Vilagarcía", "Vilanova"
]

# Funcion para clasificar las ubicaciones en cada uno de estos nombres
def process_ubicacion(serie):
    lista = serie.strip().replace("(", "").replace(")", "").split()
    for elemento in lista:
        for municipio in municipios_pontevedra:
            if elemento == municipio:
                return(elemento)
    return("".join(lista))

x = df_pisos.ubicacion.apply(process_ubicacion)

# Con el siguiente codigo vamos a comprobar cuantas de las ubicaciones
# se han conseguido clasificar correctamente. Para ello, usaremos la funcion
# creada match_property() que devuelve un True si el elemento que deseamos buscar
# aparece en el string buscado.
aciertos = 0
errores = 0
for i in range(0,len(x)):
    if match_property(df_pisos.ubicacion[i], x[i]):
        aciertos +=1
    else:
        errores +=1
print(f"El numero de elementos que se han clasificado correctamente es de: {aciertos}. \nEl numero de errores ha sido de {errores}")    

# Introducimos los nuevos datos en la variable "ubicacion", sustituyendo a los antiguos valores
df_pisos.ubicacion = x

# Exploramos la nueva variable de "ubicacion"
df_pisos.ubicacion.value_counts()

# Representacion de la Ubicacion junto al precio
sns.catplot(y = df_pisos.ubicacion, x = df_pisos.precio)
plt.show()
# COMENTARIO: Parece que la variable ubicacion si presenta bastante informacion






## GUARDAMOS LOS DATOS RESULTANTES DEL ANALISIS EXPLORATORIO DE DATOS EN UN CSV

df_pisos.to_csv("pisos-pontevedra-machine-learning.csv", index = False)
