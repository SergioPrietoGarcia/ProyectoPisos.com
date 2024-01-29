import requests
from bs4 import BeautifulSoup
import random
import time
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc
import os # os.system('cls') : limpiar la consola
import shutil


def procesar_lista(original):
    """
    Procesa una lista combinando elementos consecutivos que contienen ":" y elimina duplicados.

    Parameters:
    - original (list): Lista original a procesar.

    Returns:
    - list: Nueva lista con elementos combinados y sin duplicados.
    """
    nueva_lista = []
    i = 0
    while i < len(original):
        if ":" in original[i]:
            nueva_lista.append(original[i-1] + original[i])
            i += 1
        else:
            nueva_lista.append(original[i])
        i += 1

    # Eliminar duplicados
    i = 0
    while i < len(nueva_lista) - 1:
        if nueva_lista[i] in nueva_lista[i + 1]:
            del nueva_lista[i]
        else:
            i += 1

    return nueva_lista





# Tiempo que tarda en ejecutarse el script
inicio_tiempo = time.time()



## Scraping de las caracteristicas de todos los pisos a la venta en Pontevedra a partir del archivo "ids_pisos.csv" 
ids_pisos = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\Universidad\\Python\\Proyecto Inmobiliaria\\Proyecto Pisos.com\\ids_pisos.csv")

driver = uc.Chrome()
pisos = pd.Series() # Creamos una serie a partir de la cual se creara un dataframe con todos los pisos

def parsear_inmueble(id_inmueble):
    """
    Esta funcion tiene como objetivo extraer las correspondientes caracteristicas
    de la web que almacena la información de un piso en concreto
    """
    print('\n Casa numero: ' + id_inmueble)
    url = 'https://www.pisos.com' + id_inmueble

    driver.get(url)
    time.sleep(random.randint(1,3)) # bajar el tiempo

    """
    Introducimos un try - except por si hubiese boton de aceptar 
    cookies e impedir que se detenga el codigo si el boton no aparece
    """
    try:
        driver.find_element("xpath",'//*[@id="didomi-notice-agree-button"]').click() # Click boton cookies
    except:
        pass

    html = driver.page_source
    soup = BeautifulSoup(html, 'lxml')

    
    titulo = soup.find("div", {'class':"maindata-info"}).find("h1", {'class':"title"}).text.strip() 
    print('\n Titulo: ' + titulo)

    ubicacion = soup.find("div", {'class':"maindata-info"}).find("h2", {'class':"position"}).text.strip()
    print('\n Ubicacion: ' + ubicacion)

    precio = soup.find("div", {'class':"maindata-box"}).find("span", {'class':"h1 jsPrecioH1"}).text.replace(".", "").replace("€", "").strip()
    print('\n Precio: ' + precio)

    # Division de los bloques de html en funcion del sector de caracteristicas: básicas, muebles, equipamiento, exteriores y certificado energetico
    caracteristicas = soup.find("div", {"id":"characteristics"}).find_all("div", {'class':"charblock"}) # Html donde se encuentra el cuadro con el total de caracteristicas y otros elementos

        # Caracteristicas básicas
    caracteristicas_sector0 = caracteristicas[0].find_all("div", {'class':"charblock-right"}) # html donde se encuentra el bloque de caracteristicas sector 0
    basicas_html = caracteristicas_sector0[0].find_all("ul", {'class':"charblock-list charblock-basics"})[0].find_all("span") # html con la lista de caracteristicas basicas
    caract_basicas = procesar_lista([carac_basic.text.strip() for carac_basic in basicas_html]) # Lista con las caracteristicas basicas
    print(caract_basicas)

        # Características extra
    caract_extra = []
    for i in range(1, len(caracteristicas)):
        extra_html = soup.find("div", {"id":"characteristics"}).find_all("div", {'class':"charblock"})[i].find_all("span")
        extra = procesar_lista([caract.text for caract in extra_html])
        caract_extra.append(extra)
    
    print(caract_extra)


    terminal_width = shutil.get_terminal_size().columns
    line = '-' * terminal_width
    print(line) # Imprime una linea que ocupa el ancho de la terminal
    
    
    pisos['titulo'] = titulo
    pisos['ubicacion'] = ubicacion
    pisos['precio'] = precio
    pisos['caracteristicas_basicas'] = caract_basicas
    pisos['caracteristicas_extra'] = caract_extra
    df_pisos = pd.DataFrame(pisos)


    return(df_pisos.T)

df_pisos = parsear_inmueble(ids_pisos.iloc[0].ID) # Primera fila del dataframe

# Inicializamos un bucle con la funcion parsear_inmueble() para extrer toda
# la información que deseamos de cada piso almacenado en "ids_pisos.csv"
for i in range(1, 820): #len(ids_pisos) - 820
    try:
    # Concatena los inmuebles adicionales a df_pisos
        df_temporal = parsear_inmueble(ids_pisos.iloc[i].ID)
        df_pisos = pd.concat([df_pisos, df_temporal])
        print(i)
        time.sleep(random.randint(2, 3)) # bajar el tiempo
    except:
        pass

driver.quit()

# Calcular el tiempo total transcurrido
tiempo_total = time.time() - inicio_tiempo

# Imprimir el tiempo total de ejecución
print(f"Tiempo total de ejecución: {tiempo_total:.2f} segundos")

df_pisos.reset_index(drop=True, inplace = True)
df_pisos
df_pisos.to_csv("pisos.com-pontevedra.csv", index = False, sep = ";", encoding="utf-16")

## PRIMERA RONDA.

    # TIEMPO DE DURACION 8857.96 segundos - 2 horas y media
    # Numero de iteraciones: 820
    # Numero de pisos scrapeados: 772
    # Restantes - iteraciones de 821 a 1639

