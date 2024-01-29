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


## SCRAPING DE UNA ÚNICA ID: /comprar/piso-sanxenxo_centro_urbano-40082920493_527440/


# Abrir navegador y pulsar en boton de aceptar cookies
id_piso = "/comprar/piso-sanxenxo_centro_urbano-38407695604_100500/"
url = f"https://www.pisos.com{id_piso}/"
driver = uc.Chrome()
driver.get(url) # Apertura del navegador con la url
driver.find_element("xpath",'//*[@id="didomi-notice-agree-button"]').click() # Click boton cookies


# Importar el codigo html de la web y crear objeto soup
html = driver.page_source # html web "desordenado"
soup = BeautifulSoup(html, 'lxml') # html web preparado para poder filtrar

# Extracción de características del piso
titulo = soup.find("div", {'class':"maindata-info"}).find("h1", {'class':"title"}).text.strip()
ubicacion = soup.find("div", {'class':"maindata-info"}).find("h2", {'class':"position"}).text.strip()
precio = soup.find("div", {'class':"maindata-box"}).find("span", {'class':"h1 jsPrecioH1"}).text.replace(".", "").replace("€", "").strip()

# Division de los bloques de html en funcion del sector de caracteristicas: básicas, muebles, equipamiento, exteriores y certificado energetico
caracteristicas = soup.find("div", {"id":"characteristics"}).find_all("div", {'class':"charblock"}) # Html donde se encuentra el cuadro con el total de caracteristicas y otros elementos
len(caracteristicas) # Importante para saber el numero de bloques de caracteristicas

    # Caracteristicas básicas
caracteristicas_sector0 = caracteristicas[0].find_all("div", {'class':"charblock-right"}) # html donde se encuentra el bloque de caracteristicas sector 0
basicas_html = caracteristicas_sector0[0].find_all("ul", {'class':"charblock-list charblock-basics"})[0].find_all("span") # html con la lista de caracteristicas basicas
caract_basicas = procesar_lista([carac_basic.text.strip() for carac_basic in basicas_html]) # Lista con las caracteristicas basicas

    # Características extra
caract_extra = []
for i in range(1, len(caracteristicas)):
    extra_html = soup.find("div", {"id":"characteristics"}).find_all("div", {'class':"charblock"})[i].find_all("span")
    extra = procesar_lista([caract.text for caract in extra_html])
    caract_extra.append(extra)

titulo;ubicacion;precio;caract_basicas;caract_extra

driver.quit()