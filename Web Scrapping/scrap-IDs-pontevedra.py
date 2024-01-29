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


# Tiempo que tarda en ejecutarse el script
inicio_tiempo = time.time()

## Scraping de las IDs de todos los Pisos/Apartamentos en la provincia
## de Pontevedra

busqueda = 'pontevedra'
page = 1
ids = []
driver = uc.Chrome()
while True:
    """
    Iniciamos el navegador e introducimos la URL
    """
    url = f"https://www.pisos.com/venta/piso-{busqueda}/{page}/"
    driver.get(url)
    time.sleep(random.randint(8,10))

    """
    Introducimos un try - except por si hubiese boton de aceptar 
    cookies e impedir que se detenga el codigo si el boton no aparece
    """
    try:
        driver.find_element("xpath",'//*[@id="didomi-notice-agree-button"]').click() # Click boton cookies
    except:
        pass

    """
    Importamos el codigo html y creamos el objeto soup
    """
    html = driver.page_source
    soup = BeautifulSoup(html, "lxml")

    """ 
    Ya que la paginacion no va por numero si no que funciona mediante
    dos botones: siguiente y anterior. Se procede para la primera pagina
    con el primer try, y como a partir de la segunda dara error, se saltara
    al segundo try
    """
    try:
        pagina_actual = soup.find("div", {"class":"pagination__content"}).find("div", {"class":"pagination__next single"}).text.strip()
    except:
        pass

    try:
        pagina_actual = soup.find("div", {"class":"pagination__content"}).find("div", {"class":"pagination__next border-l"}).text.strip()
    except:
        pass
    
    """
    En este momento, planteo que si existe el boton o el texto de siguiente,
    se avance con la funcion. Si no existe boton de siguiente se detiene aqui.
    Si existe siguiente pues sumamos 1 al valor de x
    """
    try:
        pagina_actual = soup.find("div", {"class": "pagination__content"})
        siguiente_button = pagina_actual.find("div", {"class": "pagination__next"})
        if siguiente_button:
            html_pisos = soup.find("div", {'class': "grid__wrapper"}).find_all("div")
        else:
            break
    except:
        break

    page = page + 1

    """
    Despues de obtener el html de cada piso, se realiza un nuevo filtro del que se
    obtienen las ids de busqueda del piso
    """
    for piso in html_pisos:
        id_piso = piso.get("data-lnk-href")
        if id_piso is not None:
            ids.append(id_piso)
            time.sleep(random.randint(1,3)*random.random())
            print(id_piso)


# Calcular el tiempo total transcurrido
tiempo_total = time.time() - inicio_tiempo # Tardo 2304.86 segundos - 38 minutos

# Imprimir el tiempo total de ejecución
print(f"Tiempo total de ejecución: {tiempo_total:.2f} segundos")

driver.quit() # Cerramos el navegador

ids_pisos = pd.DataFrame(ids)
ids_pisos.columns = ["ID"]
ids_pisos.to_csv('ids_pisos.csv', index = False)
