# Web Scraping de Pisos en Pontevedra

Esta carpeta contiene archivos dedicados al web scraping de datos en la plataforma pisos.com. El objetivo es recopilar información sobre anuncios de ventas de pisos en la provincia de Pontevedra, España.

## Archivos del Proyecto

1. **`scrap-info-un-piso.py`**: Este archivo se encarga de realizar el web scraping de datos para un solo piso. Extrae información detallada de un anuncio de venta específico en pisos.com.

2. **`scrap-IDs-pontevedra.py`**: En este archivo, se realiza el scraping de todas las IDs de los pisos en Pontevedra. Estas IDs luego se utilizan para construir las URL de los anuncios individuales y obtener más información sobre cada piso.

3. **`scrap-caract-pisos-pontevedra.py`**: En este archivo se lleva a cabo el scraping total de los pisos en la provincia de Pontevedra. Utiliza las IDs obtenidas en el segundo archivo para recopilar información sobre múltiples anuncios de venta de pisos.

## Archivos CSV Resultantes

En la carpeta `resultados` encontrarás dos archivos CSV resultantes de la ejecución de los scripts:

1. **`ids_pisos_pontevedra.csv`**: Contiene las IDs de los pisos recopiladas durante el proceso de web scraping.

2. **`caracteristicas_brutas_pisos_pontevedra.csv`**: Almacena las características en bruto y sin procesar de todos los pisos en la provincia de Pontevedra.

## Ejecución

Para ejecutar cualquiera de estos archivos, asegúrate de tener las bibliotecas necesarias instaladas. Puedes hacerlo ejecutando el siguiente comando:

```bash
pip install -r requirements.txt
