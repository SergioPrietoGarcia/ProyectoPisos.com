# Análisis Exploratorio de Datos

Este repositorio contiene un script de Python que realiza un análisis exploratorio de datos sobre un conjunto de datos de pisos en la provincia de Pontevedra. El análisis se centra en la exploración de variables clave y la visualización de patrones y distribuciones. A continuación, se proporciona un resumen del proceso realizado:

## Contenido del Repositorio:

- **aed-pisos-pontevedra.py:** Este es el script principal que realiza el análisis exploratorio de datos utilizando la biblioteca pandas, seaborn y matplotlib.
- **dashboard-pontevedra.py:** Este script crea un dashboard interactivo en un entorno local en el que se pueden visualizar gran parte de los gráficos creados en el análisis exploratorio. Los datos empleados para crear este dashboard son los obtenidos a raíz del análisis exploratorio, que se pueden localizar en el archivo CSV **pisos-pontevedra-machine-learning.csv**.

## Resumen del Proceso:

### 1. Carga de Datos:

- Se cargaron los datos desde un archivo CSV utilizando la biblioteca pandas.

### 2. Exploración Inicial:

- Se exploraron las primeras filas, la información general y las estadísticas descriptivas del conjunto de datos.

### 3. Preprocesamiento de Variables:

- Se realizaron transformaciones en la variable "Gastos de Comunidad", convirtiéndola a tipo entero y manejando casos de horquillas de valores.

### 4. Visualización de Datos:

- Se realizaron pairplots para explorar relaciones entre variables numéricas.

### 5. Análisis de Variables Clave:

- Se analizaron variables como "precio", "Habitaciones", "Baños", "Gastos de Comunidad", "Superficie", "Planta", entre otras.

### 6. Procesamiento de Variables:

- Se crearon categorías para la variable "Planta" basadas en la información proporcionada.

- Similarmente, se procesó la variable "Calefaccion" creando categorías específicas.

### 7. Representación Gráfica de Variables Categóricas:

- Se realizaron gráficos para visualizar la relación entre variables categóricas y el precio.

### 8. Tratamiento de Outliers:

- Se identificaron y eliminaron outliers en la variable "precio".

### 9. Guardado de Datos:

- Se guardaron los datos resultantes del análisis exploratorio en un nuevo archivo CSV llamado **pisos-pontevedra-machine-learning.csv** .

Este resumen proporciona una visión general del proceso de análisis exploratorio de datos realizado en el script. Para obtener detalles específicos, consulte el script completo en el repositorio.
