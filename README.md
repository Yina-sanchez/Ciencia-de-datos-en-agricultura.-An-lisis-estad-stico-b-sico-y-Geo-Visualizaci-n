# Ciencia de datos en agricultura. Análisis estadístico básico y Geo Visualización
Este proyecto está dedicado a la descarga, pre-preparación y análisis estadístico de cuentas económicas para la agricultura y a la creación de mapas interactivos que muestren la dinámica de los precios.

## Introductión
El principal problema a resolver en este proyecto es la descarga, análisis estadístico y visualización de un DataSet.

La dificultad básica del análisis estadístico de datos reales es que se preparan o presentan en una forma que no es conveniente para los métodos mecánicos de análisis estadístico. Por lo tanto, esta práctica de laboratorio muestra métodos de pre-preparación automática de datos reales para tales casos. El siguiente problema es la capacidad de manipular y transformar de manera competente los grandes datos para obtener un informe estadístico conveniente tanto en forma tabular como en forma de gráficos.
Por lo tanto, el objetivo principal que debemos lograr en este laboratorio es aprender a descargar, preprocesar y realizar un análisis estadístico básico de las cuentas económicas para la agricultura y presentarlo en mapas interactivos.
## Materiales y métodos 
En este proyecto, se presenta cómo descargar datos, realizar análisis estadísticos básicos, visualizarlos y mostrarlos en mapas dinámicos.
Todos los datos se obtienen de Eurostat DataBase (https://ec.europa.eu/eurostat/data/database). Este laboratorio consta de los siguientes pasos:
* Descarga de datos: descargue y muestre datos de un archivo
* Preparación de datos: análisis preliminar de la estructura de datos, cambio de tipos de datos y estructura de tablas
* Análisis estadístico - análisis estadístico básico
* Visualización de datos: salida de varios datos en gráficos
* Construya una línea de tendencia: el método básico para construir una línea de tendencia y un pronóstico basado en ella
* Mapas interactivos: construcción de mapas interactivos que muestran el cambio de datos durante un período de tiempo.

Los datos estadísticos se obtuvieron de https://ec.europa.eu/eurostat/databrowser/view/aact_eaa01/default/table?lang=en. Eurostat tiene una política de fomento de la libre reutilización de sus datos, tanto para fines comerciales como no comerciales.

'import pandas as pd <
import matplotlib.pyplot as plt
import numpy as np
import openpyxl 
import seaborn as sns

%matplotlib inline
plt.rcParams["figure.figsize"] = (8, 6)

import warnings
warnings.filterwarnings('ignore')'
