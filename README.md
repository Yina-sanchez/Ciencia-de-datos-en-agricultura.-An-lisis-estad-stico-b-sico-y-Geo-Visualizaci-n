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

````
import pandas as pd <
import matplotlib.pyplot as plt
import numpy as np
import openpyxl 
import seaborn as sns

%matplotlib inline
plt.rcParams["figure.figsize"] = (8, 6)

import warnings
warnings.filterwarnings('ignore')'
````
El siguiente paso es descargar el archivo de datos del repositorio por *[read_csv()]
````
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/data-science-in-agriculture-basic-statistical-analysis-and-geo-visualisation/estat_aact_eaa01_defaultview_en.csv')
df
````
Data preparación
````
df.columns

Index(['DATAFLOW', 'LAST UPDATE', 'freq', 'itm_newa', 'indic_ag', 'unit',
       'geo', 'TIME_PERIOD', 'OBS_VALUE', 'OBS_FLAG'],
      dtype='object')
````
Podemos ver que la columna 'geo' tiene un tipo de objeto. Pero esta columna contiene códigos de países.
Por lo tanto, es necesario cambiar el tipo de estos datos a categóricos.
````
df.loc[:, 'geo'] = df['geo'].astype('category')
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 398 entries, 0 to 397
Data columns (total 3 columns):
 #   Column       Non-Null Count  Dtype   
---  ------       --------------  -----   
 0   geo          398 non-null    category
 1   TIME_PERIOD  398 non-null    int64   
 2   OBS_VALUE    398 non-null    float64 
dtypes: category(1), float64(1), int64(1
````
obtener la lista de países. 
````
df['geo'].unique()
['AT', 'BE', 'BG', 'CH', 'CY', ..., 'RO', 'SE', 'SI', 'SK', 'UK']
Length: 40
Categories (40, object): ['AT', 'BE', 'BG', 'CH', ..., 'SE', 'SI', 'SK', 'UK']
````
Cabe señalar que existen algunos códigos de países no estándar para el Reino Unido y Grecia.
Deberíamos cambiar los valores: UK a GB para el Reino Unido y EL a GR para Grecia.
Para hacer esto, debemos agregar nuevos nombres de categoría
````
df['geo'] = df['geo'].cat.add_categories(["GB", "GR"])
pd.options.mode.chained_assignment = None  # swich of the warnings
mask = df['geo'] == 'UK' # Binary mask
df.loc[mask, 'geo'] = "GB" # Change the values for mask
df

	geo	TIME_PERIOD	OBS_VALUE
0	AT	2011	       906.72
1	AT	2012	      1029.21
2	AT	2013	      717.58
3	AT	2014	      769.41
4	AT	2015	      728.38
````
````
mask = df['geo'] == 'EL'
df.loc[mask, 'geo'] = "GR"
df
````
Después de eso, agregue una nueva columna que contenga los nombres completos de los países. Para hacer esto
Para agregar una columna con los nombres completos de los países, debemos crear una función que obtenga un código de país y devuelva un nombre completo.
````
import pycountry
list_alpha_2 = [i.alpha_2 for i in list(pycountry.countries)]  # create a list of country codes
print("Country codes", list_alpha_2)

def country_flag(df):
    '''
    df: Series
    return: Full name of country or "Invalide code"
    '''
    if (df['geo'] in list_alpha_2):
        return pycountry.countries.get(alpha_2=df['geo']).name
    else:
        print(df['geo'])
        return 'Invalid Code'

df['country_name']=df.apply(country_flag, axis = 1)
df
````


