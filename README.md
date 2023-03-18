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
# Data preparación
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
Como puede ver, se agregó la columna con los nombres completos de los países y este conjunto de datos contiene una gran cantidad de datos con un código no válido. Eliminemos estos datos usando una máscara binaria.
````
mask = df['country_name'] != 'Invalid Code'
df = df[mask]
df

	geo	TIME_PERIOD	OBS_VALUE   country_name
0	AT	2011	         906.72	    Austria
1	AT	2012	         1029.2     Austria
2	AT	2013	         717.58	    Austria
````
# Análisis estadístico
````
df.info()
df.describe()
df.describe(include=['category'])
````
Como puede ver, la información estadística consiste en el número de valores únicos, el valor de la categoría más popular y el número de sus valores.
La información detallada para una columna específica se puede obtener de la siguiente manera
````
df['country_name'].value_counts()

Output exceeds the size limit. Open the full output data in a text editor
Austria           10
Belgium           10
Slovakia          10
Slovenia          10
Sweden            10
````
Puede ver que esta información no es adecuada porque los datos no están agrupados. Para obtener estadísticas adecuadas, este conjunto de datos debe transformarse utilizando una tabla dinámica  [pivot_table()]
````
pt_country = pd.pivot_table(df, values= 'OBS_VALUE', index= ['TIME_PERIOD'], columns=['country_name'], aggfunc='sum', margins=True)
pt_country
_____
country_name	Austria	Belgium	Bulgaria ...
TIME_PERIOD																					
2011	      906.72	465.34	1196.06
2012	     1029.21	606.09	1311.49
2013	      717.58	513.17	1258.57
````
Después de eso, podemos calcular la descripción estadística para cada país.
````
pt_country.describe()
````
Podemos obtener estadísticas por años:
````
pt = pd.pivot_table(df, values= 'OBS_VALUE', index= ['country_name'], columns=['TIME_PERIOD'], aggfunc='sum', margins=True)
pt
pt.describe()
````
## Data visualization
Construyamos un gráfico para la última fila ('All') excepto los últimos valores para la columna ('All'). Pandas hereda la función de Matplotlib para el trazado.
````
pt.iloc[-1][:-1].plot()
````
![image](https://user-images.githubusercontent.com/109825689/226137554-5ef889b3-ba0d-473c-b6be-1681bf1237e3.png)
Construyamos un gráfico de barras para los valores de resumen de cada país (la última columna 'All' excepto la última fila).
````
pt['All'][:-1].plot.bar(x='country_name', y='val', rot=90)
````
![image](https://user-images.githubusercontent.com/109825689/226137922-f51ec5c8-9566-4002-8df1-7a4576ba9dd4.png)
Comparemos las cuentas económicas de Alemania y Francia en un gráfico de barras.
````
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(len(pt.columns)-1)  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots() # Create subplots
rects1 = ax.bar(x - width/2, pt.loc['Germany'][:-1], width, label='Germany') # parameters of bars
rects2 = ax.bar(x + width/2, pt.loc['France'][:-1], width, label='France')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('OBS_VALUE')
ax.set_xlabel('Years')
ax.set_xticks(x)
plt.xticks(rotation = 90)
ax.set_xticklabels(pt.columns[:-1])
ax.legend()

fig.tight_layout()

plt.show()
````
![image](https://user-images.githubusercontent.com/109825689/226137998-b5ff2fe2-da6f-43ce-890e-b4a87bf3a8b0.png)
También podemos construir algunas parcelas específicas usando la biblioteca SeaBorn.
````
import seaborn as sns
d = pd.DataFrame(pt.loc['Sweden'][:-1])
print(d)
sns.regplot(x=d.index.astype(int), y="Sweden", data=d,)
___
             Sweden
TIME_PERIOD        
2011         791.98
2012         971.59
2013         757.95
````
![image](https://user-images.githubusercontent.com/109825689/226138071-a8085539-4105-4df4-92b5-5ff374019655.png)
## Línea de tendencia
Hagamos un pronóstico de la dinámica usando una línea de tendencia lineal para Suecia.
Para construir un modelo lineal, es necesario crear el propio modelo lineal, ajustarlo, probarlo y hacer una predicción.
Para hacer esto, use [sklearn.linear_model.LinearRegression()]
````
from sklearn.linear_model import LinearRegression
model = LinearRegression()
X = np.reshape(d.index, (-1, 1)) # transform X values
y = np.reshape(d.values, (-1, 1)) # transform Y values
model.fit(X, y)
````
Cuando el modelo está ajustado, podemos construir nuestro pronóstico. Deberíamos agregar nuevos valores para X y calcular Y.
````
X_pred= np.append(X, [2021, 2022, 2023])
X_pred = np.reshape(X_pred, (-1, 1))
# calculate trend
trend = model.predict(X_pred)

plt.plot(X_pred, trend, "-", X, y, ".")
````
![image](https://user-images.githubusercontent.com/109825689/226138518-5f9c6f4a-7ab3-492c-acee-c430460fdd2e.png)
## Interactive maps
Es conveniente desplegar los cambios de la contabilidad económica en un mapa para visualizarlo. Hay varias bibliotecas para esto. Es conveniente usar la biblioteca. [plotly.express]
````
import plotly.express as px
df
````
````
import json
!wget european-union-countries.geojson "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/data-science-in-agriculture-basic-statistical-analysis-and-geo-visualisation/european-union-countries.geojson"
````
````
with open("european-union-countries.geojson", encoding="utf8") as json_file:
    EU_map = json.load(json_file)
````
El siguiente paso es construir un mapa interactivo usando [plotly.express.choropleth()]
````
fig = px.choropleth(
    df,
    geojson=EU_map,
    locations='country_name',
    featureidkey='properties.name',    
    color= 'OBS_VALUE', 
    scope='europe',
    hover_name= 'country_name',
    hover_data= ['country_name', 'OBS_VALUE'],
    animation_frame= 'TIME_PERIOD', 
    color_continuous_scale=px.colors.diverging.RdYlGn[::-1]
)
````
Entonces deberíamos cambiar algunas características del mapa. Por ejemplo: showcountries, showcoastline, showland y fitbouns en función: [plotly.express.update_geos()], [plotly.express.update_layout]
````
fig.update_geos(showcountries=False, showcoastlines=False, showland=True, fitbounds=False)

fig.update_layout(
    title_text ="Agriculture Economic accounts",
    title_x = 0.5,
    geo= dict(
        showframe= False,
        showcoastlines= False,
        projection_type = 'equirectangular'
    ),
    margin={"r":0,"t":0,"l":0,"b":0}
)
````
![newplot](https://user-images.githubusercontent.com/109825689/226139118-b82dc1f3-075b-45c8-ae34-d46172af8426.png)
## Conclusions
Como se evidencia en la práctica, los datos obtenidos en experimentos de campo reales no son aptos para el procesamiento estadístico directo. Por lo tanto, en este laboratorio aprendimos los métodos básicos de descarga y preparación de datos preliminares.
A diferencia de los enfoques clásicos bien conocidos para el análisis de datos estadísticos, Python contiene muchas bibliotecas poderosas que le permiten manipular datos de manera fácil y rápida. Por lo tanto, hemos aprendido los métodos básicos para automatizar una biblioteca como Pandas para el análisis de datos estadísticos. También aprendimos los métodos básicos para visualizar los datos obtenidos con la biblioteca SeaBorn, que también contiene medios efectivos de análisis visual de datos. Al final del trabajo de laboratorio, mostramos el DataSet en un mapa interactivo dinámico en formato * .html.

 Copyright &copy; 2020 IBM Corporation. This notebook and its source code are released under the terms of the [MIT License](https://cognitiveclass.ai/mit-license/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkGuidedProjectsdatascienceinagriculturebasicstatisticalanalysisandgeovisualisation467-2022-01-01).
