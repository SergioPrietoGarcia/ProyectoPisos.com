import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

"""
COMPOSICIÓN DEL DASHBOARD

TÍTULO: VENTA DE PISOS EN LA PROVINCIA DE PONTEVEDRA
DROPDOWN 1: Variable a representar junto con el precio

SECCIÓN 1: Representacion en grafico de barras o histograma de la variable dependiente seleccionada
           Grafico de dispersion de la variable precio junto con la variable seleccionada

SECCIÓN 2: Histograma de la variable precio
           Gráfico de barras con el precio medio por ubicación     
"""

# Carga el DataFrame con los datos de los pisos totalmente depurados "pisos-pontevedra-machine-learning.csv"
df_pisos = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\Universidad\\Python\\Proyecto Inmobiliaria\\Proyecto Pisos.com\\Analisis Exploratorio de Datos\\pisos-pontevedra-machine-learning.csv")

# Crea la app
app = dash.Dash(__name__)

# Configura la supresión de excepciones hasta que se ejecute la llamada de retorno
app.config.suppress_callback_exceptions = True

# Opciones del menú desplegable
dropdown_options = [
    {'label': 'Variable Dependiente', 'value': 'Variable Dependiente'}
]

# Diseño del dashboard
app.title = 'VENTA DE PISOS EN LA PROVINCIA DE PONTEVEDRA'

app.layout = html.Div([
    html.H1('VENTA DE PISOS EN LA PROVINCIA DE PONTEVEDRA', style={'textAlign': 'center', 'color': '#503D36', 'font-size': '20'}),

# Texto a modo de título
    html.Div([  
        html.H3("SECCIÓN 1: Visualización de la variable dependiente | Visualización de la variable dependiente vs variable independiente"),
    ]),
    
    # Menú desplegable
    html.Div([
        html.Label("Selecciona la variable dependiente: ", style={"margin-right": "2em"}),
        dcc.Dropdown(
            id="dropdown-variable",
            options=[
                {'label': 'Superficie', 'value': 'Superficie'},
                {'label': 'Ubicacion', 'value': 'ubicacion'},
                {'label': 'Habitaciones', 'value': 'Habitaciones'},
                {'label': 'Baños', 'value': 'Baños'},
                {'label': 'Planta', 'value': 'Planta'},
                {'label': 'Conservacion', 'value': 'Conservacion'},
                {'label': 'Gastos de Comunidad', 'value': 'Gastos de Comunidad'},
                {'label': 'Amueblado', 'value': 'Amueblado'},
                {'label': 'Balcon', 'value': 'Balcon'},
                {'label': 'Ascensor', 'value': 'Ascensor'},
                {'label': 'Calefaccion', 'value': 'Calefaccion'},
                {'label': 'Garaje', 'value': 'Garaje'},
                {'label': 'Terraza', 'value': 'Terraza'},
                {'label': 'Trastero', 'value': 'Trastero'},
            ],
            searchable=True,
            value="Superficie",
            placeholder='Selecciona una variable...',
            style={'width': '40%', 'fontsize': '30px', 'textAlign': 'left'}
        )
    ]),
    
    # Contenedor de salida
    html.Div([
        html.Div([
            dcc.Graph(id='chart1'),
            dcc.Graph(id='chart2'),
        ], style={'display': 'flex', 'justify-content': 'space-between'}),
        
        html.Hr(),  # Línea horizontal para separar secciones
        
        html.Div([  # Texto a modo de título
            html.H3("SECCIÓN 2: Histograma de la variable precio | Gráfico de barras del precio medio por ubicación"),
        ]),
        
        html.Div([
            dcc.Graph(id='chart3'),
            dcc.Graph(id='chart4'),
        ], style={'display': 'flex', 'justify-content': 'space-between'}),
    ]),
])

# Función callback para actualizar el contenedor de salida basado en las estadísticas seleccionadas
@app.callback(
    Output('chart1', 'figure'),
    Output('chart2', 'figure'),
    Output('chart3', 'figure'),
    Output('chart4', 'figure'),
    [Input('dropdown-variable', 'value')]
)
def update_output_container(input_variable):
    if input_variable:
        df_filtered = df_pisos[["precio", input_variable]]

# PLOT 1.1. Histograma para las variables "Gastos de Comunidad" y "Superficie"
        if input_variable == "Gastos de Comunidad" or input_variable == "Superficie":
            chart1 = px.histogram(df_filtered[input_variable], title=f"Histograma de la variable '{input_variable}'", width = 950, height= 600)
        
        else:
# PLOT 1.2. Grafico de barras para las variables "Gastos de Comunidad" y "Superficie"
            chart1 = px.bar(x=df_filtered[input_variable].value_counts().index, y=df_filtered[input_variable].value_counts().values,
                            title=f"Gráfico de barras de la variable '{input_variable}'", width= 950, height = 600, labels={'x': input_variable, 'y': 'Count'})

# PLOT 2: Grafico de dispersion de la variable dependiente con la variable "precio"
        chart2 = px.scatter(df_filtered, x=input_variable, y="precio", title=f"Grafico de dispersón | '{input_variable}' vs 'precio'", width = 950, height = 600)

# PLOT 3: Hostograma de la variable "precio"
        chart3 = px.histogram(df_pisos.precio, title="Histograma de la variable 'precio'",width=900)

# PLOT 4: Grafico de barras con la media de precio del piso por ubicacion        
        media_ubi = df_pisos.groupby('ubicacion')["precio"].mean().reset_index()
        media_ubi_ordenado = media_ubi.sort_values(by='precio', ascending=True)
        chart4 = px.bar(media_ubi_ordenado, y="ubicacion", x="precio", title="Grafico de barras del precio medio por ubicacion", height=800, width=1000)

        return chart1, chart2, chart3, chart4

# Ejecuta la aplicación Dash
if __name__ == '__main__':
    app.run_server(debug=True)

