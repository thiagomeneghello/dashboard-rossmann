#BIBLIOTECAS
import pandas as pd
import numpy as np
import inflection
import plotly.express as px
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from PIL import Image
from streamlit_folium import folium_static

st.set_page_config(page_title='Cidades', page_icon='🏪', layout='wide')

#---------------------------------------------------------------------------------------------------------
# DEF FUNÇÕES
#---------------------------------------------------------------------------------------------------------
def rename_columns(dataframe):
    df = dataframe.copy()
    title = lambda x: inflection.titleize(x)
    snakecase = lambda x: inflection.underscore(x)
    spaces = lambda x: x.replace(" ", "")
    cols_old = list(df.columns)
    cols_old = list(map(title, cols_old))
    cols_old = list(map(spaces, cols_old))
    cols_new = list(map(snakecase, cols_old))
    df.columns = cols_new
    return df

COUNTRIES = {
1: "India",
14: "Australia",
30: "Brazil",
37: "Canada",
94: "Indonesia",
148: "New Zeland",
162: "Philippines",
166: "Qatar",
184: "Singapure",
189: "South Africa",
191: "Sri Lanka",
208: "Turkey",
214: "United Arab Emirates",
215: "England",
216: "United States of America",
}
def country_name(country_id):
    return COUNTRIES[country_id]

def create_price_type(price_range):
    if price_range == 1:
        return "cheap"
    elif price_range == 2:
        return "normal"
    elif price_range == 3:
        return "expensive"
    else:
        return "gourmet"
    
COLORS = {
"3F7E00": "darkgreen",
"5BA829": "green",
"9ACD32": "lightgreen",
"CDD614": "orange",
"FFBA00": "red",
"CBCBC8": "darkred",
"FF7800": "darkred",
}
def color_name(color_code):
    return COLORS[color_code]

BOOKING = {
0: "yes",
1: "no",
}
def booking(booking_code):
    return BOOKING[booking_code]

ONLINE = {
1: "yes",
0: "no",
}
def online(online_code):
    return ONLINE[online_code]

NOW = {
1: "yes",
0: "no",
}
def now(now_code):
    return NOW[now_code]

def clean_code(df1):
    
    # acerto dos tipos de cuisines
    df1["cuisines"] = df1.loc[:, "cuisines"].astype(str).apply(lambda x: x.split(",")[0])
    
    # acerto dos códigos/nome de country, color, price, booking, online deliver, deliver now
    df1["country_code"] = df1.loc[:, "country_code"].apply(lambda x: country_name(x))
    df1['rating_color'] = df1.loc[:, 'rating_color'].apply(lambda x: color_name(x))
    df1['price_range'] = df1.loc[:, 'price_range'].apply(lambda x: create_price_type(x))
    df1["has_table_booking"] = df1.loc[:, "has_table_booking"].apply(lambda x: booking(x))
    df1["has_online_delivery"] = df1.loc[:, "has_online_delivery"].apply(lambda x: online(x))	
    df1["is_delivering_now"] = df1.loc[:, "is_delivering_now"].apply(lambda x: now(x))
    
    # exlcuir IDs duplicados
    df2 = df1.drop_duplicates(subset=['restaurant_id'], keep='first', inplace=True)
    
    # excluir coluna com apenas um dado igual para todas linhas
    df1 = df1.drop(['switch_to_order_menu'], axis=1)
    
    # excluir linhas 'nan'
    aux = df1['cuisines'] != 'nan'
    df1 = df1.loc[aux, :]
    
    #resetar index e acertar nome de colunas country
    df1 = df1.reset_index(drop=True)
    df1 = df1.rename(columns={'country_code': 'country'})
    
    return df1



#---------------------------------------------------------------------------------------------------------
#                                     INÍCIO ESTRUTURA LÓGICA
# IMPORTANDO DATASET
# LIMPEZA DATASET
#---------------------------------------------------------------------------------------------------------
df_raw = pd.read_csv('dataset/zomato.csv')
df1 = rename_columns( df_raw )
df1 = clean_code( df1 )


#---------------------------------------------------------------------------------------------------------                             
# 1.3 VISÃO CIDADES//CITIES
#---------------------------------------------------------------------------------------------------------
#SIDEBAR--------------------------------------------------------------------------------------------------

#image_path = "C:/Users/Thiago/Desktop/DADOS/repos/ftc_programacao_python_PROJETO/logo_pa.png"
image = Image.open('logo_pa.png')
st.sidebar.image(image, width=100)

st.sidebar.markdown('# TáNaMesa')
st.sidebar.markdown('## Filtros para pesquisa')

country_options = st.sidebar.multiselect('Defina países para visualizar informações',
                                         ['Philippines', 'Brazil', 'Australia', 'United States of America',
                                          'Canada', 'Singapure', 'United Arab Emirates', 'India',
                                          'Indonesia', 'New Zeland', 'England', 'Qatar', 'South Africa',
                                          'Sri Lanka', 'Turkey'],
                                         default=['Brazil', 'Australia', 'United States of America', 'England', 'United Arab Emirates'])

st.sidebar.markdown("""____""")
st.sidebar.markdown('#### Restaurantes')

price_options = st.sidebar.multiselect('Defina categorias de preço',
                                       ["cheap", 'normal', 'expensive', 'gourmet'],
                                      default=["cheap", 'normal', 'expensive', 'gourmet'])

deliver_options = st.sidebar.checkbox('Com Entrega no momento')
if deliver_options:
    linhas = (df1['is_delivering_now'] == 'yes')
    df1 = df1.loc[linhas,:]

online_options = st.sidebar.checkbox('Aceita pedido online')
if online_options:
    linhas = (df1['has_online_delivery'] == 'yes')
    df1 = df1.loc[linhas,:]
    
table_options = st.sidebar.checkbox('Com Reserva de mesa')
if table_options:
    linhas = (df1['has_table_booking'] == 'yes')
    df1 = df1.loc[linhas,:]

st.sidebar.markdown("""____""")
st.sidebar.markdown('#### desenvolvido por @tfmeneghello')


linhas_selecionadas = df1['country'].isin(country_options)
df1 = df1.loc[linhas_selecionadas,:]

linhas_selecionadas = df1['price_range'].isin(price_options)
df1 = df1.loc[linhas_selecionadas,:]

#--------------------------------------------------------------------------------------------------------
#LAYOUT--------------------------------------------------------------------------------------------------
st.write("# 🏪 Cidades registradas")
st.write("##### Gráficos interativos das cidades registradas (utilize os filtros ao lado)")

with st.container():
    st.markdown('##### Top 10 Cidades com mais restaurantes registrados')
    cols = ('restaurant_id', 'city', 'country')
    dfaux = df1.loc[:, cols].groupby(['city', 'country']).count()
    dfaux = dfaux.sort_values(['restaurant_id'], ascending=False).reset_index()
    dfaux = dfaux.rename(columns={'restaurant_id': 'reg_rest'})
    dfaux = dfaux.head(10)
    graph13 = px.bar(dfaux, x='city', y='reg_rest', color='country', text_auto=True)
    st.plotly_chart(graph13, use_container_width=True)

with st.container():
    col1, col2=st.columns(2, gap='large')
    
    with col1:
        st.markdown('##### Top 7 Cidades com restaurantes de avaliação maior que 4')
        ls = df1['aggregate_rating'] > 4
        cols = ['restaurant_id', 'city', 'country']
        dfaux = df1.loc[ls, cols].groupby(['city', 'country']).count()
        dfaux = dfaux.sort_values(['restaurant_id'], ascending=False).reset_index()
        dfaux = dfaux.rename(columns={'restaurant_id': 'reg_rest'})
        dfaux = dfaux.head(7)
        graph14 = px.bar(dfaux, x='city', y='reg_rest',color='country',text_auto=True)
        st.plotly_chart(graph14, use_container_width=True)
         
    with col2:
        st.markdown('##### Top 7 Cidades com restaurantes de avaliação menor que 2,5')
        ls = df1['aggregate_rating'] < 2.5
        cols = ['restaurant_id', 'city', 'country']
        dfaux = df1.loc[ls, cols].groupby(['city', 'country']).count()
        dfaux = dfaux.sort_values(['restaurant_id'], ascending=False).reset_index()
        dfaux = dfaux.rename(columns={'restaurant_id': 'reg_rest'})
        dfaux = dfaux.head(7)
        graph15 = px.bar(dfaux, x='city', y='reg_rest',color='country',text_auto=True)
        st.plotly_chart(graph15, use_container_width=True)

with st.container():
    st.markdown('##### Top 10 cidades com mais variadades culinárias')
    cols = ('cuisines', 'city', 'country')
    dfaux = df1.loc[:, cols].groupby(['city', 'country']).nunique()
    dfaux = dfaux.sort_values(['cuisines'], ascending=False).reset_index()
    dfaux = dfaux.rename(columns={'cuisines': 'reg_cuisines'})
    dfaux = dfaux.head(10)
    graph16 = px.bar(dfaux, x='city', y='reg_cuisines', color='country',text_auto=True)
    st.plotly_chart(graph16, use_container_width=True)


print("estou aqui")