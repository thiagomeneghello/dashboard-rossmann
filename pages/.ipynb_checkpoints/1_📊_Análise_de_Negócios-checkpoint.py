#BIBLIOTECAS
import pandas as pd
import numpy as np
import inflection
import pickle
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import xgboost as xgb
import math
import datetime
from sklearn import preprocessing as pp
import matplotlib.pyplot as plt

st.set_page_config(page_title='Análises de Negócios', page_icon='📊', layout='wide')

#---------------------------------------------------------------------------------------------------------
# DEF FUNÇÕES
#---------------------------------------------------------------------------------------------------------
model = pickle.load(open('pickle/model_rossmann.pkl', 'rb'))
comp_distance_scaler = pickle.load(open('pickle/comp_distance_scaler.pkl', 'rb'))
comp_timemonth_scaler = pickle.load(open('pickle/comp_timemonth_scaler.pkl', 'rb'))
promo_timeweek_scaler = pickle.load(open('pickle/promo_timeweek_scaler.pkl', 'rb'))
year_scaler = pickle.load(open('pickle/year_scaler.pkl', 'rb'))
store_type_scaler = pickle.load(open('pickle/store_type_scaler.pkl', 'rb'))

def data_cleaning(df1):
    cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo',
                'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
                'CompetitionDistance', 'CompetitionOpenSinceMonth',
                'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                'Promo2SinceYear', 'PromoInterval']

    snakecase = lambda x: inflection.underscore(x)
    cols_new = list(map(snakecase, cols_old))
    df1.columns = cols_new
    
    ## 1.3 Data types
    df1['date'] = pd.to_datetime(df1['date'])
    
    ## 1.5 Fillout NA
    #competition_distance
    df1['competition_distance'] = [90000 if math.isnan(i) else i for i in df1['competition_distance']]
    
    #competition_open_since_month
    df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else 
                                                    x['competition_open_since_month'], axis = 1)
    
    #competition_open_since_year
    df1['competition_open_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else
                                                   x['competition_open_since_year'], axis = 1)            
    #promo2_since_week
    df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else
                                         x['promo2_since_week'], axis = 1)
    
    #promo2_since_year
    df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else 
                                         x['promo2_since_year'], axis = 1)
    
    #promo_interval  
    month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    df1['promo_interval'] = df1['promo_interval'].fillna(0)
    df1['month_map'] = df1['date'].dt.month.map(month_map)
    df1['is_promo'] = df1[['promo_interval', 'month_map']].apply(lambda x: 0 if x['promo_interval']==0 else
                                                                 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)

        
    ## 1.6 Change types
    df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int) 
    df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)
    df1['promo2_since_week'] = df1['promo2_since_week'].astype(int) 
    df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)

    return df1

def feature_engineering(df2):
    # year
    df2['year'] = df2['date'].dt.year
    #month
    df2['month'] = df2['date'].dt.month
    #day
    df2['day'] = df2['date'].dt.day
    #week of year
    df2['week_of_year'] = df2['date'].dt.isocalendar().week
    df2['week_of_year'] = df2['week_of_year'].astype(int)
    #year week
    df2['year_week'] = df2['date'].dt.strftime('%Y-%W')

    # competition since
    df2['competition_since'] = df2.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'],
                                                                     month=x['competition_open_since_month'],
                                                                     day=1), axis=1)

    df2['competition_time_month'] = ((df2['date'] - df2['competition_since'])/30).apply(lambda x: x.days).astype(int)


    # promo since
    df2['promo_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str)
    df2['promo_since'] = df2['promo_since'].apply(lambda x: datetime.datetime.strptime(x +'-1',
                                                                                       '%Y-%W-%w') - 
                                                                                        datetime.timedelta(days=7))
    df2['promo_time_week'] = ((df2['date'] - df2['promo_since'])/7).apply(lambda x: x.days).astype(int)

    #assortment
    df2['assortment'] = ['basic' if i == 'a' else 'extra' if i == 'b' else 'extended' for i in df2['assortment']]

    #state_holiday
    df2['state_holiday'] = ['public_holiday' if i == 'a' else 'easter_holiday' if i == 'b' else 'christmas' if i == 'c' else 'regular_day' for i in df2['state_holiday']]

    ## 3.1 Filtragem das linhas
    df2 = df2.loc[df2['open'] != 0, :].reset_index()
    ## 3.2 Seleção das colunas
    cols_drop = ['open', 'month_map', 'promo_interval']
    df2 = df2.drop(cols_drop, axis=1)

    return df2

def data_preparation( df5 ):
    df5['competition_distance'] = comp_distance_scaler.fit_transform(df5[['competition_distance']].values)
    df5['competition_time_month'] = comp_timemonth_scaler.fit_transform(df5[['competition_time_month']].values)
    df5['promo_time_week'] = promo_timeweek_scaler.fit_transform(df5[['promo_time_week']].values)
    df5['year'] = year_scaler.fit_transform(df5[['year']].values)

    ## 5.3 Transformation
    #state_holiday - One Hot Encoding, bom para estado de coisas
    df5 = pd.get_dummies(df5, prefix=['state_holiday'], columns=['state_holiday'], dtype='int64')

    #store_type - Label Encoding, bom para variáveis categóricas sem ordem ou relevância
    df5['store_type'] = store_type_scaler.fit_transform(df5['store_type'])

    #assortment - Ordinal Encoding, bom para variáveis com intraordem.
    assort_dict = {'basic': 1, 'extra': 2, 'extended': 3}
    df5['assortment'] = df5['assortment'].map(assort_dict)

    df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x*(2*np.pi/7)))
    df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x*(2*np.pi/7)))

    df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x*(2*np.pi/12)))
    df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x*(2*np.pi/12)))

    df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x: np.sin(x*(2*np.pi/52)))
    df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos(x*(2*np.pi/52)))

    df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x*(2*np.pi/30)))
    df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x*(2*np.pi/30)))

    cols_selected = ['store', 'promo','store_type', 'assortment', 'competition_distance',
                     'competition_open_since_month','competition_open_since_year','promo2',
                     'promo2_since_week','promo2_since_year','competition_time_month','promo_time_week',
                     'day_of_week_sin','day_of_week_cos','month_cos','month_sin','week_of_year_cos',
                     'week_of_year_sin','day_sin','day_cos']

    return df5[cols_selected]

def data_cleaning_train(df1):
    cols_old = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo',
                'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
                'CompetitionDistance', 'CompetitionOpenSinceMonth',
                'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                'Promo2SinceYear', 'PromoInterval']

    snakecase = lambda x: inflection.underscore(x)
    cols_new = list(map(snakecase, cols_old))
    df1.columns = cols_new
    
    ## 1.3 Data types
    df1['date'] = pd.to_datetime(df1['date'])
    
    ## 1.5 Fillout NA
    #competition_distance
    df1['competition_distance'] = [90000 if math.isnan(i) else i for i in df1['competition_distance']]
    
    #competition_open_since_month
    df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else 
                                                    x['competition_open_since_month'], axis = 1)
    
    #competition_open_since_year
    df1['competition_open_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else
                                                   x['competition_open_since_year'], axis = 1)            
    #promo2_since_week
    df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else
                                         x['promo2_since_week'], axis = 1)
    
    #promo2_since_year
    df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else 
                                         x['promo2_since_year'], axis = 1)
    
    #promo_interval  
    month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    df1['promo_interval'] = df1['promo_interval'].fillna(0)
    df1['month_map'] = df1['date'].dt.month.map(month_map)
    df1['is_promo'] = df1[['promo_interval', 'month_map']].apply(lambda x: 0 if x['promo_interval']==0 else
                                                                 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)

        
    ## 1.6 Change types
    df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int) 
    df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)
    df1['promo2_since_week'] = df1['promo2_since_week'].astype(int) 
    df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)

    return df1

def feature_engineering_train(df2):
    # year
    df2['year'] = df2['date'].dt.year
    #month
    df2['month'] = df2['date'].dt.month
    #day
    df2['day'] = df2['date'].dt.day
    #week of year
    df2['week_of_year'] = df2['date'].dt.isocalendar().week
    df2['week_of_year'] = df2['week_of_year'].astype(int)
    #year week
    df2['year_week'] = df2['date'].dt.strftime('%Y-%W')

    # competition since
    df2['competition_since'] = df2.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'],
                                                                     month=x['competition_open_since_month'],
                                                                     day=1), axis=1)

    df2['competition_time_month'] = ((df2['date'] - df2['competition_since'])/30).apply(lambda x: x.days).astype(int)


    # promo since
    df2['promo_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str)
    df2['promo_since'] = df2['promo_since'].apply(lambda x: datetime.datetime.strptime(x +'-1',
                                                                                       '%Y-%W-%w') - 
                                                                                        datetime.timedelta(days=7))
    df2['promo_time_week'] = ((df2['date'] - df2['promo_since'])/7).apply(lambda x: x.days).astype(int)

    #assortment
    df2['assortment'] = ['basic' if i == 'a' else 'extra' if i == 'b' else 'extended' for i in df2['assortment']]

    #state_holiday
    df2['state_holiday'] = ['public_holiday' if i == 'a' else 'easter_holiday' if i == 'b' else 'christmas' if i == 'c' else 'regular_day' for i in df2['state_holiday']]

    ## 3.1 Filtragem das linhas
    df2 = df2.loc[(df2['open'] != 0) & (df2['sales'] > 0 ), :].reset_index()
    ## 3.2 Seleção das colunas
    cols_drop = ['customers', 'open', 'month_map', 'promo_interval']
    df2 = df2.drop(cols_drop, axis=1)

    return df2


def data_preparation_train( df5 ):
    df5['competition_distance'] = comp_distance_scaler.fit_transform(df5[['competition_distance']].values)
    df5['competition_time_month'] = comp_timemonth_scaler.fit_transform(df5[['competition_time_month']].values)
    df5['promo_time_week'] = promo_timeweek_scaler.fit_transform(df5[['promo_time_week']].values)
    df5['year'] = year_scaler.fit_transform(df5[['year']].values)

    ## 5.3 Transformation
    #state_holiday - One Hot Encoding, bom para estado de coisas
    df5 = pd.get_dummies(df5, prefix=['state_holiday'], columns=['state_holiday'], dtype='int64')

    #store_type - Label Encoding, bom para variáveis categóricas sem ordem ou relevância
    df5['store_type'] = store_type_scaler.fit_transform(df5['store_type'])

    #assortment - Ordinal Encoding, bom para variáveis com intraordem.
    assort_dict = {'basic': 1, 'extra': 2, 'extended': 3}
    df5['assortment'] = df5['assortment'].map(assort_dict)

    df5['sales'] = np.log1p(df5['sales'])

    df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x*(2*np.pi/7)))
    df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x*(2*np.pi/7)))

    df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x*(2*np.pi/12)))
    df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x*(2*np.pi/12)))

    df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x: np.sin(x*(2*np.pi/52)))
    df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos(x*(2*np.pi/52)))

    df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x*(2*np.pi/30)))
    df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x*(2*np.pi/30)))

    cols_selected_full = ['store', 'promo','store_type', 'assortment', 'competition_distance',
                          'competition_open_since_month','competition_open_since_year','promo2',
                          'promo2_since_week','promo2_since_year','competition_time_month','promo_time_week',
                          'day_of_week_sin','day_of_week_cos','month_cos','month_sin','week_of_year_cos',
                          'week_of_year_sin','day_sin','day_cos', 'date', 'sales']

    return df5[cols_selected_full]

def get_prediction(model, original_data, test_data):
    #prediction
    pred = model.predict(test_data)
    # join pred into original date
    original_data['predictions'] = np.expm1(pred)

    return original_data
####################################################################################################################

#@st.cache_data
#def convert_df(df):
#    return df.to_csv().encode('utf-8')

#---------------------------------------------------------------------------------------------------------
#                                     INÍCIO ESTRUTURA LÓGICA
# IMPORTANDO DATASET
# LIMPEZA DATASET
#---------------------------------------------------------------------------------------------------------
df10 = pd.read_csv('./dataset/test.csv', low_memory=False)
df_store_raw = pd.read_csv('./dataset/store.csv', low_memory=False)
df_test = pd.merge(df10, df_store_raw, how='left', on='Store')
df_test = df_test[~df_test['Open'].isnull()]
df_test = df_test[df_test['Open'] != 0]
df_test = df_test.drop(['Id'], axis=1)
# data cleaning
df1 = data_cleaning( df_test )
# feature engineering
df2 = feature_engineering( df1 )
# data preparation
df3 = data_preparation( df2 )
# prediction
df_response = get_prediction( model, df_test, df3 )

#---------------------------------------------------------------------------------------------------------                             
# 1.2 VISÃO NEGÓCIOS//
#---------------------------------------------------------------------------------------------------------
#SIDEBAR--------------------------------------------------------------------------------------------------

image = Image.open('rossmann3.png')
st.sidebar.image(image, width=100)

st.sidebar.markdown('# Rossmann')
st.sidebar.markdown('## Filtros para pesquisa')

st.sidebar.markdown("""____""")
st.sidebar.markdown('#### Tipos de Sortimento')

assortment_options = st.sidebar.multiselect('Defina tipos de sortimentos de loja',
                                         ['basic', 'extra', 'extended'],
                                         default=['basic', 'extra', 'extended'])

st.sidebar.markdown("""____""")
st.sidebar.markdown('#### Tipos de loja')

store_options = st.sidebar.multiselect('Defina categorias de loja',
                                       ["a", 'b', 'c', 'd'],
                                      default=["a", 'b', 'c', 'd'])

st.sidebar.markdown("""____""")
promo1_options = st.sidebar.checkbox('Promoção Tradicional', value=True)
if not promo1_options:
    linhas = (df1['promo'] == 0)
    df_response = df_response.loc[linhas,:]

st.sidebar.markdown("""____""")
promo2_options = st.sidebar.checkbox('Promoção Extendida', value=True)
if not promo2_options:
    linhas = (df1['promo2'] == 0)
    df_response = df_response.loc[linhas,:]
    
st.sidebar.markdown("""____""")
st.sidebar.markdown('#### Distância concorrência')
comp_distance = st.sidebar.slider('em metros', min_value=0, max_value=90000, value=(0.0, 5000), step=2500)
#st.write('Intervalo de distância da concorrência:' comp_distance)

st.sidebar.markdown("""____""")
st.sidebar.markdown('#### desenvolvido por @tfmeneghello')


linhas_selecionadas = df_response['assortment'].isin(assortment_options)
df_response = df_response.loc[linhas_selecionadas,:]

linhas_selecionadas = df_response['store_type'].isin(store_options)
df_response = df_response.loc[linhas_selecionadas,:]

linhas_selecionadas = (df_response['competition_distance'] < comp_distance[0]) & (df_response['competition_distance'] >= comp_distance[0])
df_response = df_response.loc[linhas_selecionadas,:]

#--------------------------------------------------------------------------------------------------------
#LAYOUT--------------------------------------------------------------------------------------------------
st.write("# 📊 Análises de Negócios")
st.write("##### Gráficos interativos das previsões de vendas por loja (utilize os filtros ao lado)")

with st.container():
    cols = ['predictions', 'store']
    dfaux = df_response.loc[:, cols].groupby('store').sum()
    dfaux = dfaux.sort_values(['predictions'], ascending=False).reset_index()
    dfaux = dfaux.rename(columns={'predictions': 'previsão_de_venda'})
    dfaux = dfaux.head(7)
    graph2 = px.bar(dfaux, x='store', y='previsão_de_venda', color='store', template='plotly', text_auto=True, title='Top 7 Lojas com maiores previões de venda')
    st.plotly_chart(graph2, use_container_width=True)
    
#with st.container():
 #   cols = ['city', 'country']
  #  dfaux = df1.loc[:, cols].groupby('country').nunique()
   # dfaux = dfaux.sort_values(['city'], ascending=False).reset_index()
    #dfaux = dfaux.rename(columns={'city': 'reg_cities'})
    #graph1 = px.bar(dfaux, x='country', y='reg_cities', color='country', text_auto=True, title='Quantidade de cidades registradas por país')
    #st.plotly_chart(graph1, use_container_width=True)

#with st.container():
 #   col1, col2=st.columns(2, gap='small')
    
  #  with col1:
   #     cols = ('votes', 'country')
    #    dfaux = df1.loc[:, cols].groupby("country").sum()
     #   dfaux['média'] = (dfaux['votes'] / df1['votes'].sum())*100
      #  dfaux = dfaux.sort_values(['média'], ascending=False).reset_index()
       # graph8 = px.pie(dfaux, values='média', names='country', title='Quantidade de avaliações por país')
        #st.plotly_chart(graph8, use_container_width=True)
         
   # with col2:
    #    cols = ('average_cost_for_two', 'country')
     #   dfaux = df1.loc[:, cols].groupby("country").mean()
      #  dfaux = dfaux.sort_values(['average_cost_for_two'], ascending=False).reset_index()
       # graph11 = px.bar(dfaux, x='country', y='average_cost_for_two', color='country', text_auto=True, title='Média de preço para duas pessoas')
        #st.plotly_chart(graph11, use_container_width=True)

#with st.container():
    #cols = ['restaurant_id', 'aggregate_rating', 'country']
    #dfaux = df1.loc[:, cols].groupby(['country', 'aggregate_rating']).count()
    #dfaux = dfaux.sort_values(['restaurant_id'], ascending=False).reset_index()
    #dfaux = dfaux.rename(columns={'restaurant_id': 'n_restaurants_id'})
    #graph12 = px.scatter(dfaux, x='country', y='aggregate_rating', size='n_restaurants_id', range_y=(-1,5.5),title = 'Distribuição da quantidade de restaurantes por nota média em cada país')
    #st.plotly_chart(graph12, use_container_width=True)


print("estou aqui")
