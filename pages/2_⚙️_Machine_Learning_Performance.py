#BIBLIOTECAS
import inflection
import pickle
import math
import datetime
import seaborn              as sns
import plotly.express       as px
import plotly.graph_objects as go
import streamlit            as st
import xgboost              as xgb
import pandas               as pd
import numpy                as np
import matplotlib.pyplot    as plt
from sklearn import preprocessing as pp
from sklearn import metrics       as mt
from PIL     import Image

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='Machine Learning Performance', page_icon='⚙️', layout='wide')

#---------------------------------------------------------------------------------------------------------
# DEF FUNÇÕES
#---------------------------------------------------------------------------------------------------------
model = pickle.load(open('params/model_rossmann.pkl', 'rb'))
comp_distance_scaler = pickle.load(open('params/competition_distance_scaler.pkl', 'rb'))
comp_timemonth_scaler = pickle.load(open('params/competition_time_month_scaler.pkl', 'rb'))
promo_timeweek_scaler = pickle.load(open('params/promo_time_week_scaler.pkl', 'rb'))
year_scaler = pickle.load(open('params/year_scaler.pkl', 'rb'))
store_type_scaler = pickle.load(open('params/store_type_scaler.pkl', 'rb'))

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

    
def train_test(df6):
    #achar a última data da venda
    last_date = df6.loc[:, ['date','store']].groupby(['store']).max().reset_index()['date'][0]

    # separar data corte das ultimas 6 semanas para teste e treino
    cut_date = last_date-datetime.timedelta(days=6*7)
    train_filter = df6['date'] < cut_date
    test_filter = df6['date'] >= cut_date

    # separar dataset treino e teste por data
    X_train = df6.loc[train_filter, :]
    y_train = X_train['sales']

    X_test = df6.loc[test_filter, :]
    y_test = X_test['sales']

    x_train = X_train.drop(['date', 'sales'], axis=1)
    x_test = X_test.drop(['date', 'sales'], axis=1)
    
    return x_train, x_test, y_train, y_test

def prediction(model, x_test):
    pred = model.predict(x_test)
    
    return pred

def performance(df9, pred):
    #achar a última data da venda
    last_date = df9.loc[:, ['date','store']].groupby(['store']).max().reset_index()['date'][0]

    # separar data corte das ultimas 6 semanas para teste e treino
    cut_date = last_date-datetime.timedelta(days=6*7)
    perf_filter = df9['date'] >= cut_date
    df9 = df9.loc[perf_filter, :]
    df9['sales'] = np.expm1(df9['sales']) #rescaling vendas
    df9['predictions'] = np.expm1(pred) #rescaling previsões
    df91 = df9.loc[:, ['predictions', 'store']].groupby('store').sum().reset_index()

    #verificar mae e mape por loja
    df9_aux1 = df9.loc[:, ['sales', 'predictions', 'store']].groupby('store').apply(lambda x: mt.mean_absolute_error(x['sales'], x['predictions'])).reset_index()
    df9_aux2 = df9.loc[:, ['sales', 'predictions', 'store']].groupby('store').apply(lambda x: mt.mean_absolute_percentage_error(x['sales'], x['predictions'])).reset_index()
    #merge mae e mape
    df9_aux3 = pd.merge(df9_aux1, df9_aux2, how='inner', on='store')

    #merge erros e somas de predição
    df92 = pd.merge(df91, df9_aux3, how='inner', on='store')
    df92 = df92.rename(columns={'0_x': 'MAE', '0_y':'MAPE'})
    #possíveis cenários
    df92['worst_scenario'] = df92['predictions'] -df92['MAE']
    df92['best_scenario'] = df92['predictions'] +df92['MAE']
    df92 = df92[['store', 'predictions', 'worst_scenario', 'best_scenario', 'MAE', 'MAPE']]
    df92 = df92.round(2)
    
    return df92

def graph_performance(df10, pred):
    #achar a última data da venda
    last_date = df10.loc[:, ['date','store']].groupby(['store']).max().reset_index()['date'][0]

    # separar data corte das ultimas 6 semanas para teste e treino
    cut_date = last_date-datetime.timedelta(days=6*7)
    perf_filter = df10['date'] >= cut_date
    df10 = df10.loc[perf_filter, :]
    df10['sales'] = np.expm1(df10['sales']) #rescaling vendas
    df10['predictions'] = np.expm1(pred) #rescaling previsões
    df10['mpe'] = (df10['sales']-df10['predictions'])/df10['sales']
    df10['error'] = df10['sales']-df10['predictions']
    return df10

def ml_error(model_name, testy, haty):
    mae = mt.mean_absolute_error(testy, haty)
    mape = mt.mean_absolute_percentage_error(testy, haty)
    rmse = np.sqrt(mt.mean_squared_error(testy, haty))
                   
    return pd.DataFrame({'Model_Name': model_name,
                         'MAE': mae,
                         'MAPE': mape,
                         'RMSE': rmse}, index=[0])

@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')
#---------------------------------------------------------------------------------------------------------
#                                     INÍCIO ESTRUTURA LÓGICA
# IMPORTANDO DATASET
# LIMPEZA DATASET
#---------------------------------------------------------------------------------------------------------
df_sales_raw = pd.read_csv("dataset/train.csv", low_memory=False)
df_store_raw = pd.read_csv("dataset/store.csv", low_memory=False)
df_train_raw = pd.merge(df_sales_raw, df_store_raw, on='Store', how='left')
df_train_raw = df_train_raw[~df_train_raw['Open'].isnull()]
df_train_raw = df_train_raw[df_train_raw['Open'] != 0]

df4 = data_cleaning_train( df_train_raw )
# feature engineering
df5 = feature_engineering_train( df4 )
# data preparation
df_train = data_preparation_train( df5 )

x_train_ml, x_test_ml, y_train_ml, y_test_ml= train_test(df_train)

pred  = prediction(model, x_test_ml)

df_p = performance(df_train, pred)

df_gp = graph_performance(df_train, pred)

df_train_csv = convert_df(df_train)
#---------------------------------------------------------------------------------------------------------                             
# 1.2 Machine Learning Performance//
#---------------------------------------------------------------------------------------------------------
#SIDEBAR--------------------------------------------------------------------------------------------------

image = Image.open('rossmann3.png')
st.sidebar.image(image, width=100)

st.sidebar.markdown('# Rossmann')
st.sidebar.markdown('## Machine Learning Performance')

st.sidebar.markdown("""____""")
st.sidebar.markdown('### Dados tratados')
st.sidebar.download_button(label = 'Download Train.csv',
                           data = df_predictions_csv,
                           file_name = 'df_predictions.csv',
                           mime = 'text/csv')

st.sidebar.markdown("""____""")

st.sidebar.markdown('#### desenvolvido por @tfmeneghello')

#--------------------------------------------------------------------------------------------------------
#LAYOUT--------------------------------------------------------------------------------------------------
st.write("# ⚙️ Machine Learning Performance")
st.write("##### Principais índices do treinamento do algoritmo de Machine Learning")

with st.container():
    col1, col2, col3 = st.columns([0.4, 0.3, 0.3], gap='small')
    with col1:
        col1.metric('Algoritmo utilizado:', 'XGBoost Regressor')
        
    with col2:
        train_stores = x_train_ml['store'].count()
        col2.metric('Nº registros utilizados em treino:', train_stores)

    with col3:
        test_stores = x_test_ml['store'].count()
        col3.metric('Nº registros utilizados em teste:', test_stores)

with st.container():
    st.write("##### Métricas de Performance")
    col1, col2, col3=st.columns(3, gap='small')
    with col1:
        mae = np.round(mt.mean_absolute_error(np.expm1(y_test_ml), np.expm1(pred)), 3)
        col1.metric('Erro MAE:', mae)
        
    with col2:
        mape = np.round(mt.mean_absolute_percentage_error(np.expm1(y_test_ml), np.expm1(pred)), 3)
        col2.metric('Erro MAPE:', mape)

    with col3:
        rmse = np.round(np.sqrt(mt.mean_squared_error(np.expm1(y_test_ml), np.expm1(pred))), 3)
        col3.metric('Erro RMSE:', rmse)

with st.container():
    col1, col2 = st.columns(2, gap='small')
    with col1:
        st.write("##### Total Performance")
        df93 = df_p.loc[:, ['predictions', 'worst_scenario', 'best_scenario']].apply(lambda x: np.sum(x), axis=0).reset_index()
        df93 = df93.rename(columns={'index': 'scenario', 0: 'values'})
        df93['values'] = df93['values'].map('R$ {:,.2f}'.format)
        st.dataframe(df93, use_container_width=True)
        
    with col2:
        st.write("##### Performance por Loja")
        df_p['predictions'] = df_p['predictions'].map('R$ {:,.2f}'.format)
        df_p['worst_scenario'] = df_p['worst_scenario'].map('R$ {:,.2f}'.format)
        df_p['best_scenario'] = df_p['best_scenario'].map('R$ {:,.2f}'.format)
        df_tab = df_p.copy()
        df_tab = df_tab.set_index('store')
        st.dataframe(df_tab, use_container_width=True)
    
with st.container():
    aux_compare_date = df_gp.loc[:, ['predictions', 'sales', 'date']].groupby('date').sum().reset_index()
    fig1 = px.line(aux_compare_date, x='date', y=['sales', 'predictions'], template='plotly', title = 'Gráfico comparativo entre Vendas e Previsões')
    st.plotly_chart(fig1 , use_container_width=True)

with st.container():
    fig2 = px.scatter(df_p, x='store', y='MAPE', template='plotly', title='Distribuição do Erro Percentual Médio Absoluto (MAPE) de previsão por loja')
    st.plotly_chart(fig2 , use_container_width=True)

with st.container():
    col1, col2 = st.columns(2, gap='small')
    with col1:
        st.write("##### Erro percentual médio de previsão por dia")
        dfaux3 = df_gp.loc[:, ['mpe', 'date']].groupby('date').mean().reset_index()
        fig3 = px.line(dfaux3, x='date', y='mpe', template='plotly', title = 'MPE > 0 = prev. subestimada; MPE < 0 prev. superestimada')
        st.plotly_chart(fig3 , use_container_width=True)
        
    with col2:
        st.write("##### Distribuição do erro")
        fig4 = px.histogram(df_gp['error'])
        st.plotly_chart(fig4 , use_container_width=True)








print("estou aqui")
