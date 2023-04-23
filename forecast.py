import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.models import NBEATSModel
from darts.models import NHiTSModel
from darts.models import NLinearModel
from darts.models import DLinearModel

from darts.utils.model_selection import train_test_split


from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.metrics import mape, r2_score
from darts.metrics import mae
from darts.datasets import EnergyDataset

import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.sidebar.header('Diamond Price Forecast')
    st.sidebar.text('MAKE SURE THE DATA IS TIME SERIES')
    st.sidebar.text('WITH ONLY TWO COLUMNS INCLUDING DATE')
    option = st.sidebar.selectbox('How do you want to get the data?', ['url', 'file'])
    if option == 'url':
        url = st.sidebar.text_input('Enter a url')
        if url:
            dataframe(url)
    else:
        file = st.sidebar.file_uploader('Choose a file', type=['csv', 'txt'])
        if file:
            # dataframe(file)
            dataframe(file)
            
def dataframe(file):
    st.header('App Forecaster')
    df = pd.read_csv(file)[::-1]
    df.reset_index(inplace=True, drop=True)
    df.date = pd.to_datetime(df.iloc[:,0])
    df.set_index(df.columns[0], inplace=True)
    df_day_avg = df.groupby(df.index.astype(str).str.split(" ").str[0]).mean().reset_index()
    
    # filler missing value
    filler = MissingValuesFiller()
    series = filler.transform(
        TimeSeries.from_dataframe(
            df_day_avg, df_day_avg.columns[0], df_day_avg.columns[1]
        ).astype(np.float32))
    
    to_do = st.radio('SELECT WHAT YOU WOULD LIKE TO DO WITH THE DATA', ['Visualize', 'Forecast'])
    if to_do == 'Visualize':
        data_visualization(series)
    else:
        forecast_data(series)

def data_visualization(series):
    button = st.button('Draw')
    if button:
        st.pyplot(series.plot())
    
def forecast_data(series):
    period = st.number_input('Enter the next period(s) you want to forecast', value=7)
    button = st.button('Forecast')
    if button:
        model_forecast(series, period)
        
def model_forecast(series, period):
    # Split train, val
    train, val = train_test_split(series, test_size=0.2)
    
    # build model
    dlinear = DLinearModel(
        input_chunk_length=30, 
        output_chunk_length=period,
        n_epochs=100,
        nr_epochs_val_period=1,
        batch_size=500\
    )
    
    dlinear.to_cpu()
    
    # fit model
    dlinear.fit(train, val_series=val, verbose=False)
    
    # check val
    dlinear_pred_series = dlinear.historical_forecasts(
        series,
        start=val[0].time_index[0],
        forecast_horizon=period,
        stride=5,
        retrain=False,
        verbose=False,
)
    display_forecast(dlinear_pred_series, series, period, start_date=val[0].time_index[0])
    
    # predict
    future_pred_series = dlinear.predict(n=period).pd_dataframe().to_numpy()
    for i, j in enumerate(future_pred_series):
        st.text(f'Period {i+1}: {j}')
        
# Display
def display_forecast(pred_series, series, period, start_date=None):
    plt.figure(figsize=(8, 5))
    if start_date:
        series = series.drop_before(start_date)
    series.univariate_component(0).plot(label="actual")
    pred_series.plot(label=("historic " + str(period) + "days forecasts"))
    plt.title(
        "R2: {}".format(r2_score(series.univariate_component(0), pred_series))
    )
    plt.legend()
    st.pyplot()
                
if __name__ == '__main__':
    main()
