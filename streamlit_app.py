import os
os.system('git clone --recursive https://github.com/dmlc/xgboost')
os.system('cd xgboost')
os.system('sudo cp make/minimum.mk ./config.mk;')
os.system('sudo make -j4;')
os.system('sh build.sh')
os.system('cd python-package')
os.system('python setup.py install')
os.system('pip install graphviz')
os.system('pip install -U scikit-learn scipy matplotlib')

from collections import namedtuple
import altair as alt
import math
import streamlit as st
import pandas
import numpy
import xgboost
import graphviz
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot

"""
# AI4Industry
"""


max_depth_input = st.slider("Max depth", 1, 100, 5)
colsample_bytree_input = st.slider("Colsample bytree", 0, 1, 0.5)
learning_rate_input = st.slider("Learning rate", 0, 1, 0.2)
alpha_input = st.slider("Alpha", 1, 100, 10)
n_estimators_input = st.slider("n estimators", 1, 100, 20)

dataset = pandas.read_csv('weatherAUS.csv')

location_dataset = dataset["Location"].unique()
wind_dataset = dataset["WindGustDir"].unique()
date_dataset = dataset["Date"].unique()

cityTargeted = [
    "Canberra",
    "Albury",
    "Penrith",
    "Sydney",
    "MountGinini",
    "Bendigo",
    "Brisbane",
    "Portland"
    ]
dataset.drop(dataset.loc[dataset['Location'] != cityTargeted[0]].index, inplace=True)

i_RainTomorrow = dataset.columns.get_loc("RainTomorrow")
#i_Location = dataset.columns.get_loc("Location")
i_WindGustDir = dataset.columns.get_loc("WindGustDir")
i_Date = dataset.columns.get_loc("Date")
yes = dataset.iat[8, dataset.columns.get_loc("RainTomorrow")]
no = dataset.iat[0, dataset.columns.get_loc("RainTomorrow")]

for i in range(len(dataset)):
    if (dataset.iat[i, i_RainTomorrow] == yes):
        dataset.iat[i, i_RainTomorrow] = True
    else:
        dataset.iat[i, i_RainTomorrow] = False
    #dataset.iat[i, i_Location] = numpy.where(location_dataset == dataset.iat[i, i_Location])[0][0]
    if (pandas.isna(dataset.iat[i, i_WindGustDir])):
        dataset.iat[i, i_WindGustDir] = 0
    else:
        dataset.iat[i, i_WindGustDir] = numpy.where(wind_dataset == dataset.iat[i, i_WindGustDir])[0][0] + 1
    dataset.iat[i, i_Date] = numpy.where(date_dataset == dataset.iat[i, i_Date])[0][0]
    
    
dataset = dataset.astype({'RainTomorrow': 'bool'})
#dataset = dataset.astype({'Location': 'int'})
dataset = dataset.astype({'WindGustDir': 'int'})
dataset = dataset.astype({'Date': 'int'})

dataset.drop(columns=["WindDir9am", "WindDir3pm", "WindSpeed9am", "WindSpeed3pm", "Temp9am", "Temp3pm", "RainToday"], inplace=True)
dataset.drop(dataset.index[dataset.isnull().any(axis=1)], 0, inplace=True)

dataset["Humidity"] = 0.0
dataset["Pressure"] = 0.0
dataset["Cloud"] = 0.0

for i in dataset.index:
    humidity = (dataset["Humidity9am"][i] + dataset["Humidity3pm"][i]) / 2
    dataset.at[i, "Humidity"] = humidity
    pressure = (dataset["Pressure9am"][i] + dataset["Pressure3pm"][i]) / 2
    dataset.at[i, "Pressure"] = pressure
    cloud = (dataset["Cloud9am"][i] + dataset["Cloud3pm"][i]) / 2
    dataset.at[i, "Cloud"] = cloud

dataset.drop(columns=["Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm"], inplace=True)

x, y = dataset.iloc[:,[False, False, True, True, False, True, True, True, True, True, True, True, True]],dataset.iloc[:,4]

data_dmatrix = xgboost.DMatrix(data=x,label=y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

xg_reg = xgboost.XGBRegressor(colsample_bytree = colsample_bytree_input, learning_rate = learning_rate_input, max_depth = max_depth_input, alpha = alpha_input, n_estimators = n_estimators_input)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

rmse = numpy.sqrt(mean_squared_error(y_test, preds))
st.write("RMSE: %f" % (rmse))

params = {'colsample_bytree': colsample_bytree_input,'learning_rate': learning_rate_input,
                'max_depth': max_depth_input, 'alpha': alpha_input}

cv_results = xgboost.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

st.write((cv_results["test-rmse-mean"]).tail(1))

xg_reg = xgboost.train(params=params, dtrain=data_dmatrix, num_boost_round=10)
