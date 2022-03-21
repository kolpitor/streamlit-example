import os
os.system('git clone --recursive https://github.com/dmlc/xgboost')
os.system('cd xgboost')
os.system('sudo cp make/minimum.mk ./config.mk;')
os.system('sudo make -j4;')
os.system('sh build.sh')
os.system('cd python-package')
os.system('python setup.py install')
os.system('conda install -c conda-forge xgboost')
os.system('conda install -c conda-forge xgboost')
os.system('pip install graphviz')

from collections import namedtuple
import altair as alt
import math
import streamlit as st
import pandas
import numpy
import xgboost
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot

"""
# AI4Industry
"""


max_depth = st.slider("Max depth", 1, 5, 100)
