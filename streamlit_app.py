import os
os.system('pip install xgboost‑1.5.1‑cp310‑cp310‑win_amd64.whl')
os.system('pip install graphviz')

from collections import namedtuple
import altair as alt
import math
import streamlit as st
import pandas
import numpy
import xgboost
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot

"""
# AI4Industry
"""


max_depth = st.slider("Max depth", 1, 5, 100)
