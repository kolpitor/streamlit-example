import os
import sys
!{sys.executable} -m pip install xgboost

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
