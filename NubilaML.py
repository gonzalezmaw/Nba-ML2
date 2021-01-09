import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from SLR import SLR
from RFR import RFR
from SVM import SVM

st.title("Machine Learning for Everyone")

st.sidebar.write("""
# **Nubila ML**

""")
modelSelect_name = st.sidebar.selectbox(
    "Select a Model", ("Simple Linear Regression", "Random Forest Regression", "Support Vector Machines"))


if modelSelect_name == "Simple Linear Regression":
    st.write("""
        ## **Simple Linear Regression Model**

        """)
    st.write("""
    ### **Simple Regression Method**

    """)
    SLR()
elif modelSelect_name == "Random Forest Regression":
    st.write("""
        ## **Random Forest Regression Model**

        """)
    st.write("""
    ### **Simple Regression Method**

    """)

    RFR()
elif modelSelect_name == "Support Vector Machines":
    st.write("""
        ## **Suport Vector Machines Model**

        """)
    st.write("""
    ### **Multivariate Classification Method**

    """)
    SVM()
