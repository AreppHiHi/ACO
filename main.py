import streamlit as st
import numpy as np
import pandas as pd
import random
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath)

# ===============================
# ANT COLONY OPTIMIZATION (ACO)
# ===============================
class ACO_Knapsack:
    def __init__(self, values, w1, w2, capacity,
                 n_ants=30, n_iter=50,
                 alpha=1.0, beta=2.0, rho=0.1):

        self.values = values
        self.w1 = w1
        self.w2 = w2
        self.capacity = capacity
        self.n_items = len(values)

        self.n_ants = n_ants
        self.n_iter = n_iter
        self
