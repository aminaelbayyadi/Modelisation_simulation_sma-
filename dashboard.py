import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------- CONFIG --------
st.set_page_config(page_title="Dashboard PFE SMA", layout="wide")

# -------- STYLE --------
st.markdown("""
<style>
.big-title { font-size:30px; font-weight:bold; color:#1f77b4; }
.metric-box { background-color:#f0f2f6; padding:10px; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title"> Dashboard - Simulation Multi-Agents</p>', unsafe_allow_html=True)
