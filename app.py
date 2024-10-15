#######################
# import libraries

import streamlit as st
import pandas as pd
import altair as alt


#######################
# Page configuration

st.set_page_config(
    page_title="Predictive Maintenance Dashboard", # this will show in the browser tab
    page_icon=":bar_chart:", # the icon showing before the title
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################
# CSS styling

st.markdown("""
<style>

</style>
""", unsafe_allow_html=True)

#######################
# Load data

df_c1_001 = pd.read_csv("data/raw/c1/c1/c_1_001.csv")

#######################
# Sidebar

with st.sidebar:
    st.title('Predictive Maintenance')
    
    
###################
# Plots

