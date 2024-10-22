import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import plotly.graph_objects as go
import os
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(page_title="Predictive Maintenance Dashboard", page_icon=":bar_chart:", layout="wide", initial_sidebar_state="expanded")

# Function to load data from CSV files
def load_csv_data(folder_path):
    dataframes = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            dataframes.append(df)
    return dataframes

# Path to your CSV files folder
folder_path = '/Users/dan/Downloads/Streamlit'
dataframes = load_csv_data(folder_path)

# Assuming first file contains signal data
c1_data = dataframes[1]  # Assuming the first file contains the signal data
c1_data.columns = ["Force_X", "Force_Y", "Force_Z", "Vibration_X", "Vibration_Y", "Vibration_Z", "AE_RMS"]
n = c1_data["Force_X"].shape[0]  # number of measurements
t = 0.02 * np.arange(n)  # time in milliseconds
c1_data["time"] = t

# Sidebar with buttons
with st.sidebar:
    st.title("Predictive Maintenance")
    st.markdown("---")
    page = st.button("Problem and Data")
    if page:
        st.write("Displaying Problem and Data")
    page = st.button("Exploratory Data Analysis")
    if page:
        st.write("Displaying Exploratory Data Analysis")
    page = st.button("Machine Learning")
    if page:
        st.write("Displaying Machine Learning")
    page = st.button("Deep Learning")
    if page:
        st.write("Displaying Deep Learning")

# Function to load wear data (c1_wear.csv)
def load_wear_data():
    try:
        wear_data = pd.read_csv('/Users/dan/Downloads/Streamlit/c1_wear.csv')
        return wear_data
    except Exception as e:
        st.error(f"Error loading wear data: {e}")
        return None

# Create 2x2 grid layout
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# 1. Signal Visualization (Interactive with Altair)
with col1:
    st.title('Signal Visualization')
    signal_selection = st.radio("Select CSV file", ("C1", "C4", "C6"), horizontal=True, key="signal_radio")
    signal_dropdown = st.selectbox('Select signal to plot...', ['Force_X', 'Force_Y', 'Force_Z'], key='signal_dropdown')
    
    plot_data = c1_data[['time', signal_dropdown]].head(1500)
    chart = alt.Chart(plot_data).mark_line().encode(
        x='time', y=signal_dropdown, tooltip=['time', signal_dropdown]
    ).properties(title='Force X over Time').interactive()
    st.altair_chart(chart, use_container_width=True)

# 2. Frequency Analysis (Interactive with Plotly)
with col2:
    st.title('Frequency Analysis')
    frequency_selection = st.radio("Select CSV file", ("C1", "C4", "C6"), horizontal=True, key="frequency_radio")
    frequency_dropdown = st.selectbox('Select frequency to be plot', ['Force_X', 'Force_Y', 'Force_Z'], key='frequency_dropdown')
    
    frequencies = np.fft.fftfreq(len(c1_data[frequency_dropdown]), 0.02)
    fft_values = np.fft.fft(c1_data[frequency_dropdown])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frequencies[:n // 2], y=np.abs(fft_values[:n // 2]), mode='lines', name='Frequency Domain'))
    fig.update_layout(title='Frequency Domain', xaxis_title='Frequency [Hz]', yaxis_title='Amplitude')
    fig.update_xaxes(range=[0, 100])  # Limiting x-axis range for better clarity
    st.plotly_chart(fig)

# 3. Target Variable (Wear Flute) vs Cut (Interactive with Plotly)
with col3:  # Place the interactive chart in the grid
    st.title('Target Variable (Wear Flute) vs Cut')
    wear_data = load_wear_data()
    if wear_data is not None:
        # Create an interactive plot with Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wear_data["cut"], y=wear_data["flute_1"], mode='lines+markers', name="Flute 1"))
        fig.add_trace(go.Scatter(x=wear_data["cut"], y=wear_data["flute_2"], mode='lines+markers', name="Flute 2"))
        fig.add_trace(go.Scatter(x=wear_data["cut"], y=wear_data["flute_3"], mode='lines+markers', name="Flute 3"))

        fig.update_layout(
            title='Target Variable (Wear Flute) vs Cut',
            xaxis_title='Cut',
            yaxis_title='Wear [μm]',
            legend_title='Flutes',
            template='plotly_white'
        )

        # Display the interactive plot in Streamlit, fit to the column
        st.plotly_chart(fig, use_container_width=True)

# 4. Correlation Matrix (Interactive with Plotly)
with col4:
    st.title('Correlation Matrix')
    correlation_selection = st.radio("Select CSV file", ("C1", "C4", "C6"), horizontal=True, key="correlation_radio")
    correlation_dropdown = st.selectbox('Select features to display correlation', ['Force_X', 'Force_Y', 'Force_Z', 'Vibration_X', 'Vibration_Y', 'Vibration_Z'], key='correlation_dropdown')
    
    correlation_matrix = c1_data[[correlation_dropdown]].corr()

    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='Viridis',
        colorbar=dict(title='Correlation')
    ))
    fig.update_layout(title='Correlation Matrix', xaxis_title='Variables', yaxis_title='Variables')
    st.plotly_chart(fig)
