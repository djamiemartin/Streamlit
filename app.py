import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os

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
c1_data = dataframes[0]  # Assuming the first file contains the signal data
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

# Create 2x2 grid layout
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# 1. Signal Visualization (Interactive with Altair)
with col1:
    st.title('Signal Visualization')
    signal_selection = st.radio(
        "Select CSV file", 
        ("C1", "C4", "C6"), 
        horizontal=True,
        key="signal_radio"
    )
    signal_file = f"/Users/dan/Downloads/Streamlit/{signal_selection}.csv"
    signal_data = pd.read_csv(signal_file)
    chart = alt.Chart(signal_data[['time', 'Force_X']].head(1500)).mark_line().encode(
        x='time', y='Force_X', tooltip=['time', 'Force_X']
    ).properties(title='Force X over Time').interactive()
    st.altair_chart(chart, use_container_width=True)

# 2. Frequency Analysis (Interactive with Plotly)
with col2:
    st.title('Frequency Analysis')
    frequency_selection = st.radio(
        "Select CSV file", 
        ("C1", "C4", "C6"), 
        horizontal=True,
        key="frequency_radio"
    )
    frequency_file = f"/Users/dan/Downloads/Streamlit/{frequency_selection}.csv"
    frequency_data = pd.read_csv(frequency_file)
    frequencies = np.fft.fftfreq(len(frequency_data['Force_X']), 0.02)
    fft_values = np.fft.fft(frequency_data['Force_X'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frequencies[:n // 2], y=np.abs(fft_values[:n // 2]), mode='lines', name='Frequency Domain'))
    fig.update_layout(title='Frequency Domain', xaxis_title='Frequency [Hz]', yaxis_title='Amplitude')
    fig.update_xaxes(range=[0, 100])  # Limiting x-axis range for better clarity
    st.plotly_chart(fig)

# 3. Target Variable (Wear Flute) vs Cut (Interactive with Matplotlib)
with col3:
    st.title('Target Variable (Wear Flute) vs. Cut')
    
    # Load the wear data (replace with actual path to wear data)
    wear_data = pd.read_csv('/Users/dan/Downloads/Streamlit/c1_wear.csv')  # Adjust path accordingly
    
    # Inspect wear data (uncomment to see the DataFrame info)
    st.write(wear_data.info())  # Shows data types and missing values info
    st.write(wear_data.head(10))  # Shows first 10 rows for inspection

    # Create a plot for each flute's wear over the 'cut' variable
    plt.plot(wear_data["cut"], wear_data["flute_1"], label="Flute 1")
    plt.plot(wear_data["cut"], wear_data["flute_2"], label="Flute 2")
    plt.plot(wear_data["cut"], wear_data["flute_3"], label="Flute 3")
    plt.xlabel(r'cut')
    plt.ylabel(r'wear $[\mu m]$')
    plt.grid()
    plt.legend()

    # Display the plot in Streamlit
    st.pyplot(plt)

# 4. Correlation Matrix (Interactive with Plotly)
with col4:
    st.title('Correlation Matrix')
    correlation_selection = st.radio(
        "Select CSV file", 
        ("C1", "C4", "C6"), 
        horizontal=True,
        key="correlation_radio"
    )
    correlation_file = f"/Users/dan/Downloads/Streamlit/{correlation_selection}.csv"
    correlation_data = pd.read_csv(correlation_file)
    correlation_matrix = correlation_data[['Force_X', 'Force_Y', 'Force_Z', 'Vibration_X', 'Vibration_Y', 'Vibration_Z', 'AE_RMS']].corr()

    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='Viridis',
        colorbar=dict(title='Correlation')
    ))
    fig.update_layout(title='Correlation Matrix', xaxis_title='Variables', yaxis_title='Variables')
    st.plotly_chart(fig)
