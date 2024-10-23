import streamlit as st
import pandas as pd
import os
import numpy as np
from scipy import fft
import plotly.graph_objects as go

# Set Streamlit page configuration
st.set_page_config(page_title="Predictive Maintenance Dashboard", page_icon=":bar_chart:", layout="wide", initial_sidebar_state="expanded")

# Function to load wear data based on selected chart
def load_wear_data(selected_chart):
    wear_file_path = os.path.join(folder_path, selected_chart.lower(), f"{selected_chart.lower()}_wear.csv")
    try:
        wear_data = pd.read_csv(wear_file_path)
        return wear_data
    except Exception as e:
        st.error(f"Error loading wear data: {e} (Tried path: {wear_file_path})")
        return None

# Load main data for correlation matrix
def load_main_data(selected_chart):
    folder_path = f'/Users/dan/Downloads/Streamlit/data/dashboard/{selected_chart.lower()}/{selected_chart.lower()}/'
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and '_freq' not in f]
    
    if not csv_files:
        st.error("No valid data files found.")
        return None
    
    selected_file = csv_files[0]  # Use the first file found
    file_path = os.path.join(folder_path, selected_file)
    return pd.read_csv(file_path)

# Path to your CSV files folder
folder_path = '/Users/dan/Downloads/Streamlit/data/dashboard/'

# Sidebar with buttons
with st.sidebar:
    st.title("Predictive Maintenance")
    st.markdown("---")
    if st.button("Problem and Data"):
        st.write("Displaying Problem and Data")
    if st.button("Exploratory Data Analysis"):
        st.write("Displaying Exploratory Data Analysis")
    if st.button("Machine Learning"):
        st.write("Displaying Machine Learning")
    if st.button("Deep Learning"):
        st.write("Displaying Deep Learning")

# Create 2x2 grid layout
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# 1. Signal Visualization (Interactive with Altair)
with col1:
    st.title('Signal Visualization')
    selected_chart = st.radio("Select Chart", ("C1", "C4", "C6"), key='signal_chart_selection')
    folder_path_signals = f'{folder_path}/{selected_chart.lower()}/{selected_chart.lower()}/'

    csv_files = [f for f in os.listdir(folder_path_signals) if f.endswith('.csv') and '_freq' not in f]
    selected_file = st.selectbox("Select a CSV file", csv_files, key='signal_file_selection')

    def load_first_chart_data(file_name):
        file_path = os.path.join(folder_path_signals, file_name)
        data = pd.read_csv(file_path)
        return data

    first_chart_data = load_first_chart_data(selected_file)
    first_chart_data.columns = ["Force_X", "Force_Y", "Force_Z", "Vibration_X", "Vibration_Y", "Vibration_Z", "AE_RMS", "time"]

    signal_to_plot = st.selectbox("Choose signal to plot", ["Force_X", "Force_Y", "Force_Z", "Vibration_X", "Vibration_Y", "Vibration_Z", "AE_RMS"], key='signal_selection')
    st.write(f"**Data from: {selected_file}**")
    st.line_chart(first_chart_data[['time', signal_to_plot]].set_index('time'))

# 2. Frequency Analysis (Interactive with Plotly)
with col2:
    st.title('Frequency Analysis')
    selected_chart = st.radio("Select Chart", ("C1", "C4", "C6"), key='frequency_chart_selection')
    folder_path_frequencies = f'{folder_path}/{selected_chart.lower()}/{selected_chart.lower()}/'

    csv_files = [f for f in os.listdir(folder_path_frequencies) if f.endswith('.csv') and '_freq' not in f]
    selected_file = st.selectbox("Select a CSV file", csv_files, key='frequency_file_selection')

    def load_data(file_name):
        file_path = os.path.join(folder_path_frequencies, file_name)
        data = pd.read_csv(file_path)
        return data

    chart_data = load_data(selected_file)
    chart_data.columns = ["Force_X", "Force_Y", "Force_Z", "Vibration_X", "Vibration_Y", "Vibration_Z", "AE_RMS", "time"]

    signal_to_plot = st.selectbox("Choose signal to plot", ["Force_X", "Force_Y", "Force_Z", "Vibration_X", "Vibration_Y", "Vibration_Z", "AE_RMS"], key='frequency_signal_selection')

    def plot_frequency_analysis(data):
        y = np.asarray(data[signal_to_plot])
        n = len(y)
        yf = fft.fft(y)
        xf = fft.fftfreq(n, d=1/50000)
        freq_data = 2.0/n * np.abs(yf[0:n//2])
        freq_df = pd.DataFrame({'Frequency (Hz)': xf[0:n//2], 'Magnitude': freq_data})
        st.line_chart(freq_df.set_index('Frequency (Hz)'))

    st.write(f"**Data from: {selected_file}**")
    plot_frequency_analysis(chart_data)

# 3. Target Variable (Wear Flute) vs Cut (Interactive with Plotly)
with col3:
    st.title('Target Variable (Wear Flute) vs Cut')
    wear_data = load_wear_data(selected_chart)
    if wear_data is not None:
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
        st.plotly_chart(fig, use_container_width=True)

# 4. Correlation Matrix (Interactive with Plotly)
with col4:
    st.title('Correlation Matrix')
    correlation_selection = st.radio("Select CSV file", ("C1", "C4", "C6"), horizontal=True, key="correlation_radio")
    wear_data = load_wear_data(correlation_selection)
    main_data = load_main_data(correlation_selection)

    correlation_dropdown = st.selectbox('Select features to display correlation', 
                                         ['force_x', 'force_y', 'force_z', 'vibration_x', 'vibration_y', 'vibration_z'], 
                                         key='correlation_dropdown')

    if correlation_dropdown in main_data.columns:
        combined_data = pd.concat([main_data[[correlation_dropdown]], wear_data[['flute_1', 'flute_2', 'flute_3']]], axis=1)
        correlation_matrix = combined_data.corr()

        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='Viridis',
            colorbar=dict(title='Correlation')
        ))

        fig.update_layout(title='Correlation Matrix', xaxis_title='Variables', yaxis_title='Variables')
        st.plotly_chart(fig)
    else:
        st.write(f"Feature {correlation_dropdown} not found in main data.")