# Import Libraries
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import seaborn as sns
from scipy.stats import skew, kurtosis
from warnings import simplefilter

# Silences pandas warning that ruin the display of the notebook on GitHub
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Function to calculate zero crossings
def calculate_zero_crossings(signal):
    return ((np.diff(np.sign(signal)) != 0).sum())

# Function to calculate energy
def calculate_energy(signal):
    return np.sum(signal ** 2)

# Function to calculate statistics for each column in a DataFrame
def calculate_statistics(df):
    stats = pd.DataFrame()
    for column in df.columns:
        col_data = df[column]
        stats[column + "_min"] = [col_data.min()]
        stats[column + "_max"] = [col_data.max()]
        stats[column + "_mean"] = [col_data.mean()]
        stats[column + "_std"] = [col_data.std()]
        stats[column + "_skew"] = [skew(col_data)]
        stats[column + "_kurtosis"] = [kurtosis(col_data)]
        stats[column + "_energy"] = [calculate_energy(col_data)]
        stats[column + "_zero_crossings"] = [calculate_zero_crossings(col_data)]
    return stats

# Collect statistics from CSV files
folder_path = '/Users/dermotlyons/Downloads/BearIT/Project/c1/c1'
stats_list = []
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path) and filename.endswith('.csv'):
        df = pd.read_csv(file_path)
        stats_list.append(calculate_statistics(df))

# Concatenate all statistics into a single DataFrame
c1_stats = pd.concat(stats_list, ignore_index=True)

# Streamlit page config
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

alt.themes.enable("dark")

# Load Data
df = pd.read_csv("c_1_001.csv")
df.columns = ["Force_X", "Force_Y", "Force_Z", "Vibration_X", "Vibration_Y", "Vibration_Z", "AE_RMS"]
n = df["Force_X"].shape[0]  # number of measurements
t = 0.02 * np.arange(n)  # time in milliseconds
df["time"] = t

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

# Plots
##### 1: Altair Plot #####
plot_data = df[["time", "Force_X"]][:1500]
chart = (
    alt.Chart(plot_data)
    .mark_line(point=True)
    .encode(
        x=alt.X("time:Q", title="Time"),
        y=alt.Y("Force_X:Q", title="Force (X)"),
        tooltip=[alt.Tooltip("time:Q", title="Time"), alt.Tooltip("Force_X:Q", title="Force (X)")],
    )
    .properties(title="Force X over Time", width=800, height=400)
)
st.altair_chart(chart, use_container_width=True)

##### 2: Plotly Plot #####
st.title('Acoustic Emission vs. Time')
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['time'][:1500], y=df['AE_RMS'][:1500], mode='lines', name='AE-RMS', line=dict(color='cyan', width=2)))
fig.update_layout(title='Acoustic Emission vs. Time', xaxis_title='Time [ms]', yaxis_title='Acoustic Emission [V]', template='plotly_white')
st.plotly_chart(fig)

##### 3: Moving Average Plot #####
window_size = 100
df['moving_average'] = df['Force_Z'].rolling(window=window_size).mean()

st.title('Force and Moving Average in Z Dimension vs. Time')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['time'][:1500], df['Force_Z'][:1500], label='Force Z', color='g', alpha=0.5)
ax.plot(df['time'][:1500], df['moving_average'][:1500], label='Moving Average', color='r')
ax.set_title('Force and Moving Average in Z Dimension vs. Time')
ax.set_xlabel('Time [$ms$]')
ax.set_ylabel('Force [$N$]')
ax.grid(True)
ax.legend()

##### 4: Interactive Plotly Plot #####
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['time'][:1500], y=df['Force_Z'][:1500], mode='lines', name='Force Z', line=dict(color='green', width=2), opacity=0.5))
fig.add_trace(go.Scatter(x=df['time'][:1500], y=df['moving_average'][:1500], mode='lines', name='Moving Average', line=dict(color='red', width=2)))
fig.update_layout(title='Force and Moving Average in Z Dimension vs. Time', xaxis_title='Time [ms]', yaxis_title='Force [N]', template='plotly_white')
st.plotly_chart(fig)

#######################
# Statistics and Correlation
c1_wear_data = pd.read_csv('c1_wear.csv', sep=',')
c1_wear_data = c1_wear_data.drop(columns=['cut'])

c1_combined_data = pd.concat([c1_stats, c1_wear_data], axis=1)
c1_combined_corr_matrix = c1_combined_data.corr()

c1_wear_corr_matrix = c1_combined_corr_matrix[['flute_1', 'flute_2', 'flute_3']].drop(['flute_1', 'flute_2', 'flute_3'], axis=0)

min_columns = [
    'Force_X_min', 
    'Force_Y_min', 
    'Force_Z_min', 
    'Vibration_X_min', 
    'Vibration_Y_min', 
    'Vibration_Z_min', 
    'AE_RMS_min'
]
min_stats = c1_stats[min_columns]
combined_data_min = pd.concat([min_stats.reset_index(drop=True), c1_wear_data.reset_index(drop=True)], axis=1)
corr_matrix_min = combined_data_min.corr()
wear_columns = ['flute_1', 'flute_2', 'flute_3']
min_correlation_matrix = corr_matrix_min.loc[min_columns, wear_columns]
plt.figure(figsize=(5, 4))
plt.title('Correlations Between Wear Data and Minimum Statistics', pad=20)
plt.xlabel('Wear Data', labelpad=20)
plt.imshow(min_correlation_matrix, cmap='cool', aspect='auto')

# Annotate with numbers
for (i, j), val in np.ndenumerate(min_correlation_matrix):
    plt.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')

plt.xticks(range(len(wear_columns)), wear_columns)
plt.yticks(range(len(min_columns)), min_columns)
plt.colorbar(label='')
plt.tight_layout()
plt.show()