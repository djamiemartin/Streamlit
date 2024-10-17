#######################
# import libraries

import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import seaborn as sns
from scipy.stats import skew, kurtosis


# Silences pandas warning that ruin the display of the notebook on github
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

#######################
# Page configuration

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",  # this will show in the browser tab
    page_icon=":bar_chart:",  # the icon showing before the title
    layout="wide",
    initial_sidebar_state="expanded",
)

alt.themes.enable("dark")

#######################
# CSS styling

st.markdown(
    """
<style>

</style>
""",
    unsafe_allow_html=True,
)

#######################
# Load data

df = pd.read_csv("c_1_001.csv")

df.columns = [
    "Force_X",
    "Force_Y",
    "Force_Z",
    "Vibration_X",
    "Vibration_Y",
    "Vibration_Z",
    "AE_RMS",
]

n = df["Force_X"].shape[0]  # number of measurements
t = 0.02 * np.arange(n)  # time in miliseconds

df["time"] = t


#######################
# Sidebar

with st.sidebar:
    st.title("Predictive Maintenance")


#######################
# Plots

##### 1: Plotting with matplotlib and then display with streamlit #####

plt.figure(figsize=(12, 6))
plt.plot(df["time"][:1500], df["Force_X"][:1500], label="Force X", color="b")
plt.xlabel("Time [$ms$]")
plt.ylabel("Force [$N$]")
plt.grid()
plt.legend()

st.pyplot(plt)


##### Plotting natively with streamlit #####

time_data = df["time"][:1500]
force_x_data = df["Force_X"][:1500]

# Create a DataFrame for easier plotting
plot_data = pd.DataFrame({"Time (ms)": time_data, "Force (N)": force_x_data})


##### 2: Plot using Streamlit's native plotting #####

st.line_chart(plot_data.set_index("Time (ms)"))

# You can add additional information if needed
st.write("Force X plotted against Time.")


##### 3: Plotting with altair and then display with streamlit #####

# Select the data you want to plot
plot_data = df[["time", "Force_X"]][:1500]  # Use the first 1500 rows

# Create the Altair
chart = (
    alt.Chart(plot_data)
    .mark_line(point=True)  # Add points to make hovering easier
    .encode(
        x=alt.X("time:Q", title="Time"),  # 'Q' is for quantitative data
        y=alt.Y("Force_X:Q", title="Force (X)"),  # 'Q' is for quantitative data
        tooltip=[alt.Tooltip("time:Q", title="Time"), alt.Tooltip("Force_X:Q", title="Force (X)")]
    )
    .properties(title="Force X over Time", width=800, height=400)
)

# Display the chart in Streamlit
st.altair_chart(chart, use_container_width=True)


# Streamlit title
st.title('Acoustic Emission vs. Time')

# Create the plot using matplotlib
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['time'][:1500], df['AE_RMS'][:1500], label='AE-RMS', color='c')
ax.set_title('Acoustic Emission vs. Time')
ax.set_xlabel('Time [$ms$]')
ax.set_ylabel('Acoustic Emission [$V$]')
ax.grid(True)
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)

st.title('Interactive: Acoustic Emission vs. Time')

# Create the interactive Plotly chart
fig = go.Figure()

# Add AE_RMS line
fig.add_trace(go.Scatter(
    x=df['time'][:1500],
    y=df['AE_RMS'][:1500],
    mode='lines',
    name='AE-RMS',
    line=dict(color='cyan', width=2)
))

# Customize the chart layout
fig.update_layout(
    title='Acoustic Emission vs. Time',
    xaxis_title='Time [ms]',
    yaxis_title='Acoustic Emission [V]',
    template='plotly_white',
    legend=dict(x=0.1, y=1.1)
)

# Display the interactive Plotly chart in Streamlit
st.plotly_chart(fig)

# Window size for moving average
window_size = 100

# Calculate the moving average for the 'Force_Z' column
df['moving_average'] = df['Force_Z'].rolling(window=window_size).mean()

# Streamlit title
st.title('Force and Moving Average in Z Dimension vs. Time')

# Matplotlib plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['time'][:1500], df['Force_Z'][:1500], label='Force Z', color='g', alpha=0.5)
ax.plot(df['time'][:1500], df['moving_average'][:1500], label='Moving Average', color='r')
ax.set_title('Force and Moving Average in Z Dimension vs. Time')
ax.set_xlabel('Time [$ms$]')
ax.set_ylabel('Force [$N$]')
ax.grid(True)
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)

window_size = 100

# Calculate the moving average for the 'Force_Z' column
df['moving_average'] = df['Force_Z'].rolling(window=window_size).mean()

# Streamlit title
st.title('Interactive: Force and Moving Average in Z Dimension vs. Time')

# Create the interactive Plotly chart
fig = go.Figure()

# Add Force_Z line
fig.add_trace(go.Scatter(
    x=df['time'][:1500],
    y=df['Force_Z'][:1500],
    mode='lines',
    name='Force Z',
    line=dict(color='green', width=2),
    opacity=0.5
))

# Add Moving Average line
fig.add_trace(go.Scatter(
    x=df['time'][:1500],
    y=df['moving_average'][:1500],
    mode='lines',
    name='Moving Average',
    line=dict(color='red', width=2)
))

# Customize the chart layout
fig.update_layout(
    title='Force and Moving Average in Z Dimension vs. Time',
    xaxis_title='Time [ms]',
    yaxis_title='Force [N]',
    template='plotly_white',
    legend=dict(x=0.1, y=1.1)
)

# Display the Plotly chart in Streamlit
st.plotly_chart(fig)
# Function to calculate zero crossings
def calculate_zero_crossings(signal):
    return ((np.diff(np.sign(signal)) != 0).sum())

# Function to calculate energy
def calculate_energy(signal):
    return np.sum(signal ** 2)

# Function to calculate statistics
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

# Streamlit app
def main():
    st.title("Statistics Calculator for CSV Files")
    
    df = pd.read_csv(c_1_001)
    st.write("Data Preview:")
    st.dataframe(df.head())  # Show the first few rows of the DataFrame
        
    # Calculate and display statistics
    stats = calculate_statistics(df)
    st.write("Statistics:")
    st.dataframe(stats)  # Show the calculated statistics
        
    # Option to download the statistics as CSV
    csv = stats.to_csv(index=False)
    st.download_button(
            label="Download Statistics as CSV",
            data=csv,
            file_name='statistics.csv',
            mime='text/csv',
        )
    c1_corr_matrix = c1_stats.corr()
