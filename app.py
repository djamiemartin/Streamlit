#######################
# import libraries

import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


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

df = pd.read_csv("data/raw/c1/c1/c_1_001.csv")

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
# Plots

def display_starter_chart():

    time_data = df["time"][:1500]
    force_x_data = df["Force_X"][:1500]

    # Create a DataFrame for easier plotting
    st.title('Interactive: Force X vs Time')

    plot_data = pd.DataFrame({"Time (ms)": time_data, "Force (N)": force_x_data})

    st.line_chart(plot_data.set_index("Time (ms)"))

    # You can add additional information if needed
    st.write("Force X plotted against Time.")


##### 1: Plotting natively with streamlit #####

def display_chart_1():
    st.title('Interactive: Force Y vs Time')
    
    time_data = df["time"][:1500]
    force_y_data = df["Force_Y"][:1500]

    # Create a DataFrame for easier plotting
    plot_data = pd.DataFrame({"Time (ms)": time_data, "Force (N)": force_y_data})

    st.line_chart(plot_data.set_index("Time (ms)"))

    # You can add additional information if needed
    st.write("Force Y plotted against Time.")


##### 2: Plotting with altair and then display with streamlit #####
def display_chart_2():
    st.title('Interactive: Force Z vs Time')
    # Select the data you want to plot
    plot_data = df[["time", "Force_Z"]][:1500]  # Use the first 1500 rows

    # Create the Altair
    chart = (
        alt.Chart(plot_data)
        .mark_line(point=True)  # Add points to make hovering easier
        .encode(
            x=alt.X("time:Q", title="Time"),  # 'Q' is for quantitative data
            y=alt.Y("Force_Z:Q", title="Force (Z)"),  # 'Q' is for quantitative data
            tooltip=[alt.Tooltip("time:Q", title="Time"), alt.Tooltip("Force_Z:Q", title="Force (Z)")]
        )
        .properties(title="Force Z over Time", width=800, height=400)
    )

    # Display the chart in Streamlit
    st.altair_chart(chart, use_container_width=True)


##### 4: Interavtive AE chart ####
def display_chart_3():
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

##### 4: Interavtive Rolling chart ####
def display_chart_4():
    # Window size for moving average
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


#######################
# Sidebar

st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Go to",
    ('Starter Chart', 'Chart 1', 'Chart 2', 'Chart 3', 'Chart 4')
)

# Display content based on selection
if option == 'Starter Chart':
    st.title("Starter Chart")
    display_starter_chart()
elif option == 'Chart 1':
    st.title("Force Y vs Time")
    display_chart_1()
elif option == 'Chart 2':
    st.title("Force Z vs Time")
    display_chart_2()
elif option == 'Chart 3':
    st.title("Acoustic Emission vs. Time")
    display_chart_3()
elif option == 'Chart 4':
    st.title("Force and Moving Average in Z Dimension vs. Time")
    display_chart_4()