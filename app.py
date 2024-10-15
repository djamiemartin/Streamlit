#######################
# import libraries

import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import altair


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
    .mark_line()
    .encode(
        x="time",  # Automatically inferred as quantitative
        y="Force_X",  # Automatically inferred as quantitative
        tooltip=["time", "Force_X"],  # Tooltip information
    )
    .properties(title="Force X over Time", width=800, height=400)
)

# Display the chart in Streamlit
st.altair_chart(chart, use_container_width=True)
