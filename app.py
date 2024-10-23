import streamlit as st
import pandas as pd
import os
import numpy as np
from scipy import fft
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.multioutput import MultiOutputRegressor

# Set Streamlit page configuration
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def train_lasso_select_fold(X, y, n_splits=5, alpha=0.1, max_iter=10000):
    if X.shape[0] != y.shape[0]:
        return None, None, None, None, None  # Early exit on shape mismatch

    tscv = TimeSeriesSplit(n_splits=n_splits)
    train_errors = []
    val_errors = []
    best_fold_index = -1
    best_val_mse = float("inf")

    lasso = MultiOutputRegressor(Lasso(alpha=alpha, max_iter=max_iter))
    scaler = StandardScaler()

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Normalize the data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        lasso.fit(X_train_scaled, y_train)

        # Predict
        y_pred_test = lasso.predict(X_test_scaled)

        # Calculate MSE to evaluate prediction performance
        mse_val = np.mean((y_test - y_pred_test) ** 2, axis=0).mean()
        mse_train = np.mean(
            (y_train - lasso.predict(X_train_scaled)) ** 2, axis=0
        ).mean()

        train_errors.append(mse_train)
        val_errors.append(mse_val)

        # Update best fold based on MSE
        if mse_val < best_val_mse:
            best_val_mse = mse_val
            best_fold_index = fold

    return lasso, scaler, train_errors, val_errors, best_fold_index


def plot_mse_st(train_errors, val_errors):
    # Create the figure
    fig = go.Figure()

    # Add the Training MSE line
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(train_errors) + 1)),
            y=train_errors,
            mode="lines+markers",
            name="Training MSE",
            marker=dict(symbol="circle"),
        )
    )

    # Add the Validation MSE line
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(val_errors) + 1)),
            y=val_errors,
            mode="lines+markers",
            name="Validation MSE",
            marker=dict(symbol="x"),
        )
    )

    # Update layout
    fig.update_layout(
        title="Mean Squared Error (MSE) per Fold",
        xaxis_title="Fold",
        yaxis_title="MSE",
        legend_title="Legend",
        height=400,  # Adjust height (you can also control width here)
        width=500,  # Set the width of the plot
        template="plotly_white",  # Optional: sets the plot style
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=False)


def plot_predictions_st(y_actual, y_pred, flute_labels=None):
    if flute_labels is None:
        flute_labels = [f"Flute {i+1}" for i in range(y_actual.shape[1])]

    for i in range(y_actual.shape[1]):
        st.write(f"### {flute_labels[i]}")

        # Create the figure
        fig = go.Figure()

        # Add the actual wear line
        fig.add_trace(
            go.Scatter(
                x=list(range(len(y_actual))),
                y=y_actual[:, i],
                mode="lines+markers",
                name="Actual wear",
                marker=dict(symbol="circle"),
                line=dict(width=2),
            )
        )

        # Add the predicted wear line
        fig.add_trace(
            go.Scatter(
                x=list(range(len(y_pred))),
                y=y_pred[:, i],
                mode="lines+markers",
                name="Predicted wear",
                marker=dict(symbol="x"),
                line=dict(width=2),
            )
        )

        # Update layout
        fig.update_layout(
            title=f"Wear Prediction for {flute_labels[i]}",
            xaxis_title="Test sample",
            yaxis_title="Wear Measurement",
            legend_title="Legend",
            height=400,  # Adjust as needed
            width=600,  # Adjust as needed
            template="plotly_white",
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=False)


# Function to load wear data based on selected chart
def load_wear_data(selected_chart):
    wear_file_path = os.path.join(
        folder_path, selected_chart.lower(), f"{selected_chart.lower()}_wear.csv"
    )
    try:
        wear_data = pd.read_csv(wear_file_path)
        return wear_data
    except Exception as e:
        st.error(f"Error loading wear data: {e} (Tried path: {wear_file_path})")
        return None


def load_stats_data(selected_chart):
    stats_file_path = os.path.join(
        folder_path,
        selected_chart.lower(),
        f"{selected_chart.lower()}_statistics.csv",
    )
    try:
        stats_data = pd.read_csv(stats_file_path, delimiter=",")
        return stats_data
    except Exception as e:
        st.error(f"Error loading stats data: {e} (Tried path: {stats_file_path})")
        return None


# Load main data for correlation matrix
def load_main_data(selected_chart):
    folder_path = f"./data/dashboard/{selected_chart.lower()}/{selected_chart.lower()}/"
    csv_files = [
        f for f in os.listdir(folder_path) if f.endswith(".csv") and "_freq" not in f
    ]

    if not csv_files:
        st.error("No valid data files found.")
        return None

    selected_file = csv_files[0]  # Use the first file found
    file_path = os.path.join(folder_path, selected_file)
    return pd.read_csv(file_path)


# Path to your CSV files folder
folder_path = "./data/dashboard/"


# Function to display "Exploratory Data Analysis" page
def show_eda():
    st.title("Exploratory Data Analysis")

    # Create 2x2 grid layout
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    # 1. Signal Visualization (Interactive with Altair)
    with col1:
        st.title("Signal Visualization")
        selected_chart = st.radio(
            "Select dataset",
            ("C1", "C4", "C6"),
            key="signal_chart_selection",
            horizontal=True,
        )
        folder_path_signals = (
            f"{folder_path}/{selected_chart.lower()}/{selected_chart.lower()}/"
        )

        csv_files = [
            f
            for f in os.listdir(folder_path_signals)
            if f.endswith(".csv") and "_freq" not in f
        ]
        selected_file = st.selectbox(
            "Select a cut file", csv_files, key="signal_file_selection"
        )

        def load_first_chart_data(file_name):
            file_path = os.path.join(folder_path_signals, file_name)
            data = pd.read_csv(file_path)
            return data

        first_chart_data = load_first_chart_data(selected_file)
        first_chart_data.columns = [
            "Force_X",
            "Force_Y",
            "Force_Z",
            "Vibration_X",
            "Vibration_Y",
            "Vibration_Z",
            "AE_RMS",
            "time",
        ]

        signal_to_plot = st.selectbox(
            "Choose signal to plot",
            [
                "Force_X",
                "Force_Y",
                "Force_Z",
                "Vibration_X",
                "Vibration_Y",
                "Vibration_Z",
                "AE_RMS",
            ],
            key="signal_selection",
        )
        st.write(f"**Data from: {selected_file}**")
        st.line_chart(first_chart_data[["time", signal_to_plot]].set_index("time"))

    # 2. Frequency Analysis (Interactive with Plotly)
    with col2:
        st.title("Frequency Analysis")
        selected_chart = st.radio(
            "Select dataset",
            ("C1", "C4", "C6"),
            key="frequency_chart_selection",
            horizontal=True,
        )
        folder_path_frequencies = (
            f"{folder_path}/{selected_chart.lower()}/{selected_chart.lower()}/"
        )

        csv_files = [
            f
            for f in os.listdir(folder_path_frequencies)
            if f.endswith(".csv") and "_freq" not in f
        ]
        selected_file = st.selectbox(
            "Select a cut file", csv_files, key="frequency_file_selection"
        )

        def load_data(file_name):
            file_path = os.path.join(folder_path_frequencies, file_name)
            data = pd.read_csv(file_path)
            return data

        chart_data = load_data(selected_file)
        chart_data.columns = [
            "Force_X",
            "Force_Y",
            "Force_Z",
            "Vibration_X",
            "Vibration_Y",
            "Vibration_Z",
            "AE_RMS",
            "time",
        ]

        signal_to_plot = st.selectbox(
            "Choose signal to plot",
            [
                "Force_X",
                "Force_Y",
                "Force_Z",
                "Vibration_X",
                "Vibration_Y",
                "Vibration_Z",
                "AE_RMS",
            ],
            key="frequency_signal_selection",
        )

        def plot_frequency_analysis(data):
            y = np.asarray(data[signal_to_plot])
            n = len(y)
            yf = fft.fft(y)
            xf = fft.fftfreq(n, d=1 / 50000)
            freq_data = 2.0 / n * np.abs(yf[0 : n // 2])
            freq_df = pd.DataFrame(
                {"Frequency (Hz)": xf[0 : n // 2], "Magnitude": freq_data}
            )
            st.line_chart(freq_df.set_index("Frequency (Hz)"))

        st.write(f"**Data from: {selected_file}**")
        plot_frequency_analysis(chart_data)

    # 3. Target Variable (Wear Flute) vs Cut (Interactive with Plotly)
    with col3:
        st.title("Target Variable (Wear Flute) vs Cut")
        selected_chart = st.radio(
            "Select dataset",
            ("C1", "C4", "C6"),
            key="wear   _chart_selection",
            horizontal=True,
        )
        wear_data = load_wear_data(selected_chart)
        if wear_data is not None:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=wear_data["cut"],
                    y=wear_data["flute_1"],
                    mode="lines+markers",
                    name="Flute 1",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=wear_data["cut"],
                    y=wear_data["flute_2"],
                    mode="lines+markers",
                    name="Flute 2",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=wear_data["cut"],
                    y=wear_data["flute_3"],
                    mode="lines+markers",
                    name="Flute 3",
                )
            )

            fig.update_layout(
                title="Target Variable (Wear Flute) vs Cut",
                xaxis_title="Cut",
                yaxis_title="Wear [μm]",
                legend_title="Flutes",
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

    # 4. Correlation Matrix (Interactive with Plotly)
    with col4:
        st.title("Correlation Matrix")
        correlation_selection = st.radio(
            "Select dataset",
            ("C1", "C4", "C6"),
            horizontal=True,
            key="correlation_radio",
        )
        wear_data = load_wear_data(correlation_selection)
        main_data = load_main_data(correlation_selection)
        stats_data = load_stats_data(correlation_selection)

        correlation_dropdown = st.selectbox(
            "Select features to display correlation",
            [
                "force_x",
                "force_y",
                "force_z",
                "vibration_x",
                "vibration_y",
                "vibration_z",
                "ae_rms",
            ],
            key="correlation_dropdown",
        )

        if correlation_dropdown in main_data.columns:
            selected_columns = [
                column
                for column in stats_data.columns
                if correlation_dropdown in column
            ]
            combined_data = pd.concat(
                [
                    stats_data[selected_columns],
                    wear_data[["flute_1", "flute_2", "flute_3"]],
                ],
                axis=1,
            )
            correlation_matrix = combined_data.corr()

            fig = go.Figure(
                data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns.drop(selected_columns),
                    y=correlation_matrix.columns.drop(
                        ["flute_1", "flute_2", "flute_3"]
                    ),
                    colorscale="Viridis",
                    colorbar=dict(title="Correlation"),
                    zmin=-1,
                    zmax=1,
                )
            )

            fig.update_layout(
                title="Correlation Matrix",
                xaxis_title="Variables",
                yaxis_title="Variables",
            )
            st.plotly_chart(fig)
        else:
            st.write(f"Feature {correlation_dropdown} not found in main data.")


# Function to display "Machine Learning" page
def show_ml():
    st.title("Machine Learning")
    st.write("### About the Model:")

    def load_main_data(selected_chart):
        main_file_path = os.path.join(
            folder_path,
            selected_chart.lower(),
            f"{selected_chart.lower()}_statistics.csv",
        )
        try:
            main_data = pd.read_csv(main_file_path)
            return main_data
        except Exception as e:
            st.error(f"Error loading main data: {e} (Tried path: {main_file_path})")
            return None

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Features Used:")
        st.markdown(
            """ 
            - Force statistics (X, Y and Z)
            - Vibration statistics (X, Y and Z)
            - AE_RMS statistics
        """
        )
        st.write("Statistics calculated: minimums, maximums, mean value,")
        st.write("std, zero crossings, kurtosis, skewness, energy")

    with col2:
        st.write("#### Lasso Linear Regression Model:")
        st.markdown(
            """
            - A linear model that estimates the coefficients of linear relationships.
            - It uses L1 regularization to prevent overfitting.
            - Suitable for high-dimensional datasets.
            - Can perform variable selection.
        """
        )
        st.write("Wrapped in a Multi-output regression model.")
        st.write("This allows for having multiple targets (3 flutes) at the same time.")

    # Dropdown selector for model selection
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(" ")
    with col2:
        selected_model = st.selectbox("Select a model:", ("c1", "c4", "c6"))
    with col1:
        st.write(" ")

    # Load data based on selected model
    st.write(f"## {selected_model.upper()} Dataset Analysis")

    wear_data = load_wear_data(selected_model)
    main_data = load_main_data(selected_model)

    if wear_data is not None and main_data is not None:
        X = main_data.values
        y = wear_data[["flute_1", "flute_2", "flute_3"]].values

        # Train model and display results
        best_lasso, scaler, train_errors, val_errors, best_fold_index = (
            train_lasso_select_fold(X, y)
        )

        if best_lasso is not None:  # Check if model training was successful
            st.write(
                f"### Best fold selected: {best_fold_index + 1} with validation MSE: {min(val_errors)}"
            )
            left_co, cent_co, last_co = st.columns(3)

            with cent_co:
                plot_mse_st(train_errors, val_errors)

            # Make predictions
            y_pred = best_lasso.predict(scaler.transform(X))

            # Plot predictions for selected model data
            pred_col1, pred_col2, pred_col3 = st.columns(3)

            with pred_col1:
                plot_predictions_st(y[:, 0:1], y_pred[:, 0:1], flute_labels=["Flute 1"])

            with pred_col2:
                plot_predictions_st(y[:, 1:2], y_pred[:, 1:2], flute_labels=["Flute 2"])

            with pred_col3:
                plot_predictions_st(y[:, 2:3], y_pred[:, 2:3], flute_labels=["Flute 3"])

            # Prepare data for validation results
            validation_data_mapping = {"c1": "c6", "c4": "c1", "c6": "c4"}

            validation_selected_model = validation_data_mapping[selected_model]
            X_val, y_val = (
                load_main_data(validation_selected_model).values,
                load_wear_data(validation_selected_model)[
                    ["flute_1", "flute_2", "flute_3"]
                ].values,
            )

            # Train model on validation data
            (
                best_lasso_val,
                scaler_val,
                train_errors_val,
                val_errors_val,
                best_fold_index_val,
            ) = train_lasso_select_fold(X_val, y_val)

            # Display best fold for validation data
            st.write(
                f"### Validation Results on {validation_selected_model.upper()} Dataset"
            )
            st.write(
                f"### Best fold selected: {best_fold_index_val + 1} with validation MSE: {min(val_errors_val)}"
            )

            # Plot validation predictions
            val_pred_col1, val_pred_col2, val_pred_col3 = st.columns(3)

            # Make predictions on validation data
            val_y_pred = best_lasso_val.predict(scaler_val.transform(X_val))

            with val_pred_col1:
                plot_predictions_st(
                    y_val[:, 0:1], val_y_pred[:, 0:1], flute_labels=["Flute 1"]
                )

            with val_pred_col2:
                plot_predictions_st(
                    y_val[:, 1:2], val_y_pred[:, 1:2], flute_labels=["Flute 2"]
                )

            with val_pred_col3:
                plot_predictions_st(
                    y_val[:, 2:3], val_y_pred[:, 2:3], flute_labels=["Flute 3"]
                )
        else:
            st.error("Model training failed due to data shape mismatch.")
    else:
        st.error("Failed to load the necessary data for the selected model.")


# Function to display "Deep Learning" page
def show_dl():
    st.title("Deep Learning")

    # Load image
    image = Image.open("./CNN_methods.drawio.png")

    left_co, cent_co, last_co = st.columns(3)

    with left_co:
        st.markdown("#")

        st.write(
            "Instead of defining features manually, convolutional neural networks can do the work for us."
            + " They process the raw data,creating and optimizing features for prediction."
        )
        st.write(
            "The model we chose consists of: two 1D convolutional layers,"
            + "a maxpool layer, a 1D convolutional layer, a maxpool layer, a droput layer"
            + " and a fully connected layer. The kernel size was set to 3, the amount of filters to 128 and RELU"
            " activation functions were used after each convolutional layer."
        )
        st.write(
            "Training was done using dataset C1, in batches of 5 cuts and for 500 epochs, at a learning rate of 0.002."
        )

    with cent_co:
        # Display image in Streamlit app
        st.image(image, caption="1D Convolutional Neural Network", width=800)

    # --------------------------------------------------------------------------------

    # Training performance

    dl_df = pd.read_csv("data/dashboard/DL_C1.csv")

    MSE_w1 = np.mean((dl_df["c1w1"] - dl_df["c1w1p"]) ** 2)
    MSE_w2 = np.mean((dl_df["c1w2"] - dl_df["c1w2p"]) ** 2)
    MSE_w3 = np.mean((dl_df["c1w3"] - dl_df["c1w3p"]) ** 2)

    fig = make_subplots(
        rows=1, cols=3, subplot_titles=("Flute 1", "Flute 2", "Flute 3")
    )

    # Sample data for the three charts
    x = list(range(1, 316))
    y1_a = dl_df["c1w1"]
    y1_b = dl_df["c1w1p"]
    y2_a = dl_df["c1w2"]
    y2_b = dl_df["c1w2p"]
    y3_a = dl_df["c1w3"]
    y3_b = dl_df["c1w3p"]

    # Add first chart (line + marker plot with two data sets)
    fig.add_trace(
        go.Scatter(x=x, y=y1_a, mode="lines+markers", name="Real flute 1 wear"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=y1_b, mode="lines+markers", name="Predicted flute 1 wear"),
        row=1,
        col=1,
    )

    # Add second chart (line + marker plot with two data sets)
    fig.add_trace(
        go.Scatter(x=x, y=y2_a, mode="lines+markers", name="Real flute 2 wear"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=x, y=y2_b, mode="lines+markers", name="Predicted flute 2 wear"),
        row=1,
        col=2,
    )

    # Add third chart (line + marker plot with two data sets)
    fig.add_trace(
        go.Scatter(x=x, y=y3_a, mode="lines+markers", name="Real flute 3 wear"),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(x=x, y=y3_b, mode="lines+markers", name="Real flute 3 wear"),
        row=1,
        col=3,
    )

    # Update layout to add numerical boxes as annotations to each plot
    annotations = [
        dict(
            x=0.18,
            y=0.2,
            xref="paper",
            yref="paper",
            text="MSE:" + str(MSE_w1),
            showarrow=False,
            font=dict(size=15),
            bgcolor=None,
            bordercolor="black",
        ),
        dict(
            x=0.59,
            y=0.2,
            xref="paper",
            yref="paper",
            text="MSE:" + str(MSE_w2),
            showarrow=False,
            font=dict(size=15),
            bgcolor=None,
            bordercolor="black",
        ),
        dict(
            x=0.99,
            y=0.2,
            xref="paper",
            yref="paper",
            text="MSE:" + str(MSE_w3),
            showarrow=False,
            font=dict(size=15),
            bgcolor=None,
            bordercolor="black",
        ),
    ]

    # Update layout to span the full width
    fig.update_layout(
        height=500,  # Adjust height if needed
        width=None,  # Set width to None for auto width
        title_text="Performance of the model on training dataset (C1)",
        showlegend=True,
        annotations=annotations,
    )

    # Add x-axis and y-axis labels to each subplot
    fig.update_xaxes(title_text="Cut number", row=1, col=1)
    fig.update_xaxes(title_text="Cut number", row=1, col=2)
    fig.update_xaxes(title_text="Cut number", row=1, col=3)

    fig.update_yaxes(title_text="wear (um)", row=1, col=1)
    fig.update_yaxes(title_text="wear (um)", row=1, col=2)
    fig.update_yaxes(title_text="wear (um)", row=1, col=3)

    # Display the plot in the Streamlit app, using the full width of the page
    st.plotly_chart(fig, use_container_width=True, key="c1")

    # -----------------------------------------------------------------

    dl_df = pd.read_csv("data/dashboard/DL_C1.csv")

    MSE_w1 = np.mean((dl_df["c4w1"] - dl_df["c4w1p"]) ** 2)
    MSE_w2 = np.mean((dl_df["c4w2"] - dl_df["c4w2p"]) ** 2)
    MSE_w3 = np.mean((dl_df["c4w3"] - dl_df["c4w3p"]) ** 2)

    fig = make_subplots(rows=1, cols=3)

    # Sample data for the three charts
    x = list(range(1, 316))
    y1_a = dl_df["c4w1"]
    y1_b = dl_df["c4w1p"]
    y2_a = dl_df["c4w2"]
    y2_b = dl_df["c4w2p"]
    y3_a = dl_df["c4w3"]
    y3_b = dl_df["c4w3p"]

    # Add first chart (line + marker plot with two data sets)
    fig.add_trace(
        go.Scatter(x=x, y=y1_a, mode="lines+markers", name="Real flute 1 wear"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=y1_b, mode="lines+markers", name="Predicted flute 1 wear"),
        row=1,
        col=1,
    )

    # Add second chart (line + marker plot with two data sets)
    fig.add_trace(
        go.Scatter(x=x, y=y2_a, mode="lines+markers", name="Real flute 2 wear"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=x, y=y2_b, mode="lines+markers", name="Predicted flute 2 wear"),
        row=1,
        col=2,
    )

    # Add third chart (line + marker plot with two data sets)
    fig.add_trace(
        go.Scatter(x=x, y=y3_a, mode="lines+markers", name="Real flute 3 wear"),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(x=x, y=y3_b, mode="lines+markers", name="Real flute 3 wear"),
        row=1,
        col=3,
    )

    # Update layout to add numerical boxes as annotations to each plot
    annotations = [
        dict(
            x=0.18,
            y=0.2,
            xref="paper",
            yref="paper",
            text="MSE:" + str(MSE_w1),
            showarrow=False,
            font=dict(size=15),
            bgcolor=None,
            bordercolor="black",
        ),
        dict(
            x=0.59,
            y=0.2,
            xref="paper",
            yref="paper",
            text="MSE:" + str(MSE_w2),
            showarrow=False,
            font=dict(size=15),
            bgcolor=None,
            bordercolor="black",
        ),
        dict(
            x=0.99,
            y=0.2,
            xref="paper",
            yref="paper",
            text="MSE:" + str(MSE_w3),
            showarrow=False,
            font=dict(size=15),
            bgcolor=None,
            bordercolor="black",
        ),
    ]

    # Update layout to span the full width
    fig.update_layout(
        height=500,  # Adjust height if needed
        width=None,  # Set width to None for auto width
        title_text="Performance of the model on test dataset (C4)",
        showlegend=True,
        annotations=annotations,
    )

    # Display the plot in the Streamlit app, using the full width of the page
    st.plotly_chart(fig, use_container_width=True, key="c4")

    # -----------------------------------------------------------------

    dl_df = pd.read_csv("data/dashboard/DL_C1.csv")

    MSE_w1 = np.mean((dl_df["c6w1"] - dl_df["c6w1p"]) ** 2)
    MSE_w2 = np.mean((dl_df["c6w2"] - dl_df["c6w2p"]) ** 2)
    MSE_w3 = np.mean((dl_df["c6w3"] - dl_df["c6w3p"]) ** 2)

    fig = make_subplots(rows=1, cols=3)

    # Sample data for the three charts
    x = list(range(1, 316))
    y1_a = dl_df["c6w1"]
    y1_b = dl_df["c6w1p"]
    y2_a = dl_df["c6w2"]
    y2_b = dl_df["c6w2p"]
    y3_a = dl_df["c6w3"]
    y3_b = dl_df["c6w3p"]

    # Add first chart (line + marker plot with two data sets)
    fig.add_trace(
        go.Scatter(x=x, y=y1_a, mode="lines+markers", name="Real flute 1 wear"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=y1_b, mode="lines+markers", name="Predicted flute 1 wear"),
        row=1,
        col=1,
    )

    # Add second chart (line + marker plot with two data sets)
    fig.add_trace(
        go.Scatter(x=x, y=y2_a, mode="lines+markers", name="Real flute 2 wear"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=x, y=y2_b, mode="lines+markers", name="Predicted flute 2 wear"),
        row=1,
        col=2,
    )

    # Add third chart (line + marker plot with two data sets)
    fig.add_trace(
        go.Scatter(x=x, y=y3_a, mode="lines+markers", name="Real flute 3 wear"),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(x=x, y=y3_b, mode="lines+markers", name="Real flute 3 wear"),
        row=1,
        col=3,
    )

    # Update layout to add numerical boxes as annotations to each plot
    annotations = [
        dict(
            x=0.18,
            y=0.2,
            xref="paper",
            yref="paper",
            text="MSE:" + str(MSE_w1),
            showarrow=False,
            font=dict(size=15),
            bgcolor=None,
            bordercolor="black",
        ),
        dict(
            x=0.59,
            y=0.2,
            xref="paper",
            yref="paper",
            text="MSE:" + str(MSE_w2),
            showarrow=False,
            font=dict(size=15),
            bgcolor=None,
            bordercolor="black",
        ),
        dict(
            x=0.99,
            y=0.2,
            xref="paper",
            yref="paper",
            text="MSE:" + str(MSE_w3),
            showarrow=False,
            font=dict(size=15),
            bgcolor=None,
            bordercolor="black",
        ),
    ]

    # Update layout to span the full width
    fig.update_layout(
        height=500,  # Adjust height if needed
        width=None,  # Set width to None for auto width
        title_text="Performance of the model on test dataset (C6)",
        showlegend=True,
        annotations=annotations,
    )

    # Display the plot in the Streamlit app, using the full width of the page
    st.plotly_chart(fig, use_container_width=True, key="c6")

    # -----------------------------------------------------------------


# Function to display different pages
def show_page(page):
    if page == "Exploratory Data Analysis":
        show_eda()  # Call the function for the "Exploratory Data Analysis" page
    elif page == "Machine Learning":
        show_ml()  # Call the function for the "Machine Learning" page
    elif page == "Deep Learning":
        show_dl()  # Call the function for the "Deep Learning" page


# Initialize session state for navigation if it doesn't exist
if "page" not in st.session_state:
    st.session_state.page = "Exploratory Data Analysis"

# Sidebar with buttons
with st.sidebar:
    st.title("Predictive Maintenance")
    st.markdown("---")
    if st.button("Exploratory Data Analysis"):
        st.session_state.page = "Exploratory Data Analysis"
    if st.button("Machine Learning"):
        st.session_state.page = "Machine Learning"
    if st.button("Deep Learning"):
        st.session_state.page = "Deep Learning"


show_page(st.session_state.page)
