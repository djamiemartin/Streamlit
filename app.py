import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(page_title="Predictive Maintenance Dashboard", page_icon=":bar_chart:", layout="wide", initial_sidebar_state="expanded")

# Path to your CSV files folder
folder_path = './data/dashboard'

# Load wear data
def load_wear_data(selected_chart):
    wear_file_path = os.path.join(folder_path, selected_chart.lower(), f"{selected_chart.lower()}_wear.csv")
    try:
        wear_data = pd.read_csv(wear_file_path)
        return wear_data
    except Exception as e:
        st.error(f"Error loading wear data: {e} (Tried path: {wear_file_path})")
        return None

# Load statistics data
def load_statistics_data(selected_chart):
    stats_file_path = os.path.join(folder_path, selected_chart.lower(), f"{selected_chart.lower()}_statistics.csv")
    try:
        stats_data = pd.read_csv(stats_file_path)
        return stats_data
    except Exception as e:
        st.error(f"Error loading statistics data: {e} (Tried path: {stats_file_path})")
        return None

# Load main data for correlation matrix
def load_main_data(selected_chart):
    main_file_path = os.path.join(folder_path, selected_chart.lower(), f"{selected_chart.lower()}_statistics.csv")
    try:
        main_data = pd.read_csv(main_file_path)
        return main_data
    except Exception as e:
        st.error(f"Error loading main data: {e} (Tried path: {main_file_path})")
        return None

def train_lasso_select_fold(X, y, n_splits=5, alpha=0.1, max_iter=10000):
    if X.shape[0] != y.shape[0]:
        return None, None, None, None, None  # Early exit on shape mismatch

    tscv = TimeSeriesSplit(n_splits=n_splits)
    train_errors = []
    val_errors = []
    best_fold_index = -1
    best_val_mse = float('inf')

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
        mse_train = np.mean((y_train - lasso.predict(X_train_scaled)) ** 2, axis=0).mean()

        train_errors.append(mse_train)
        val_errors.append(mse_val)

        # Update best fold based on MSE
        if mse_val < best_val_mse:
            best_val_mse = mse_val
            best_fold_index = fold

    return lasso, scaler, train_errors, val_errors, best_fold_index

# Plot MSE during cross-validation using Streamlit
def plot_mse_st(train_errors, val_errors):
    fig, ax = plt.subplots()
    ax.plot(range(1, len(train_errors) + 1), train_errors, label='Training MSE', marker='o')
    ax.plot(range(1, len(val_errors) + 1), val_errors, label='Validation MSE', marker='x')
    ax.set_xlabel('Fold')
    ax.set_ylabel('MSE')
    ax.set_title('Mean Squared Error (MSE) per Fold')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# Plot predictions using Streamlit
def plot_predictions_st(y_actual, y_pred, flute_labels=None):
    if flute_labels is None:
        flute_labels = [f'Flute {i+1}' for i in range(y_actual.shape[1])]

    for i in range(y_actual.shape[1]):
        st.write(f"### {flute_labels[i]}")
        fig, ax = plt.subplots()
        ax.plot(range(len(y_actual)), y_actual[:, i], label='Actual wear', marker='o', linewidth=2.0)
        ax.plot(range(len(y_pred)), y_pred[:, i], label='Predicted wear', marker='x', linewidth=2.0)
        ax.set_xlabel('Test sample')
        ax.set_ylabel('Wear Measurement')
        ax.legend()
        st.pyplot(fig)

# Initialize session state for buttons
if 'show_machine_learning' not in st.session_state:
    st.session_state.show_machine_learning = False

# Sidebar with buttons
with st.sidebar:
    st.title("Predictive Maintenance")
    st.markdown("---")
    if st.button("Problem and Data"):
        st.write("Displaying Problem and Data")
    if st.button("Exploratory Data Analysis"):
        st.write("Displaying Exploratory Data Analysis")
    
    if st.button("Machine Learning"):
        st.session_state.show_machine_learning = True  # Set the flag to True

    if st.button("Deep Learning"):
        st.write("Displaying Deep Learning")

# Display content in the main area
if st.session_state.show_machine_learning:
    # Display title and description
    st.write("## Machine Learning Modeling")
    st.write("### About the Model:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Features Used:")
        st.markdown(""" 
            - Force statistics (X, Y and Z)
            - Vibration statistics (X, Y and Z)
            - AE_RMS statistics
        """)
        st.write("Statistics calculated: minimums, maximums, mean value,")
        st.write("std, zero crossings, kurtosis, skewness, energy")

    with col2:
        st.write("#### Lasso Linear Regression Model:")
        st.markdown("""
            - A linear model that estimates the coefficients of linear relationships.
            - It uses L1 regularization to prevent overfitting.
            - Suitable for high-dimensional datasets.
            - Can perform variable selection.
        """)
        st.write("Wrapped in a Multi-output regression model.")
        st.write("This allows for having multiple targets (3 flutes) at the same time.")

    # Dropdown selector for model selection
    col1, col2, col3= st.columns(3)
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
        y = wear_data[['flute_1', 'flute_2', 'flute_3']].values

        # Train model and display results
        best_lasso, scaler, train_errors, val_errors, best_fold_index = train_lasso_select_fold(X, y)

        if best_lasso is not None:  # Check if model training was successful
            st.write(f"### Best fold selected: {best_fold_index + 1} with validation MSE: {min(val_errors)}")
            plot_mse_st(train_errors, val_errors)

            # Make predictions
            y_pred = best_lasso.predict(scaler.transform(X))

            # Plot predictions for selected model data
            pred_col1, pred_col2, pred_col3 = st.columns(3)

            with pred_col1:
                plot_predictions_st(y[:, 0:1], y_pred[:, 0:1], flute_labels=['Flute 1'])

            with pred_col2:
                plot_predictions_st(y[:, 1:2], y_pred[:, 1:2], flute_labels=['Flute 2'])

            with pred_col3:
                plot_predictions_st(y[:, 2:3], y_pred[:, 2:3], flute_labels=['Flute 3'])

            # Prepare data for validation results
            validation_data_mapping = {
                "c1": "c6",
                "c4": "c1",
                "c6": "c4"
            }

            validation_selected_model = validation_data_mapping[selected_model]
            X_val, y_val = load_main_data(validation_selected_model).values, load_wear_data(validation_selected_model)[['flute_1', 'flute_2', 'flute_3']].values

            # Train model on validation data
            best_lasso_val, scaler_val, train_errors_val, val_errors_val, best_fold_index_val = train_lasso_select_fold(X_val, y_val)

            # Display best fold for validation data
            st.write(f"### Validation Results on {validation_selected_model.upper()} Dataset")
            st.write(f"### Best fold selected: {best_fold_index_val + 1} with validation MSE: {min(val_errors_val)}")

            # Plot validation predictions
            val_pred_col1, val_pred_col2, val_pred_col3 = st.columns(3)

            # Make predictions on validation data
            val_y_pred = best_lasso_val.predict(scaler_val.transform(X_val))

            with val_pred_col1:
                plot_predictions_st(y_val[:, 0:1], val_y_pred[:, 0:1], flute_labels=['Flute 1'])

            with val_pred_col2:
                plot_predictions_st(y_val[:, 1:2], val_y_pred[:, 1:2], flute_labels=['Flute 2'])

            with val_pred_col3:
                plot_predictions_st(y_val[:, 2:3], val_y_pred[:, 2:3], flute_labels=['Flute 3'])
        else:
            st.error("Model training failed due to data shape mismatch.")
    else:
        st.error("Failed to load the necessary data for the selected model.")
