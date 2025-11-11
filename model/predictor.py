import os
import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib 
from sklearn.metrics import mean_squared_error, r2_score

# --- Constants ---
# Paths are relative to this file (which is inside 'model/'),
# so we just need the filenames.
MODEL_FILE = 'crypto_model.joblib'
SCALER_FILE = 'crypto_scaler.joblib'

# --- Load Model and Scaler on Startup ---
try:
    model_path = os.path.join(os.path.dirname(__file__), MODEL_FILE)
    scaler_path = os.path.join(os.path.dirname(__file__), SCALER_FILE)
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"--- Model '{MODEL_FILE}' and Scaler '{SCALER_FILE}' loaded. Predictor is ready. ---")
except FileNotFoundError:
    print(f"--- FATAL ERROR: Model or Scaler file not found in 'model/' directory. ---")
    print(f"--- Please run 'model/training.py' first to create '{MODEL_FILE}' and '{SCALER_FILE}'. ---")
    model = None
    scaler = None
except Exception as e:
    print(f"--- FATAL ERROR: Error loading model files: {e} ---")
    model = None
    scaler = None

def is_model_ready():
    """Helper function to check if model is loaded."""
    return (model is not None) and (scaler is not None)

# --- [HELPER FUNCTION] ML Prediction Pipeline ---
def run_prediction_pipeline(csv_file_paths):
    """
    Runs the data processing and PREDICTION pipeline.
    
    Args:
        csv_file_paths (list): A list of paths to the user-uploaded CSV files.
        
    Returns:
        tuple: (metrics, plots, error)
        - metrics (dict): R², RMSE, etc.
        - plots (dict): Base64-encoded plot images.
        - error (str): An error message, if any.
    """
    
    if not model or not scaler:
        return None, None, "Model files are not loaded. Please check server logs and run 'model/training.py'."

    # --- 1. Data Loading and Preprocessing ---
    try:
        processed_dfs = []
        
        for file_path in csv_file_paths:
            try:
                df_coin = pd.read_csv(file_path)
                
                if not {'date', 'price', 'total_volume'}.issubset(df_coin.columns):
                    continue 

                df_coin.rename(columns={
                    'date': 'Date', 'price': 'Price', 'total_volume': 'Volume'
                }, inplace=True)

                df_coin['Date'] = pd.to_datetime(df_coin['Date'])
                df_coin.sort_values('Date', inplace=True)
                df_coin.set_index('Date', inplace=True)

                # Create Target Variable (for comparison)
                df_coin['Prediction_Target'] = df_coin['Price'].shift(-1)

                # Create Feature Variables (for prediction)
                df_coin['SMA_7'] = df_coin['Price'].rolling(window=7).mean()
                df_coin['SMA_30'] = df_coin['Price'].rolling(window=30).mean()
                df_coin['Feature_Price'] = df_coin['Price']
                df_coin['Feature_Volume'] = df_coin['Volume']
                
                processed_dfs.append(df_coin)

            except Exception as e:
                print(f"Skipping file {file_path} due to error: {e}")
                continue
        
        if not processed_dfs:
            raise ValueError("No valid data could be loaded. Ensure CSV files have 'date', 'price', and 'total_volume' columns.")

        df = pd.concat(processed_dfs)
        df.dropna(inplace=True)
        
        if df.empty:
            raise ValueError("Data became empty after processing. No data to predict.")

        # --- 2. Prepare Data for Prediction ---
        feature_columns = ['Feature_Price', 'Feature_Volume', 'SMA_7', 'SMA_30']
        X_predict = df[feature_columns] 
        y_actual = df['Prediction_Target'] 

    except (ValueError, Exception) as e:
        return None, None, str(e)


    # --- 3. Run Prediction (No Training!) ---
    try:
        X_scaled = scaler.transform(X_predict)
        y_pred = model.predict(X_scaled)
    except Exception as e:
        return None, None, f"Error during prediction: {e}. Was the model trained with the same features?"


    # --- 4. Evaluate and Store Metrics ---
    metrics = {}
    metrics['mse'] = mean_squared_error(y_actual, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['r2'] = r2_score(y_actual, y_pred)
    metrics['data_shape'] = df.shape
    metrics['prediction_count'] = len(y_pred)

    # --- 5. Visualization ---
    plots = {}
    plt.switch_backend('Agg') 
    plt.style.use('ggplot') # Use a nice style for the plots

    # --- Plot 1: Time-Series comparison ---
    fig1 = plt.figure(figsize=(14, 7))
    plt.plot(df.index, y_actual, label='Actual Price', color='#007bff', linewidth=1, alpha=0.8)
    plt.plot(df.index, y_pred, label='Predicted Price (from model)', color='#dc3545', linestyle='--', linewidth=1.5)
    plt.title(f'Crypto Price Prediction', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    buf1.seek(0)
    plots['time_series'] = base64.b64encode(buf1.getvalue()).decode('utf-8')
    plt.close(fig1)

    # --- Plot 2: Scatter plot ---
    fig2 = plt.figure(figsize=(8, 8))
    plt.scatter(y_actual, y_pred, alpha=0.6, edgecolors='k', s=50)
    
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.title('Actual vs. Predicted Price', fontsize=16)
    plt.xlabel('Actual Price (USD)', fontsize=12)
    plt.ylabel('Predicted Price (USD)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.tight_layout()
    
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png')
    buf2.seek(0)
    plots['scatter'] = base64.b64encode(buf2.getvalue()).decode('utf-8')
    plt.close(fig2)

    # --- NEW PLOT 3: Histogram of Prediction Errors (Residuals) ---
    errors = y_actual - y_pred
    fig3 = plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Distribution of Prediction Errors (Residuals)', fontsize=16)
    plt.xlabel('Prediction Error (Actual - Predicted)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean Error: ${errors.mean():.2f}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    buf3 = io.BytesIO()
    fig3.savefig(buf3, format='png')
    buf3.seek(0)
    plots['error_histogram'] = base64.b64encode(buf3.getvalue()).decode('utf-8')
    plt.close(fig3)

    # --- NEW PLOT 4: Feature Importance ---
    try:
        importances = model.feature_importances_
        # Create a DataFrame for easy plotting
        feature_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=True)

        fig4 = plt.figure(figsize=(10, 6))
        # Horizontal bar chart is better for feature names
        plt.barh(feature_df['Feature'], feature_df['Importance'], color='green', alpha=0.7)
        plt.title('Model Feature Importance', fontsize=16)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6, axis='x')
        plt.tight_layout()
        
        buf4 = io.BytesIO()
        fig4.savefig(buf4, format='png')
        buf4.seek(0)
        plots['feature_importance'] = base64.b64encode(buf4.getvalue()).decode('utf-8')
        plt.close(fig4)
    
    except AttributeError:
        # Handle cases where the model might not have feature_importances_ (e.g., if you change model type)
        print("Model does not have 'feature_importances_' attribute. Skipping plot.")
        plots['feature_importance'] = None
    
    return metrics, plots, None