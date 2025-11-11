# --- CELL 1: Imports and Setup ---
import pandas as pd
import numpy as np
import joblib # Used for saving the model and scaler
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

print("Libraries imported.")

# --- Configuration ---

# Get the directory that this script is in (e.g., 'model/')
SCRIPT_DIR = os.path.dirname(__file__)

# --- Defines the paths to all your *TRAINING* CSV files ---
# IMPORTANT: This data is used to CREATE the model.
# We assume these CSVs are in the *same directory* as this script.
# --- Defines the paths to all your *TRAINING* CSV files ---
TRAINING_CSV_FILES = [
    'algorand.csv',
    'apecoin.csv',
    'avalanche-2.csv',
    'axie-infinity.csv',
    'binancecoin.csv',
    'binance-usd.csv',
    'bitcoin.csv',
    'bitcoin-cash.csv',
    'cardano.csv',
    'chain-2.csv',
    'chainlink.csv',
    'chiliz.csv',
    'cosmos.csv',
    'crypto-com-chain.csv',
    'dai.csv',
    'decentraland.csv',
    'dogecoin.csv',
    'eos.csv',
    'ethereum-classic.csv',
    'ethereum.csv',
    'filecoin.csv',
    'flow.csv',
    'frax.csv',
    'ftx-token.csv',
    'hedera-hashgraph.csv',
    'internet-computer.csv',
    'leo-token.csv',
    'lido-dao.csv',
    'litecoin.csv',
    'matic-network.csv',
    'monero.csv',
    'near.csv',
    'okb.csv',
    'polkadot.csv',
    'quant-network.csv',
    'ripple.csv',
    'shiba-inu.csv',
    'solana.csv',
    'staked-ether.csv',
    'stellar.csv',
    'terra-luna.csv',
    'tether.csv',
    'tezos.csv',
    'the-sandbox.csv',
    'theta-token.csv',
    'tron.csv',
    'uniswap.csv',
    'usd-coin.csv',
    'vechain.csv',
    'wrapped-bitcoin.csv'
]

# Build the full paths to the CSVs
TRAINING_CSV_PATHS = [os.path.join(SCRIPT_DIR, f) for f in TRAINING_CSV_FILES]

# --- Model Output Files ---
# These are the files the Flask app will load
# We use SCRIPT_DIR to save them right next to this training.py file
MODEL_FILE = os.path.join(SCRIPT_DIR, 'crypto_model.joblib')
SCALER_FILE = os.path.join(SCRIPT_DIR, 'crypto_scaler.joblib')

print("Configuration set.")

# --- CELL 2: Data Loading and Preprocessing ---
try:
    processed_dfs = []
    
    print(f"Loading and processing {len(TRAINING_CSV_PATHS)} files for training...")

    for file_path in TRAINING_CSV_PATHS:
        try:
            df_coin = pd.read_csv(file_path)
            
            if not {'date', 'price', 'total_volume'}.issubset(df_coin.columns):
                print(f"--- WARNING: Skipping {file_path}. Missing required columns.")
                continue

            df_coin.rename(columns={
                'date': 'Date',
                'price': 'Price',
                'total_volume': 'Volume'
            }, inplace=True)

            df_coin['Date'] = pd.to_datetime(df_coin['Date'])
            df_coin.sort_values('Date', inplace=True)
            df_coin.set_index('Date', inplace=True)

            # 3. Target Variable (y)
            df_coin['Prediction_Target'] = df_coin['Price'].shift(-1)

            # 4. Feature Variables (X)
            df_coin['SMA_7'] = df_coin['Price'].rolling(window=7).mean()
            df_coin['SMA_30'] = df_coin['Price'].rolling(window=30).mean()
            df_coin['Feature_Price'] = df_coin['Price']
            df_coin['Feature_Volume'] = df_coin['Volume']
            
            processed_dfs.append(df_coin)
            print(f"Successfully processed {os.path.basename(file_path)}")

        except FileNotFoundError:
            print(f"--- ERROR: File not found at '{file_path}'. Skipping. ---")
        except Exception as e:
            print(f"--- ERROR: Unexpected error processing {file_path}: {e}. Skipping. ---")
    
    if not processed_dfs:
        raise ValueError("No data could be loaded for training. Please check file paths.")

    # 5. Concatenate all processed dataframes
    df = pd.concat(processed_dfs)
    
    # 6. Clean up NaNs
    df.dropna(inplace=True)
    
    if df.empty:
         raise ValueError("Data became empty after processing.")

    print(f"\nAll training data loaded and processed.")
    print(f"Final combined data shape: {df.shape}")

    # Prepare X (features) and y (target) for the model
    feature_columns = ['Feature_Price', 'Feature_Volume', 'SMA_7', 'SMA_30']
    X = df[feature_columns]
    y = df['Prediction_Target']

    # 7. Splitting Data: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    print(f"\nTraining data size (X_train): {len(X_train)}")
    print(f"Testing data size (X_test): {len(X_test)}")
    
    df_empty = False

except (ValueError, Exception) as e:
    print(f"--- FATAL ERROR ---")
    print(f"An error occurred during data processing: {e}")
    df_empty = True

# --- CELL 3: Model Training & Finetuning ---

if not df_empty:
    # 1. Initialize and Fit the Scaler
    # We fit the scaler ONLY on the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    print("Scaler fitted on training data.")

    # 2. Initialize and Train the Decision Tree Regressor
    # *** FINETUNING: You can change model parameters here! ***
    model = DecisionTreeRegressor(random_state=42, max_depth=10, min_samples_leaf=5)

    # Train the model using the SCALED training data
    model.fit(X_train_scaled, y_train)
    print(f"Model ({type(model).__name__}) trained.")
    
    # 3. Evaluate the model (Optional, but good practice)
    print("\n--- Model Performance on Test Set ---")
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"R-squared (R²) Score: {r2:.4f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")

# --- CELL 4: Save Model and Scaler --- 

if not df_empty:
    # Save the trained model
    joblib.dump(model, MODEL_FILE)
    print(f"\nSuccessfully saved model to: {MODEL_FILE}")

    # Save the fitted scaler
    joblib.dump(scaler, SCALER_FILE)
    print(f"Successfully saved scaler to: {SCALER_FILE}")
else:
    print("\nSkipping model saving due to data processing error.")