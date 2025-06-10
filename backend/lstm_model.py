import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 📌 Utility: Create sequences for LSTM input
def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# 🔁 Update CSV if needed using yfinance
def update_csv_if_needed(symbol="INFY.NS", csv_path="backend/infy-bse.csv"):
    try:
        df = pd.read_csv(csv_path)
        df['Price Date'] = pd.to_datetime(df['Price Date'], errors='coerce', infer_datetime_format=True)
        df.dropna(subset=['Price Date'], inplace=True)
        df.sort_values("Price Date", inplace=True)
        last_date = df['Price Date'].max().date()
        today = datetime.today().date()

        if today > last_date and today.weekday() < 5:
            new_data = yf.download(symbol, start=last_date + timedelta(days=1), end=today + timedelta(days=1), progress=False)
            if not new_data.empty:
                new_row = {
                    'Price Date': new_data.index[-1].strftime('%d-%m-%Y'),
                    'Close Price': new_data['Close'].iloc[-1]
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df.to_csv(csv_path, index=False)
                print("✅ CSV updated with latest close price.")
            else:
                print("⚠️ No new market data available.")
        else:
            print("ℹ️ No update needed or market closed.")
    except Exception as e:
        print("❌ Error updating CSV:", e)

# 🔮 Predict next day's price
def predict_lstm(model_path="models/lstm_model.keras", scaler_path="models/scaler.pkl", csv_path="backend/infy-bse.csv"):
    if datetime.today().weekday() >= 5:
        print("⚠️ Weekend: Market is closed.")
        return None

    update_csv_if_needed(csv_path=csv_path)

    df = pd.read_csv(csv_path)
    df['Price Date'] = pd.to_datetime(df['Price Date'], dayfirst=True)
    df.sort_values("Price Date", inplace=True)
    close_prices = df[['Close Price']].values

    scaler = joblib.load(scaler_path)
    scaled_data = scaler.transform(close_prices)

    if len(scaled_data) < 60:
        raise ValueError("Need at least 60 records to predict.")

    last_60 = scaled_data[-60:].reshape(1, 60, 1)
    model = load_model(model_path)
    prediction = model.predict(last_60)
    return float(scaler.inverse_transform(prediction)[0][0])

# 📊 Evaluate LSTM model
def evaluate_lstm(model_path="models/lstm_model.keras", scaler_path="models/scaler.pkl", csv_path="backend/infy-bse.csv"):
    df = pd.read_csv(csv_path)
    df['Price Date'] = pd.to_datetime(df['Price Date'], dayfirst=True)
    df.sort_values("Price Date", inplace=True)

    close_prices = df[['Close Price']].values
    dates = df['Price Date'].reset_index(drop=True)
    scaler = joblib.load(scaler_path)
    scaled_data = scaler.transform(close_prices)

    X, y = create_sequences(scaled_data)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = load_model(model_path)
    predictions = model.predict(X)
    predictions_inverse = scaler.inverse_transform(predictions)
    y_inverse = scaler.inverse_transform(y.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_inverse, predictions_inverse))
    mae = mean_absolute_error(y_inverse, predictions_inverse)

    # 📈 Plot
    plt.figure(figsize=(12, 6))
    plt.plot(dates[60:], y_inverse.flatten(), label="Actual", marker='o')
    plt.plot(dates[60:], predictions_inverse.flatten(), label="Predicted", marker='x')
    plt.title(f"Infosys Stock Prediction (LSTM)\nMAE: {mae:.2f} | RMSE: {rmse:.2f}")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        "rmse": rmse,
        "mae": mae,
        "dates": dates[60:].dt.strftime('%Y-%m-%d').tolist(),
        "actual": y_inverse.flatten(),
        "predicted": predictions_inverse.flatten()
    }
