import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
import joblib

# Load dataset
df = pd.read_csv("backend/infy-bse.csv")
df['Price Date'] = pd.to_datetime(df['Price Date'], format='%d-%m-%Y')
df.sort_values("Price Date", inplace=True)
df.dropna(subset=["Close Price"], inplace=True)

# Scale close prices
close_prices = df[['Close Price']].values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_prices)
joblib.dump(scaler, "models/scaler.pkl")

# Create LSTM sequences
def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build and train model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
model.add(LSTM(50))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer=Adam(), loss="mean_squared_error")
model.fit(X, y, batch_size=32, epochs=20)

# Save model
model.save("models/lstm_model.keras")
print("✅ Model and scaler saved.")
