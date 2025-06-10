import yfinance as yf
import pandas as pd

# 📈 Fetch last known closing price from Yahoo (fallback-safe)
def fetch_live_price(symbol="INFY.NS"):
    try:
        data = yf.download(symbol, period="1d", interval="1m", progress=False)
        if not data.empty:
            return float(data['Close'].iloc[-1])
        else:
            raise ValueError("No live price data found.")
    except Exception as e:
        print(f"⚠️ Error fetching live price: {e}")
        return None

# 📊 Load last available price from historical CSV
def fetch_latest_price(csv_path="backend/infy-bse.csv"):
    df = pd.read_csv(csv_path)
    df['Price Date'] = pd.to_datetime(df['Price Date'], format='%d-%m-%Y')
    df.sort_values("Price Date", inplace=True)
    return float(df.iloc[-1]['Close Price'])
