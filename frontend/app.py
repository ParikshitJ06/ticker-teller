import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Access backend

import streamlit as st
import matplotlib.pyplot as plt

from backend.news_processor import analyze_and_store_news
from backend.lstm_model import predict_lstm, evaluate_lstm
from backend.ensemble_model import combine_predictions
from backend.fetch_live_data import fetch_live_price
from backend.mongo import get_latest_llm

# ───────────── Streamlit UI ─────────────
st.set_page_config(page_title="Infosys Stock AI Dashboard", layout="centered")
st.title("📊 Ticker Teller – INFY Stock Insight")

# 🔁 Refresh Button
if st.button("🔄 Refresh Analysis"):
    with st.spinner("Running analysis..."):
        analyze_and_store_news()
        latest_llm = get_latest_llm()
        lstm_price = predict_lstm()
        st.success("✅ Refreshed successfully!")
    st.rerun()

# 🧠 LLM Sentiment Output
st.subheader("🧠 LLM Sentiment Analysis")
llm_data = get_latest_llm()

if llm_data:
    st.markdown(f"**Sentiment:** {llm_data['sentiment']}")
    st.markdown(f"**Verdict:** {llm_data['verdict']}")
    st.markdown(f"**Pros:** {', '.join(llm_data['pros'])}")
    st.markdown(f"**Cons:** {', '.join(llm_data['cons'])}")
else:
    st.warning("No LLM data available yet.")

# 📊 LSTM Prediction
st.subheader("📊 LSTM Predicted Price")
lstm_pred = predict_lstm()
if lstm_pred is not None:
    st.metric("Predicted Price (Next Day)", f"₹{lstm_pred:.2f}")
else:
    st.warning("Prediction skipped – market is closed today.")

# 💰 Live Price
st.subheader("💰 Actual Market Price")
try:
    live_price = fetch_live_price()
    st.metric("Live Market Price", f"₹{live_price:.2f}")
except:
    st.error("Failed to fetch live market price.")

# 🧩 Ensemble Final Verdict
st.subheader("🧩 Ensemble Final Recommendation")
if lstm_pred is not None and llm_data:
    delta = lstm_pred - live_price
    verdict = combine_predictions(delta, llm_data['sentiment'])
    st.success(f"📌 Final Verdict: **{verdict}**")
else:
    st.info("Awaiting both prediction and LLM data to generate recommendation.")

# 📉 Model Evaluation
st.subheader("📉 LSTM Model Evaluation")
if st.button("Evaluate LSTM Accuracy"):
    eval_results = evaluate_lstm()

    st.write(f"**RMSE:** {eval_results['rmse']:.4f}")
    st.write(f"**MAE:** {eval_results['mae']:.4f}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(eval_results["dates"], eval_results["actual"], label="Actual")
    ax.plot(eval_results["dates"], eval_results["predicted"], label="Predicted")
    ax.set_title("Predicted vs Actual Close Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)
