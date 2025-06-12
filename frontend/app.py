import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Access backend

import streamlit as st
import matplotlib.pyplot as plt

from backend.news_processor import analyze_and_store_news
from backend.lstm_model import predict_lstm, evaluate_lstm
from backend.fetch_live_data import fetch_live_price
from backend.mongo import get_latest_llm
from backend.llm_verdict import generate_llm_verdict  # 🔥 new import

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

# 🧠 LLM-Based Final Verdict (No Ensemble)
st.subheader("🧠 Final Verdict by LLM")

# Sample fundamentals (can be dynamically loaded later)
fundamentals = {
    "P/E": 24.5,
    "EPS": 53.2,
    "ROE": "19%",
    "Revenue Growth": "13% YoY"
}

if lstm_pred is not None and live_price is not None and llm_data:
    with st.spinner("🧠 Thinking..."):
        verdict_result = generate_llm_verdict(
            predicted_price=lstm_pred,
            current_price=live_price,
            news_sentiment=llm_data,
            fundamentals=fundamentals
        )
    st.success(f"📌 **Verdict:** {verdict_result['verdict']}")
    st.info(f"📝 **Reasoning:** {verdict_result['reasoning']}")
else:
    st.info("Please refresh to generate prediction and sentiment before showing verdict.")

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
