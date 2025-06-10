import os
import json
import cohere
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / "env/.env")

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

def generate_llm_analysis(news_list, stock_name="Infosys"):
    prompt = f"""
You are a financial analyst.

Based on the following news headlines about {stock_name}, provide:
1. A summary
2. Key pros
3. Key cons
4. Sentiment (Positive, Neutral, or Negative)
5. Final verdict (Buy, Hold, or Sell)

News Headlines:
{chr(10).join(news_list)}

Respond in JSON format like this:
{{
  "summary": "...",
  "pros": ["..."],
  "cons": ["..."],
  "sentiment": "...",
  "verdict": "..."
}}
"""
    try:
        response = co.chat(
            model="command-r-plus",
            message=prompt,
            temperature=0.7,
            connectors=[]  # disable web-search
        )
        text = response.text.strip()

        # Attempt to parse output as JSON
        if "```json" in text:
            text = text.split("```json")[-1].split("```")[0].strip()
        result = json.loads(text)

        # Add impact score for Mongo
        sentiment = result.get("sentiment", "").lower()
        result["impact_score"] = {"positive": 1, "neutral": 0, "negative": -1}.get(sentiment, 0)

        return result

    except Exception as e:
        print("❌ Cohere error:", e)
        return None
