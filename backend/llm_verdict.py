"""
llm_verdict.py

This module generates a stock verdict ('Buy', 'Hold', or 'Sell') using a Large Language Model (Cohere Command model),
based on the LSTM predicted price, current price, recent news sentiment, and company fundamentals. It then stores 
the verdict and reasoning in MongoDB and returns the result.

Functions:
- generate_llm_verdict(predicted_price, current_price, news_sentiment, fundamentals) -> dict
"""

import os
import json
from datetime import datetime

import cohere
from backend.mongo import db
 # Uses existing MongoDB connection from mongo.py

# Initialize Cohere client with API key from environment (ensure COHERE_API_KEY is set in environment or config)
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if COHERE_API_KEY is None:
    raise RuntimeError("COHERE_API_KEY environment variable not set. Please configure your Cohere API key.")

# Cohere model name and configuration
MODEL_NAME = "command-r-plus"  # using Cohere's Command model (R+) for generation
MAX_TOKENS = 100               # limit the response length (enough for verdict + brief reasoning)
TEMPERATURE = 0.5              # moderate randomness for coherent but not deterministic output

def generate_llm_verdict(predicted_price: float, current_price: float, news_sentiment, fundamentals: dict) -> dict:
    """
    Generate an LLM-based stock verdict ("Buy", "Hold", or "Sell") with reasoning.
    
    Parameters:
        predicted_price (float): The stock price predicted by the LSTM model.
        current_price   (float): The current stock price.
        news_sentiment        : News sentiment analysis output (schema as per llm_analysis_cohere.py, e.g. a dict or JSON from MongoDB).
        fundamentals   (dict): Company fundamentals (e.g., {"P/E": 15.2, "EPS": 2.5, "ROE": "18%", "Revenue Growth": "12%"}).
    
    Returns:
        dict: A dictionary with keys "verdict" (str) and "reasoning" (str), representing the LLM's recommendation and explanation.
    """
    # 1. Prepare context information for the prompt:
    ticker = None
    company_name = None
    # If news_sentiment is a dict (e.g., a MongoDB document), try to extract ticker or company name for context
    if isinstance(news_sentiment, dict):
        ticker = news_sentiment.get("ticker") or news_sentiment.get("symbol")
        company_name = news_sentiment.get("company") or news_sentiment.get("company_name")
    # Alternatively, check fundamentals for a company identifier
    if ticker is None and isinstance(fundamentals, dict):
        ticker = fundamentals.get("ticker") or fundamentals.get("symbol") or fundamentals.get("company")
    
    # 2. Build the prompt with structured data:
    prompt_lines = []
    if ticker or company_name:
        # Include the company identifier in the prompt for clarity (prefer company name over ticker if available)
        identifier = company_name if company_name else ticker
        prompt_lines.append(f"Company: {identifier}")
    # Add predicted and current prices
    prompt_lines.append(f"Predicted Price: {predicted_price}")
    prompt_lines.append(f"Current Price: {current_price}")
    # Add each fundamental metric
    if isinstance(fundamentals, dict):
        for key, value in fundamentals.items():
            prompt_lines.append(f"{key}: {value}")
    # Add news sentiment summary or info
    news_summary_text = ""
    if isinstance(news_sentiment, dict):
        # If the sentiment analysis dict contains a summary or analysis text, use it
        if news_sentiment.get("summary"):
            news_summary_text = news_sentiment["summary"]
        elif news_sentiment.get("analysis"):
            news_summary_text = news_sentiment["analysis"]
        # If only a sentiment label or score is available, include that
        elif news_sentiment.get("sentiment"):
            news_summary_text = f"Overall News Sentiment: {news_sentiment['sentiment']}"
    elif isinstance(news_sentiment, str):
        # If a plain string is provided (e.g., already formatted summary)
        news_summary_text = news_sentiment
    # Append news sentiment info (or N/A if not provided)
    if news_summary_text:
        prompt_lines.append(f"News Sentiment: {news_summary_text}")
    else:
        prompt_lines.append("News Sentiment: N/A")
    
    # Compose the final prompt text with all lines and clear instruction for output format
    prompt_text = "Consider the following information about the stock:\n"
    for line in prompt_lines:
        prompt_text += f"- {line}\n"
    # Instruction: ask for Buy/Hold/Sell verdict with reasoning, output strictly as JSON
    prompt_text += (
        "\nDecide whether this stock should be a 'Buy', 'Hold', or 'Sell' based on the above data, "
        "and explain the reasoning in one or two sentences.\n"
        "Provide the answer as a JSON object with the keys \"verdict\" and \"reasoning\" only."
    )
    
    # 3. Call Cohere API to get LLM-generated verdict:
    co = cohere.Client(COHERE_API_KEY)
    try:
        response = co.generate(
            model=MODEL_NAME,
            prompt=prompt_text,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stop_sequences=[]  # no explicit stop sequence; rely on JSON format instruction
        )
    except Exception as api_error:
        # If the Cohere API call fails, handle the error and return a default response
        print(f"[Error] Cohere API call failed: {api_error}")
        default_result = {"verdict": "Hold", "reasoning": "Unable to generate verdict due to an LLM error."}
        # We can still log this default verdict to DB (with an error flag) if needed
        try:
            verdict_doc = {
                "verdict": default_result["verdict"],
                "reasoning": default_result["reasoning"],
                "timestamp": datetime.utcnow(),
                "error": str(api_error)
            }
            if ticker:
                verdict_doc["ticker"] = ticker
            db["llm_verdicts"].insert_one(verdict_doc)
        except Exception as db_error:
            print(f"[Warning] Failed to save default verdict to MongoDB: {db_error}")
        return default_result
    
    # Extract the generated text from Cohere's response
    output_text = response.generations[0].text.strip() if response.generations else ""
    if not output_text:
        # If no output was generated, return a default "Hold" verdict
        default_result = {"verdict": "Hold", "reasoning": "No response from LLM."}
        return default_result
    
    # 4. Parse LLM output expecting a JSON format:
    verdict = None
    reasoning = None
    try:
        # Attempt to parse the output as JSON
        result_json = json.loads(output_text)
        verdict = result_json.get("verdict")
        reasoning = result_json.get("reasoning")
    except json.JSONDecodeError:
        # If JSON decoding fails, try to recover by extracting JSON substring
        start_idx = output_text.find("{")
        end_idx = output_text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            try:
                result_json = json.loads(output_text[start_idx:end_idx+1])
                verdict = result_json.get("verdict")
                reasoning = result_json.get("reasoning")
            except Exception:
                verdict = None  # will trigger default if still not parsed
    except Exception as parse_error:
        # Handle any other parsing-related exceptions
        print(f"[Warning] Unexpected parsing error: {parse_error}")
    
    # Validate that we have the required fields
    if not verdict or not reasoning:
        # If the LLM output didn't follow the expected JSON format, use a safe default
        verdict = verdict or "Hold"
        reasoning = reasoning or "LLM response could not be parsed into a valid JSON."
    
    # Prepare the result dictionary
    result = {"verdict": verdict, "reasoning": reasoning}
    
    # 5. Save the verdict result to MongoDB (llm_verdicts collection) with a timestamp
    try:
        verdict_record = {
            "verdict": verdict,
            "reasoning": reasoning,
            "timestamp": datetime.utcnow()  # use UTC time for consistency
        }
        # Include additional context in the stored record
        if ticker:
            verdict_record["ticker"] = ticker
        if predicted_price is not None:
            verdict_record["predicted_price"] = predicted_price
        if current_price is not None:
            verdict_record["current_price"] = current_price
        if fundamentals:
            verdict_record["fundamentals"] = fundamentals
        if news_summary_text:
            verdict_record["news_sentiment_summary"] = news_summary_text
        db["llm_verdicts"].insert_one(verdict_record)  # insert the document into the collection:contentReference[oaicite:2]{index=2}
    except Exception as db_error:
        # If saving to Mongo fails, log the error and continue (not critical for function return)
        print(f"[Warning] Failed to save verdict to MongoDB: {db_error}")
    
    # 6. Return the verdict result for immediate use (e.g., in the UI)
    return result
