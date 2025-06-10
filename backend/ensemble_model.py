
def combine_predictions(delta, llm_sentiment):
    if llm_sentiment.lower() == "positive":
        if delta > 0:
            return "Buy"
        else:
            return "Hold"
    elif llm_sentiment.lower() == "negative":
        return "Sell"
    return "Hold"
