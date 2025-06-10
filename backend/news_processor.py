
from datetime import datetime
from backend.llm_analysis_cohere import generate_llm_analysis

from backend.mongo import insert_llm_analysis

def fetch_mock_news():
    return [
        "Infosys reports higher-than-expected quarterly profits",
        "Infosys launches new generative AI tool for enterprise clients",
        "IT sector remains strong as Infosys leads growth",
        "Infosys expands European operations with new acquisitions"
    ]

def analyze_and_store_news():
    headlines = fetch_mock_news()
    if not headlines:
        print("❌ No headlines available.")
        return

    llm_result = generate_llm_analysis(headlines, stock_name="Infosys")

    if llm_result:
        llm_result.update({
            "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "article_count": len(headlines),
            "source": "mock",
            "stock": "INFY.NS"
        })
        insert_llm_analysis(llm_result)
        print("✅ News analysis stored in MongoDB.")
    else:
        print("❌ OpenAI analysis failed.")
