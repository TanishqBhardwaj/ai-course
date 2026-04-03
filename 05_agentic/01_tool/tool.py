import yfinance as yf
import getpass
import os
from langchain.tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)

@tool
def get_stock_fundamentals(ticker: str) -> dict:
    """Get current stock price, P/E ratio, market cap, and revenue growth for a ticker."""
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "price": info.get("currentPrice"),
        "pe_ratio": info.get("trailingPE"),
        "market_cap": info.get("marketCap"),
        "revenue_growth": info.get("revenueGrowth"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
    }

tool_augmented_model = llm.bind_tools([get_stock_fundamentals])

response = tool_augmented_model.invoke("How is NVIDIA doing financially")

for tool_call in response.tool_calls:
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
    tool_result = get_stock_fundamentals.invoke(tool_call)
    print(tool_result.content)