import yfinance as yf
import getpass
import os
from langchain.tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["SERP_API_KEY"] = os.getenv("SERP_API_KEY")
os.environ["USER_AGENT"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)

serp = SerpAPIWrapper(
    serpapi_api_key=os.getenv("SERP_API_KEY"),
    params={
        "tbm": "nws",     # Search in Google News
        "tbs": "qdr:d",   # Past day (24 hours)
    },
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

@tool
def search_news(query: str) -> str:
    """
    Search last-24h Google News via SerpAPI.
    Returns news results with URLs.
    """
    return serp.run(query)

tool_augmented_model = llm.bind_tools([get_stock_fundamentals])

# response = tool_augmented_model.invoke("How is NVIDIA doing financially")

# for tool_call in response.tool_calls:
#     print(f"Tool: {tool_call['name']}")
#     print(f"Args: {tool_call['args']}")
#     tool_result = get_stock_fundamentals.invoke(tool_call)
#     print(tool_result.content)

tools = [
    get_stock_fundamentals,
    YahooFinanceNewsTool(),
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
    search_news
]

SYSTEM_PROMPT = """
You are FinBot, an expert equity research analyst with deep knowledge of financial markets,
valuation methodologies, and macroeconomic trends.

## Task
Given a stock ticker or company name, produce a concise, structured analyst brief that helps users evaluate the investment. Do not give buy/sell advice. Present data-driven signals only.

## Rules
1. Gather data before analysis. Never rely on memory for numbers.
2. If a tool fails or returns empty data, state it and proceed.
3. Never fabricate prices, ratios, or news.
4. Always follow the output format.
5. Flag notable risks or red flags.

## Output Format

**[TICKER] — Analyst Brief**
- 📊 **Fundamentals:** price, P/E, market cap, revenue growth (one line)
- 📈 **Valuation Signal:** OVERVALUED / FAIRLY VALUED / UNDERVALUED + reason
- 📰 **News Sentiment:** bullish / neutral / bearish + key headline
- ⚠️ **Key Risks:** 1-2 bullets
- 🧭 **Outlook:** 1-2 sentence synthesis, no advice
"""

my_finance_agent = create_agent(
    model= llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread_id": "user-tano-session-1"}}

agent_trace = my_finance_agent.invoke(
    {"messages": [{"role": "user", "content": "Give me a quick analyst brief on NVDA. Is now a good time to buy?"}]},
    config
)

print(agent_trace["messages"][-1].content)

agent_trace = my_finance_agent.invoke(
    {"messages": [{"role": "user", "content": "How does it compare with AMD?"}]},
    config
)

print(agent_trace["messages"][-1].content)