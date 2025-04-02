import yfinance as yf
import requests
from langchain.tools import BaseTool
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel, Field, ConfigDict
import json
from duckduckgo_search import DDGS

class StockPriceInput(BaseModel):
    ticker: str = Field(description="The stock ticker symbol, e.g., AAPL, MSFT, GOOGL")
    model_config = ConfigDict(extra="forbid")


class StockPriceTool(BaseTool):
    name: str ="get_stock_price"
    description:str ="Gets the latest stock price information for a given ticker symbol"
    args_schema: Type[BaseModel]=StockPriceInput

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch the latest stock price information using Yahoo Finance API.
        
        Args:
            ticker (str): The stock ticker symbol
            
        Returns:
            dict: Dictionary containing stock price data
        """
        try:
            # Get stock information
            stock = yf.Ticker(ticker)
            
            # Get the historical market data
            hist = stock.history(period="1d")
            
            # Get today's data
            if not hist.empty:
                latest_data = hist.iloc[-1]
                
                # Get additional info
                info = stock.info
                
                result = {
                    "ticker": ticker,
                    "current_price": round(latest_data["Close"], 2),
                    "open": round(latest_data["Open"], 2),
                    "high": round(latest_data["High"], 2),
                    "low": round(latest_data["Low"], 2),
                    "volume": int(latest_data["Volume"]),
                    "change_percent": round((latest_data["Close"] - latest_data["Open"]) / latest_data["Open"] * 100, 2),
                    "market_cap": info.get("marketCap", None),
                    "timestamp": hist.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
                }
                
                return result
            else:
                return {"error": f"No data found for ticker {ticker}"}
                
        except Exception as e:
            return {"error": f"Error fetching data for {ticker}: {str(e)}"}
        
class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to perform")

class WebSearchTool(BaseTool):
    name: str= "web_search"
    description:str = "Search the web for the latest information about stocks and market data"
    args_schema: Type[BaseModel] = WebSearchInput
    
    def _run(self, query: str) -> str:
        """
        Search the web using DuckDuckGo.
        
        Args:
            query (str): The search query
            
        Returns:
            str: Search results
        """
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=5)
                
            formatted_results = []
            for r in results:
                formatted_results.append(f"Title: {r['title']}\nBody: {r['body']}\nURL: {r['href']}\n")
                
            return "\n".join(formatted_results) if formatted_results else "No results found."
            
        except Exception as e:
            return f"Error performing web search: {str(e)}"