from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from agent import StockMarketAgent

load_dotenv()

app = FastAPI(
    title="Stock Market AI Agent API", 
    description="API for fetching stock prices and providing analysis using LangChain and LLMs",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Stock Market Agent
agent = StockMarketAgent()

class StockRequest(BaseModel):
    ticker: str
    analyze: bool

class StockResponse(BaseModel):
    ticker: str
    price_data: dict
    analysis: str | None  

@app.get("/")
async def root():
    return {"message": "Stock Market AI Agent API is running"}

@app.post("/stock", response_model=StockResponse)
async def get_stock_info(request: StockRequest):
    """
    Fetch stock price information and optionally provide analysis
    
    Parameters:
    - ticker: Stock symbol (e.g., AAPL, MSFT, GOOGL)
    - analyze: Boolean flag to request buy/sell/hold recommendation
    
    Returns:
    - Stock price data and optional analysis
    """
    try:
        # Get stock price data
        price_data = agent.get_stock_price(request.ticker)
        
        # Get analysis if requested
        analysis = None
        if request.analyze:
            analysis = agent.analyze_stock(request.ticker, price_data)
        
        return StockResponse(
            ticker=request.ticker,
            price_data=price_data,
            analysis=analysis
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))