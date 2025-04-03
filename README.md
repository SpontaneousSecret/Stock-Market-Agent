Stock Market AI Agent API

Overview

Stock Market AI Agent API is a FastAPI-based service that fetches stock price data and provides insightful analysis using LangChain and Groq LLM. The agent integrates tools for real-time stock market analysis and web search capabilities to enhance decision-making.

Features

Fetch real-time stock prices using Yahoo Finance API

Analyze stock trends and provide buy/sell/hold recommendations

Utilize Groq's LLM for intelligent insights

Web search tool for fetching the latest stock news and sentiment analysis

Built with FastAPI for a fast and efficient API experience

Deployment

The API is deployed on Render and can be accessed at:

Base URL: https://stock-market-pricefetch-analyze.onrender.com

Installation

Prerequisites

Ensure you have Python 3.8+ installed.

Clone the repository:

git clone https://github.com/SpontaneousSecret/Stock-Market-PriceFetch-Analyze.git

Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Set up environment variables:
Create a .env file in the root directory and add:

GROQ_API_KEY=your_api_key_here

Usage

Running the API Locally

Start the FastAPI server:

uvicorn main:app --reload

The API will be available at http://127.0.0.1:8000

Endpoints

Root Endpoint

GET /

Returns a welcome message to confirm the API is running.

Fetch Stock Data

POST /stock

Request Body:

{
    "ticker": "AAPL",
    "analyze": true
}

Response:

{
    "ticker": "AAPL",
    "price_data": {
        "current_price": 150.25,
        "open": 149.5,
        "high": 152.0,
        "low": 148.75,
        "volume": 50000000,
        "change_percent": 0.5,
        "market_cap": "2.5T"
    },
    "analysis": "AAPL is in an upward trend based on historical performance. Consider holding or buying if the trend continues."
}

Postman API Requests

For testing the deployed url, use the following Postman configurations:

GET Request:

URL: https://stock-market-pricefetch-analyze.onrender.com/

Headers:

Content-Type: application/json

POST Request:

URL: https://stock-market-pricefetch-analyze.onrender.com/stock

Headers:

Content-Type: application/json

Body:

{
    "ticker": "GOOGL",
    "analyze": "false"
}

Project Structure

![image](https://github.com/user-attachments/assets/5319ed36-7a74-49ca-b556-2ec9da4b8f11)


Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

License

This project is licensed under the MIT License.

Contact

For any queries, reach out to gsujalr@gmail.com

