import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from tools.stock_tools import StockPriceTool, WebSearchTool
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from typing import List

class StockMarketAgent:
    """
    Agent for fetching stock market data and providing analysis using LangChain and Groq LLM.
    """

    def __init__(self):
        # Initialize Groq LLM
        self.llm = ChatGroq(
            model=os.getenv("LLM_MODEL", "llama3-70b-8192"),
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )

        # Initialize tools
        self.stock_price_tool = StockPriceTool()
        self.web_search_tool = WebSearchTool()
        
        # Use a simpler approach with tools
        self.tools = [
            self.stock_price_tool,
            self.web_search_tool
        ]
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create tool description text
        tool_descriptions = "\n\n".join([
            f"Tool: {tool.name}\nDescription: {tool.description}\nUsage: {tool.name}({{arguments}})" 
            for tool in self.tools
        ])

        # Create a prompt template with manual tool formatting - avoid using functions API
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional stock market analyst assistant. 
            Your job is to provide accurate information about stock prices and 
            offer insightful analysis when requested.
            
            Always verify information by using the most recent data available.
            
            When providing analysis, consider:
            1. Current price vs. historical trends
            2. Recent news and market sentiment
            3. Industry comparison
            4. Technical indicators
            
            Be specific and provide clear recommendations when asked.
            
            Available tools:
            {tool_names}

            You have access to the following tools:
            {tools}
             
            Use the tools when needed and reason step by step to provide answers.
            """),

            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create the ReAct agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )

        # Create the agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

    def get_stock_price(self, ticker):
        """
        Fetch the latest stock price for the given ticker symbol.
        
        Args:
            ticker (str): The stock ticker symbol (e.g., AAPL, MSFT)
            
        Returns:
            dict: Dictionary containing stock price data
        """
        try:
            # Directly call the tool to get stock price
            result = self.stock_price_tool.invoke({"ticker": ticker})
            return result
        except Exception as e:
            # If the first method fails, try using the agent with web search
            response = self.agent_executor.invoke({
                "input": f"What is the current stock price of {ticker}? Just return the price data."
            })
            return response.get("output", {"error": str(e)})
        
    def analyze_stock(self, ticker, price_data):
        """
        Analyze the stock and provide buy/sell/hold recommendation.
        
        Args:
            ticker (str): The stock ticker symbol
            price_data (dict): Dictionary containing stock price data
            
        Returns:
            str: Analysis and recommendation
        """
        response = self.agent_executor.invoke({
            "input": f"""Analyze {ticker} stock based on this price data: {price_data}. 
            Search for recent news and market sentiment about {ticker}.
            Provide a clear buy, sell, or hold recommendation with brief justification."""
        })
        
        return response.get("output", "Unable to analyze stock at this time.")