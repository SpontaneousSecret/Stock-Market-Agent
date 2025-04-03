import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory

class StockMarketAgent:
    """
    Agent for fetching stock market data and providing analysis using LangChain and Groq LLM.
    """

    def __init__(self):
        # Initialize Groq LLM
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
            
        self.llm = ChatGroq(
            model=os.getenv("LLM_MODEL", "llama3-70b-8192"),  # Default to Llama3 70B model
            temperature=0,
            api_key=api_key
        )

        # Initialize tools
        from tools.stock_tools import StockPriceTool, WebSearchTool
        self.stock_price_tool = StockPriceTool()
        self.web_search_tool = WebSearchTool()
        self.tools = [self.stock_price_tool, self.web_search_tool]

        # Set up memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create prompt template that includes agent_scratchpad explicitly
        template = """You are a professional stock market analyst assistant. 
        Your job is to provide accurate information about stock prices and 
        offer insightful analysis when requested.
        
        Always verify information by using the most recent data available.
        
        When providing analysis, consider:
        1. Current price vs. historical trends
        2. Recent news and market sentiment
        3. Industry comparison
        4. Technical indicators
        
        Be specific and provide clear recommendations when asked.
        
        You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Previous conversation history:
        {chat_history}

        Question: {input}
        {agent_scratchpad}
        """
        
        # Create prompt with the tools and tool names explicitly formatted
        tool_names = [tool.name for tool in self.tools]
        tools_formatted = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["chat_history", "input", "agent_scratchpad"],
            partial_variables={"tools": tools_formatted, "tool_names": ", ".join(tool_names)}
        )
        
        # Create the ReAct agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )

        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )

    def get_stock_price(self, ticker):
        """
        Fetch the latest stock price for the given ticker symbol.
        
        Args:
            ticker (str): The stock ticker symbol (e.g., AAPL, MSFT)
            
        Returns:
            dict: Dictionary containing stock price data
        """
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Ticker must be a valid string")
            
        try:
            # Directly call the tool to get stock price
            result = self.stock_price_tool.invoke({"ticker": ticker})
            return result
        except Exception as e:
            # Log the specific error for debugging
            print(f"Error fetching stock price directly: {str(e)}")
            
            # If the first method fails, try using the agent with web search
            response = self.agent_executor.invoke({
                "input": f"What is the current stock price of {ticker}? Just return the price data."
            })
            
            # Extract the output from the response
            if isinstance(response, dict) and "output" in response:
                return response["output"]
            return response
        
    def analyze_stock(self, ticker, price_data):
        """
        Analyze the stock and provide buy/sell/hold recommendation with fallback mechanism.
    
        Args:
            ticker (str): The stock ticker symbol
            price_data (dict): Dictionary containing stock price data
        
        Returns:
            str: Analysis and recommendation
        """
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Ticker must be a valid string")
    
        try:
        # Try with the agent first
            response = self.agent_executor.invoke({
                "input": f"""Analyze {ticker} stock based on this price data: {price_data}.
                Search for recent news and market sentiment about {ticker}.
                Provide a clear buy, sell, or hold recommendation with brief justification."""
            })
        
        # Check if we got a proper response
            output = response.get("output", "")
            if output and "iteration limit" not in output.lower() and "time limit" not in output.lower():
                return output
            
        # If we hit the limit, use direct LLM call as fallback
            print(f"Agent hit iteration/time limit. Using direct LLM fallback for {ticker}.")
        
        # Use a direct call to the LLM without the agent framework
            fallback_prompt = f"""
            You are a professional stock market analyst. Based on the following information about {ticker} stock:
        
            Price data: {price_data}
        
            Please provide:
            1. A brief summary of what this stock data indicates
            2. A clear BUY, SELL, or HOLD recommendation
            3. A 1-2 sentence justification for your recommendation
        
            Format your response as a brief paragraph with your recommendation clearly stated.
            """
        
            direct_response = self.llm.invoke(fallback_prompt)
            return direct_response.content
        
        except Exception as e:
            print(f"Error in stock analysis: {str(e)}")
            return f"Analysis failed for {ticker}: {str(e)}"
        
    def reset_memory(self):
        """
        Clear the conversation history in memory.
        Useful for starting fresh conversations or managing memory usage.
        """
        self.memory.clear()