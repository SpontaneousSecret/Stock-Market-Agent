import os
import json
from typing import Dict, Any, Optional

def validate_api_keys():
    """
    Validate that all required API keys are present in the environment.
    
    Returns:
        bool: True if all required keys are present, False otherwise
    """
    required_keys = ["GROQ_API_KEY"]
    
    for key in required_keys:
        if not os.getenv(key):
            print(f"Missing required environment variable: {key}")
            return False
    
    return True

def format_currency(value: float) -> str:
    """
    Format a numeric value as currency.
    
    Args:
        value (float): Numeric value to format
        
    Returns:
        str: Formatted currency string
    """
    if value is None:
        return "N/A"
    
    return f"${value:,.2f}"

def format_large_number(value: int) -> str:
    """
    Format large numbers with K, M, B suffixes.
    
    Args:
        value (int): Number to format
        
    Returns:
        str: Formatted number string
    """
    if value is None:
        return "N/A"
    
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.2f}K"
    else:
        return f"{value}"
    

def save_cache(key: str, data: Dict[str, Any], cache_dir: str = ".cache"):
    """
    Save data to a cache file.
    
    Args:
        key (str): Cache key
        data (Dict[str, Any]): Data to cache
        cache_dir (str): Directory to store cache files
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{key}.json")
    
    with open(cache_file, "w") as f:
        json.dump(data, f)


def load_cache(key: str, cache_dir: str = ".cache") -> Optional[Dict[str, Any]]:
    """
    Load data from a cache file.
    
    Args:
        key (str): Cache key
        cache_dir (str): Directory storing cache files
        
    Returns:
        Optional[Dict[str, Any]]: Cached data if available, None otherwise
    """
    cache_file = os.path.join(cache_dir, f"{key}.json")
    
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    
    return None