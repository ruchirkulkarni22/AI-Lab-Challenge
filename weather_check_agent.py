"""
Weather Checker Agent with LLM Integration

This script enhances the original weather checker by incorporating LangChain and 
Ollama to create a more robust agent. It validates if the weather in a specified 
city matches an expected condition.

Usage:
    python weather_checker_agent.py <city_name> <expected_condition>

Example:
    python weather_checker_agent.py "Pune" "Sunny"
"""

import sys
import time
import os
import platform
import re
from typing import Dict, Any, List

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# LangChain imports
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.schema import AgentAction, AgentFinish

class WeatherCheckerAgent:
    def __init__(self):
        """Initialize the Weather Checker Agent with tools and LLM."""
        # Initialize Ollama LLM (open source)
        try:
            # Configure Ollama - ensure it's running locally at the default URL
            self.llm = Ollama(model="llama3", base_url="http://localhost:11434")
            print("Using Ollama with llama3 model")
        except Exception as e:
            print(f"Error initializing Ollama: {e}")
            print("Make sure Ollama server is running with 'ollama serve' in a separate terminal")
            print("And that you've downloaded the llama3 model with 'ollama pull llama3'")
            sys.exit(1)

        # Define tools that the agent can use
        self.tools = [
            Tool(
                name="CheckWeather",
                func=self.check_weather,
                description="Checks the current weather for a city and compares it with expected conditions."
            ),
            Tool(
                name="AnalyzeWeatherResult",
                func=self.analyze_weather_match,
                description="Analyzes if the actual weather condition matches the expected condition."
            )
        ]

        # Create a prompt for the agent
        template = """You are a Weather Checking Agent designed to verify weather conditions.

Given a city name and an expected weather condition, your job is to check the actual weather 
and determine if it matches the expectation.

Use these tools to help you:
{tools}

To solve this problem:
1. Use the CheckWeather tool to fetch the current weather for the city
2. Use the AnalyzeWeatherResult tool to determine if it matches the expected condition
3. Return the final result with a clear explanation

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}
"""

        prompt = PromptTemplate.from_template(template)
        
        # Create the agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create the agent executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent, 
            tools=self.tools, 
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3  # Limit iterations to prevent infinite loops
        )

    def setup_webdriver(self):
        """Set up and return a Selenium WebDriver instance."""
        # Setup Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Set a common user agent to avoid detection
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        # Add settings to bypass bot detection
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        
        # Determine if running on Windows and set appropriate settings
        is_windows = platform.system() == "Windows"
        
        try:
            print(f"Setting up WebDriver...")
            
            # For Windows, use a more specific installation approach
            if is_windows:
                # Get the ChromeDriver executable path
                driver_path = ChromeDriverManager().install()
                print(f"ChromeDriver installed at: {driver_path}")
                
                # Create service object
                service = Service(executable_path=driver_path)
                
                # Create driver with service
                driver = webdriver.Chrome(service=service, options=chrome_options)
            else:
                # For non-Windows systems, use the default approach
                driver = webdriver.Chrome(
                    service=Service(ChromeDriverManager().install()),
                    options=chrome_options
                )
            
            # Add JavaScript to modify navigator properties to avoid detection
            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": """
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    });
                """
            })
            
            return driver
            
        except Exception as e:
            print(f"Error setting up WebDriver: {str(e)}")
            return None

    def check_weather(self, input_str: str) -> str:
        """
        Fetch the current weather condition for a city and return it.
        
        Args:
            input_str: A string in format "city_name|expected_condition"
        
        Returns:
            A string with the actual weather condition.
        """
        # Parse the input
        parts = input_str.split('|')
        if len(parts) != 2:
            return "Invalid input format. Please use: city_name|expected_condition"
        
        city, expected_condition = [p.strip() for p in parts]
        print(f"Checking weather for {city}, expecting: {expected_condition}")
        
        driver = self.setup_webdriver()
        if not driver:
            return "Failed to set up WebDriver."
        
        try:
            # Navigate to Google Weather for the specified city
            url = f"https://www.google.com/search?q=weather+{city.replace(' ', '+')}"
            print(f"Navigating to {url}...")
            driver.get(url)
            
            # Wait for page to load completely
            print("Waiting for page to load completely...")
            time.sleep(5)
            
            # Take an initial screenshot to see the page state
            initial_screenshot = f"{city.replace(' ', '_')}_initial.png"
            driver.save_screenshot(initial_screenshot)
            print(f"Initial page screenshot saved as {initial_screenshot}")
            
            # Try to extract weather using multiple approaches
            print("Attempting to extract weather information...")
            
            # Use known Google Weather element selectors
            weather_selectors = [
                (By.ID, "wob_dc"),
                (By.CSS_SELECTOR, ".wob_dc"),
                (By.XPATH, "//span[@id='wob_dc']"),
                (By.XPATH, "//div[@id='wob_dcp']/div"),
                (By.XPATH, "//div[@class='UQt4rd']"),
                (By.XPATH, "//div[contains(@class, 'VQF4g')]"),
                (By.XPATH, "//span[contains(@class, 'BBwThe')]"),
                (By.XPATH, "//div[contains(text(), '°')]/../div[1]"),
            ]
            
            actual_condition = None
            for selector_type, selector_value in weather_selectors:
                try:
                    print(f"Trying selector: {selector_type} = {selector_value}")
                    wait = WebDriverWait(driver, 3)
                    element = wait.until(EC.visibility_of_element_located((selector_type, selector_value)))
                    condition = element.text.strip()
                    if condition and len(condition) > 0:
                        actual_condition = condition
                        print(f"Found weather condition: '{actual_condition}' using {selector_value}")
                        break
                except Exception as e:
                    print(f"Selector {selector_value} failed: {str(e)[:100]}...")
            
            # Look for weather-related terms in visible text
            if not actual_condition:
                print("Trying to find weather conditions in page text...")
                try:
                    # Common weather conditions to look for
                    conditions = [
                        "Sunny", "Clear", "Partly cloudy", "Cloudy", "Rain", "Showers",
                        "Thunderstorm", "Snow", "Mist", "Fog", "Haze", "Smoke", "Dust",
                        "Drizzle", "Overcast"
                    ]
                    
                    # Get all text elements on the page
                    elements = driver.find_elements(By.XPATH, "//div[string-length(text()) > 2]")
                    
                    for element in elements:
                        text = element.text.strip()
                        # Check if any known weather condition is in this text
                        for condition in conditions:
                            if condition.lower() in text.lower():
                                actual_condition = text
                                print(f"Found likely weather text: '{actual_condition}'")
                                break
                        if actual_condition:
                            break
                except Exception as e:
                    print(f"Text search approach failed: {str(e)[:100]}...")
            
            # Try to extract from page title
            if not actual_condition:
                try:
                    title = driver.title
                    print(f"Page title: {title}")
                    if "weather" in title.lower():
                        # Google weather titles often have format "Weather Condition - Weather for City"
                        parts = title.split(" - ")
                        if len(parts) > 0:
                            actual_condition = parts[0].strip()
                            print(f"Extracted weather from title: '{actual_condition}'")
                except Exception as e:
                    print(f"Title extraction failed: {str(e)[:100]}...")
            
            # Parse the page source for weather patterns
            if not actual_condition:
                try:
                    print("Analyzing page source for weather information...")
                    page_source = driver.page_source.lower()
                    
                    # Common patterns in Google's weather widget HTML
                    patterns = [
                        "id=\"wob_dc\">([^<]+)<",
                        "class=\"BBwThe\">([^<]+)<",
                        "class=\"VQF4g\">([^<]+)<",
                        "data-local-attribute=\"weather-condition\">([^<]+)<"
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, page_source)
                        if matches and len(matches) > 0:
                            actual_condition = matches[0].strip()
                            print(f"Found condition in source: '{actual_condition}'")
                            break
                except Exception as e:
                    print(f"Source analysis failed: {str(e)[:100]}...")
            
            # Capture a final screenshot
            screenshot_path = f"{city.replace(' ', '_')}_weather.png"
            driver.save_screenshot(screenshot_path)
            print(f"Final screenshot saved as {screenshot_path}")
            
            # Final result
            if actual_condition:
                return f"Current weather in {city}: {actual_condition}. Expected: {expected_condition}"
            else:
                return f"Couldn't extract weather condition for {city}. Expected: {expected_condition}"
            
        except Exception as e:
            error_msg = f"Error occurred while checking weather: {str(e)}"
            print(error_msg)
            if driver:
                print("Taking error screenshot...")
                driver.save_screenshot(f"{city.replace(' ', '_')}_error.png")
            return error_msg
        
        finally:
            # Close the browser
            if driver:
                driver.quit()
                print("Browser closed.")

    def analyze_weather_match(self, input_str: str) -> str:
        """
        Analyze if the actual weather matches the expected condition.
        
        Args:
            input_str: A string containing the weather information to analyze
        
        Returns:
            A string with the analysis result
        """
        print(f"Analyzing weather match: {input_str}")
        
        # Extract the actual and expected conditions
        try:
            # Pattern to extract actual and expected conditions
            actual_pattern = r"Current weather in .+?: (.+?)\."
            expected_pattern = r"Expected: (.+?)(?:\.|$)"
            
            actual_match = re.search(actual_pattern, input_str)
            expected_match = re.search(expected_pattern, input_str)
            
            if not actual_match or not expected_match:
                return "Could not parse weather information correctly."
            
            actual_condition = actual_match.group(1).strip().lower()
            expected_condition = expected_match.group(1).strip().lower()
            
            print(f"Actual condition: {actual_condition}")
            print(f"Expected condition: {expected_condition}")
            
            # Check if actual contains expected or vice versa (for partial matches)
            is_match = (
                expected_condition in actual_condition or 
                actual_condition in expected_condition
            )
            
            if is_match:
                return f"✅ SUCCESS: The weather matches your expectation! Actual: '{actual_condition}', Expected: '{expected_condition}'"
            else:
                return f"❌ FAILURE: The weather doesn't match your expectation. Actual: '{actual_condition}', Expected: '{expected_condition}'"
                
        except Exception as e:
            return f"Error analyzing weather match: {str(e)}"

    def run(self, city: str, expected_condition: str) -> Dict[str, Any]:
        """
        Run the weather checker agent with the given city and expected condition.
        
        Args:
            city: The name of the city to check weather for
            expected_condition: The expected weather condition
            
        Returns:
            The result from the agent execution
        """
        query = f"Check if the weather in {city} matches the expected condition '{expected_condition}'."
        
        print(f"Weather Checker Agent Started")
        print(f"Checking if the weather in {city} is {expected_condition}...")
        print("-" * 50)
        
        result = self.agent_executor.invoke({"input": query})
        
        print("-" * 50)
        print("Weather Checker Agent Completed")
        
        return result


def main():
    """
    Main function to parse command line arguments and run the weather checker agent.
    """
    # Check if correct number of arguments provided
    if len(sys.argv) != 3:
        print("Usage: python weather_checker_agent.py <city_name> <expected_condition>")
        print("Example: python weather_checker_agent.py \"Pune\" \"Sunny\"")
        sys.exit(1)
    
    # Extract command line arguments
    city = sys.argv[1]
    expected_condition = sys.argv[2]
    
    # Create and run the agent
    agent = WeatherCheckerAgent()
    result = agent.run(city, expected_condition)
    
    # Determine if the test was successful based on the agent's output
    output = result.get("output", "")
    success = "SUCCESS" in output
    
    # Print the final result
    print("\nFinal Result:")
    print(output)
    
    # Return exit code based on whether expectation was met
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()