# AI-Lab-Challenge
A Calfus AI Lab Challenge Project.

## Problem Statement
Many users perform daily weather checks manually to plan their day. Weather Check Agent automates this routine by:

- Accepting a city name (e.g., "New York") and an expected weather condition (e.g., "Sunny") as input.
- Navigating to Google’s weather search results for that city.
- Extracting the current weather condition from the page.
- Comparing the actual weather condition with the expected condition and outputting whether they match.
  
This simple agentic AI implementation using Llama3 reduces manual effort and provides a quick way to verify weather conditions each day.

## Features
- Automated Weather Retrieval: Opens a browser and fetches the current weather for a given city.
- Validation: Compares the extracted weather condition with the expected condition.
- Command-Line Interface: Run the script with a single command.
- Simple & Extendable: A basic framework that can be adapted for other browser automation tasks.

## Setup & Execution
1. Setting Up Ollama (Download Ollama and install)
- Pull the llama3 model
  ```bash
  ollama pull llama3
  ```

- Start the Ollama Server
  ```bash
  ollama serve
  ```

**NOTE: Leave this terminal open while using the agent**

2. Install Python Packages
- Open new terminal and install all required packages
  ```bash
  pip install selenium webdriver-manager langchain langchain-community ollama
  ```

3. Run the Agent and script
- Open a new terminal window or use any Code editor like VS Code
- Navigate to the directory where the script is saved
- Run the script using the following command
  ```bash
  python weather_check_agent.py "New York" "Cloudy"
  ```

## Check Output
- The agent will display its thinking process and the actions it's taking
- It will navigate to Google, search for the weather, and extract the condition
- It will compare the actual weather with your expected condition
- It will return a success/failure message based on the comparison
- Screenshots will be saved in the current directory

## Contact
For questions or suggestions, please feel free to contact: ruchir.kulkarni@calfus.com
Thanks.






