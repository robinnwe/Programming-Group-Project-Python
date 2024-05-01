# Import necessary libraries
import yfinance as yf                  # To fetch financial data
import pandas as pd                    # For data manipulation and presentation

# Initial information print for the user
print("You have selected the module GENERAL INFORMATION & ANALYSIS.")
print("This module provides an overview and financial analysis of publicly traded companies.")
print("\nThe module progresses as follows:\n1. Enter a ticker symbol to fetch data.\n2. Review the data presented.\n3. Choose to inspect another company or exit.")

# Function to fetch general company information
def fetch_general_info(ticker):
    """
    Fetches general information about a company using its ticker symbol.

    Args:
    ticker (str): The stock ticker symbol of the company.

    Returns:
    dict: A dictionary containing key company information.
    """
    stock = yf.Ticker(ticker)  # Create a Ticker object
    info = stock.info          # Retrieve stock information
    general_info = {
        'Name': info.get('shortName', 'N/A'),          # Company name
        'Country': info.get('country', 'N/A'),         # Country of operation
        'City': info.get('city', 'N/A'),               # City of operation
        'Industry': info.get('industry', 'N/A'),       # Industry category
        'Sector': info.get('sector', 'N/A'),           # Sector category
        'Full Time Employees': info.get('fullTimeEmployees', 'N/A')  # Number of employees
    }
    return general_info


# Function to fetch financial ratios
def fetch_financial_ratios(ticker):
    """
    Fetches financial ratios for a company.

    Args:
    ticker (str): The stock ticker symbol of the company.

    Returns:
    dict: A dictionary containing key financial ratios.
    """
    stock = yf.Ticker(ticker)
    ratios = {
        'PE Ratio': round(stock.info.get('trailingPE', 'N/A'), 2),
        'PEG Ratio': round(stock.info.get('pegRatio', 'N/A'), 2),
        'Price to Book': round(stock.info.get('priceToBook', 'N/A'), 2)
    }
    return ratios

# Main loop
while True:
    ticker = input("\nWhat company do you wish to inspect? \n\nTICKER: ")
    try:
        # Fetch and display general company information
        general_info = fetch_general_info(ticker)
        if 'Name' in general_info and general_info['Name'] != 'N/A':
            print(f"\nGENERAL INFORMATION:\n{pd.DataFrame.from_dict(general_info, orient='index')}")
        else:
            raise ValueError("Invalid ticker symbol or no data available.")

        # Fetch and display financial ratios
        ratios = fetch_financial_ratios(ticker)
        print(f"\nFINANCIAL RATIOS:\n{pd.DataFrame.from_dict(ratios, orient='index')}")

    except ValueError as e:
        # Handle exceptions by printing an error message
        print(f"\nFailed to retrieve data for {ticker}. Error: {e}")

    # Ask user if they want to inspect another company
    if input("\nDo you wish to inspect another company? (yes/no): ").lower() != 'yes':
        break
