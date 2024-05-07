# This program was written in Python programming language
# Disclaimer: This is no investment advice
# ChatGPT was used to correct and improve the program


# Importing necessary libraries for the program

# Standard Libraries
from datetime import datetime  # For handling date and time
from dateutil.relativedelta import relativedelta  # For manipulating dates with respect to different time intervals

# Data Handling and Manipulation
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis

# Financial Data and Optimization
import yfinance as yf  # For downloading financial data from Yahoo Finance
from scipy.optimize import minimize  # For optimization tasks

# Visualization
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For enhanced visualization style



##############################################
# INTRO QUESTIONS WHICH DEFINE INVESTOR TYPE #
##############################################

# Source: Bodie et al., 2024, pp. 174-175
# You start with 0 points which must be defined, so the points of the answers can be added

points = 0

# Introduction
print("Hey there, ")
print( "answer the following questions to see what type of investor you are. Do so by choosing between A, B or C. ")
print("Let's begin!")

# spacing, better layout
print()
print()

# Question 1, giving the user three different options to choose from, depending on the chosen answer, the points will be added
# \n for better layout
print("1. Just 60 days after you put money into an investment, its price falls 20%. Assuming none of the fundamentals have changed, what would you do? ")
print("\nA: Sell to avoid further worry and try something else, ") 
print("\nB: Do nothing and wait for the investment to come back, ")
print("\nC: Buy more. It was a good investment before; now it's a cheap investment, too ")
while True: 
    choice = input("\nenter your choice (A/B/C): ")
    if choice.upper () == "A":
      print("you have chosen the first answer")
      points += 0
      break
    elif choice.upper () == "B":
      print("you have chosen the second answer")
      points += 1
      break
    elif choice.upper () == "C":
      print("you have chosen the third answer")
      points += 2
      break # This will break the loop 
    
    else: 
        print("Invalid choice. Please choose either A, B or C.")

print()

# Question 2a
print("2. Now look at the previous question another way. Your investment fell 20%, but it's part of a portfolio being used to meet investment goals with three different time horizons. ")
print("\n2a. What would you do if the goal were five years away? ")
print( "\nA: Sell ")
print( "\nB: Do nothing")
print( "\nC: Buy more")
while True:
      choice = input("\nenter your choice (A/B/C): ")
      if choice.upper () == "A":
        print("you have chosen the first answer")
        points += 0
        break
      elif choice.upper () == "B":
        print("you have chosen the second answer")
        points += 1
        break
      elif choice.upper () == "C":
        print("you have chosen the third answer")
        points += 2
        break #This will break the loop
        
      else:
          print("Invalid choice. Please choose either A, B or C.")
        
print()

# Question 2b
print("2b. What would you do if the goal were 15 years away? ")
print( "\nA: Sell ")
print( "\nB: Do nothing ")
print( "\nC: Buy more")
while True:
      choice = input("\nenter your choice (A/B/C): ")
      if choice.upper () == "A":
        print("you have chosen the first answer")
        points += 0
        break
      elif choice.upper () == "B":
        print("you have chosen the second answer")
        points += 1
        break
      elif choice.upper () == "C":
        print("you have chosen the third answer")
        points += 2
        break #This will break the loop
        
      else:
          print("Invalid choice. Please choose either A, B or C.")
             
print()

# Question 2c
print("2c. What would you do if the goal were 30 years away? ")
print( "\nA: Sell ")
print( "\nB: Do nothing ")
print( "\nC: Buy more")
while True:
      choice = input("\nenter your choice (A/B/C): ")
      if choice.upper () == "A":
        print("you have chosen the first answer")
        points += 0
        break
      elif choice.upper () == "B":
        print("you have chosen the second answer")
        points += 1
        break
      elif choice.upper () == "C":
        print("you have chosen the third answer")
        points += 2
        break #This will break the loop
        
      else:
          print("Invalid choice. Please choose either A, B or C.")
                
print()

# Question 3
print("3. The price of your retirement investment jumps 25% a month after you buy it. Again, the fundamentals haven't changed. After you finish gloating, what do you do? ")
print( "\nA: Sell it and lock in your gains ")
print( "\nB: Stay put and hope for more gain ")
print( "\nC: Buy more; it could go higher")
while True:
      choice = input("\nenter your choice (A/B/C): ")
      if choice.upper () == "A":
        print("you have chosen the first answer")
        points += 0
        break
      elif choice.upper () == "B":
        print("you have chosen the second answer")
        points += 1
        break
      elif choice.upper () == "C":
        print("you have chosen the third answer")
        points += 2
        break#This will break the loop
      else:
          print("Invalid choice. Please choose either A, B or C.")
      
print()

# Question 4
print("4. You're investing for retirement, which is 15 years away. Which would you rather do? ")
print( "\nA: Invest in a money-market fund or guaranteed investment contract, giving up the possibility of major gains, but virtually assuring the safety of your principal ")
print(" \nB: Invest in a 50-50 mix of bond funds and stock funds, in hopes of getting some growth, but also giving yourself some protection in the form of steady income ")
print( "\nC: Invest in aggressive growth mutual funds whose value will probably fluctuate significantly during the year, but have the potential for impressive gains over five or 10 years ")
while True:
      choice = input("\nenter your choice (A/B/C): ")
      if choice.upper () == "A":
        print("you have chosen the first answer")
        points += 0
        break
      elif choice.upper () == "B":
        print("you have chosen the second answer")
        points += 1
        break
      elif choice.upper () == "C":
        print("you have chosen the third answer")
        points += 2
        break #This will break the loop
      else:
          print("Invalid choice. Please choose either A, B or C.")
         
print()
      
# Question 5
print("5. You just won a big prize! But which one? It's up to you. ")
print( "\nA: $2,000 in cash ")
print( "\nB: A 50% chance to win $5,000") 
print( "\nC: A 20% chance to win $15,000")
while True: 
      choice = input("\nenter your choice (A/B/C): ")
      if choice.upper () == "A":
        print("you have chosen the first answer")
        points += 0
        break
      elif choice.upper () == "B":
        print("you have chosen the second answer")
        points += 1
        break
      elif choice.upper () == "C":
        print("you have chosen the third answer")
        points += 2
        break #This will break the loop
      else:
          print("Invalid choice. Please choose either A, B or C.")        

print()

# Question 6
print("6. A good investment opportunity just came along. But you have to borrow money to get in. Would you take out a loan? ")
print( "\nA: Definitely not ")
print( "\nB: Perhaps ")
print( "\nC: yes")
while True:
      choice = input("\nenter your choice (A/B/C): ")
      if choice.upper () == "A":
        print("you have chosen the first answer")
        points += 0
        break
      elif choice.upper () == "B":
        print("you have chosen the second answer")
        points += 1
        break
      elif choice.upper () == "C":
        print("you have chosen the third answer")
        points += 2
        break #This will break the loop
      else:
          print("Invalid choice. Please choose either A, B or C.")
              
print()
      
# Question 7
print("7. Your company is selling stock to its employees. In three years, management plans to take the company public. Until then, you won't be able to sell your shares and you will get no dividends. But your investment could multiply as much as 10 times when the company goes public. How much money would you invest? ")
print( "\nA: None ") 
print( "\nB: Two months' salary ")
print( "\nC: Four months' salary")
while True:
      choice = input("\nenter your choice (A/B/C): ")
      if choice.upper () == "A":
        print("you have chosen the first answer")
        points += 0
        break
      elif choice.upper () == "B":
        print("you have chosen the second answer")
        points += 1
        break
      elif choice.upper () == "C":
        print("you have chosen the third answer")
        points += 2
        break #This will break the loop
      else:
          print("Invalid choice. Please choose either A, B or C.")
         
print()
      
# Question 8 (added questions to the form, only choosing between A and B)
print("8. Make a choice: ")
print( "\nA: a probability of 25% for a profit of CHF 30'000 ")
print("\nB: a probability of 20% for a profit of CHF 45'000")
while True:
      choice = input("\nenter your choice (A/B): ")
      if choice.upper () == "A":
        print("you have chosen the first answer")
        points += 1
        break
      elif choice.upper () == "B":
        print("you have chosen the second answer")
        points += 2
        break #This will break the loop
      else:
          print("Invalid choice. Please choose either A or B.")
               
print()

# Question 9 (added questions to the form, only choosing between A and B)
print("9. Make a choice: ")
print( "\nA: Profit of CHF 50'000 with a probability of 80% ")
print( "\nB: Profit of CHF 30'000 with a probability of 100%")
while True:
      choice = input("\nenter your choice (A/B): ")
      if choice.upper () == "A":
        print("you have chosen the first answer")
        points += 1
        break
      elif choice.upper () == "B":
        print("you have chosen the second answer")
        points += 2
        break #This will break the loop
      else:
          print("Invalid choice. Please choose either A or B.")
              
# Spacing, Layout
print()
print()
print()

# Introduction of the results
print("thank you for your answers")
print()

# telling the user the amount of collected points
print("you have collected: ", points, "points")

# depending on the amout of points, the user will be assigned a type of investor
print("your investment type is: ")
if 0 <= points <= 3:
    A = 8
    print( "Type A = 8, extremely conservative")
if 4 <= points <= 7:
    A = 7
    print("Type A = 7, conservative investor")
if 8 <= points <= 11 :
    A = 6
    print("Type A = 6, conservative to moderate investor ")
if 12 <= points <= 15 :
    A = 5
    print("Type A = 5, moderate to aggressive investor")
if 16  <= points <= 19 :
    A = 4
    print("Type A = 4, aggressive investor")
if 20  <= points <= 22 :
    A = 3
    print(" Type A = 3, extremely aggressive investor")




########################################################
# STOCK PICKER: GENERAL INFORMATION AND RATIO ANALYSIS #
########################################################


# Initial information print for the user
print("\n\nYou are now in the module GENERAL INFORMATION & ANALYSIS.")
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



#####################################
# VISUALIZATION AND CHARTING MODULE #
#####################################



# In this module we ask the user to input tickers from the stocks he wants tu use
# We then download the necessary stock information from yfinance

print("\n\nYou are now in the module VISUALIZATION AND CHARTING.")
print("This module provides graphical information on the stock of") 
print("desired companies such as monthly returns, standard deviation") 
print("and stock prices.")
print("\nThe module progresses as follows:\n1. Enter multiple ticker symbols to fetch data.\n2. Review the data presented with plots and graphs.")


# Define function to check if a ticker is valid using yfinance.
def is_valid_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return "symbol" in info and info["symbol"] == ticker
    except Exception:
        return False

# Define function to get the input from user and control for user input
def get_valid_tickers():
    while True:
        tickers_input = input("Enter multiple tickers separated by commas (e.g., AFX.DE, NESN.SW, LIN, MDLZ): ").strip()
        if tickers_input and "," in tickers_input:
            # Split the input string into individual tickers
            tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
                        
            if len(tickers) < 3:
                print("Please enter at least three tickers.")
                continue
                       
            # Check if all tickers are valid
            if all(is_valid_ticker(ticker) for ticker in tickers):
                return tickers
            else:
                print("One or more tickers are invalid. Please provide valid tickers separated by commas.")
        else:
            print("Invalid input. Please enter tickers separated by commas.")

              

# Get the user input 
tickers = get_valid_tickers()

# Download data for the specified tickers
multpl_stocks = yf.download(tickers,
                            start="2019-05-04",
                            end="2024-04-26")


###################################################

# In this part we plot the stock infos of the inputed tickers and create a multiple figure
# We plot the stock over 5 years and the respective standard deviation 

# Assuming you have already obtained multpl_stocks data based on user input

# Create a new figure
fig = plt.figure()

# Determine the number of subplots based on the number of tickers
num_plots = len(tickers)
fig, axes = plt.subplots(num_plots, 1, figsize=(8, 12))

# Iterate through tickers and create subplots
for ax, ticker in zip(axes, tickers):
    ax.plot(multpl_stocks['Close'][ticker])
    ax.set_title(ticker)
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()

# Calculate the monthly returns for each ticker
monthly_returns = multpl_stocks['Close'].pct_change()



# Get the standard deviation of the inputed stocks
# Assuming std_dev is a pandas Series containing standard deviations of the inputted stocks
std_dev = monthly_returns.std()

# Filter the standard deviations based on inputted tickers
filtered_std_dev = {ticker: std_dev.loc[ticker] for ticker in tickers if ticker in std_dev}

# Extract the stocks and their respective standard deviations
stocks = list(filtered_std_dev.keys())
std_values = list(filtered_std_dev.values())

# Plot the standard deviations
#We also label x and y axes, legend our graph and use layout
plt.figure(figsize=(8, 6))
plt.bar(stocks, std_values, color='skyblue')
plt.title('Standard Deviation of Monthly Returns')
plt.xlabel('Stocks')
plt.ylabel("Standard Deviation (in%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

################################################

# In this part we calculate the daily and monthly returns of the stocks 
multpl_stock_daily_returns = multpl_stocks['Adj Close'].pct_change()
multpl_stock_monthly_returns = multpl_stocks['Adj Close'].resample('M').ffill().pct_change()

###############################################

# In this part we plot the above calculated returns 
fig = plt.figure()
(multpl_stock_monthly_returns + 1).cumprod().plot()
plt.show()
# Add labels
plt.xlabel('Date')
plt.ylabel("Cumulative Returns (in%)")

# Add a legend
plt.legend(loc='upper left')

# Show the plot
plt.show()
#################################################

# In this part we calculate the 50 and 200 days moving averages of our stocks 

# we use this function to download the info over a 300 days interval 
multpl_stocks = {ticker: yf.download(ticker, period='300d') for ticker in tickers} 

# We plot the subplots with the a sharing x-axes based on our tickers
fig, axes = plt.subplots(nrows=len(tickers), ncols=1, figsize=(10, 15))

# Here we plot the closing prices in black, 50 days MA in blue and 200 days MA in red
# We also label x and y axes and legend our graph   
for ax, (ticker, df) in zip(axes, multpl_stocks.items()):
    df['Close'].plot(ax=ax, label='Closing Price', color='black')
    df['50_MA'] = df['Close'].rolling(window=50).mean()
    df['50_MA'].plot(ax=ax, label='50-Day MA', color='blue')
    df['200_MA'] = df['Close'].rolling(window=200).mean()
    df['200_MA'].plot(ax=ax, label='200-Day MA', color='red')
    
    ax.set_title(ticker)
    ax.legend()
# Here we do some layout to make it more clear
plt.tight_layout()
plt.savefig("simplefinance.png", dpi=200)
plt.show()



###################################
# PORTFOLIO OPTIMALIZATION MODULE #
###################################


# Step 1: Risk/Return optimization of the Risky Asset (portfolio of stocks)
# Source: O’Connell, R. (2023). Portfolio Optimization in Python: Boost Your Financial Performance. Youtube

# Initial information print for the user
print("\n\nIn the following, the return of the risky asset for a given level of risk will be maximized using the Sharpe Ratio.")
print("The tickers entered above are used to compose the risky asset. Each ticker will be assigned the optimal weight")
print("In a second step, the investors degree of risk aversion from the first module is incorporated.")
print("This is done to determine the amount of wealth (%) that should be invested in the risk-free asset given the risk preferences.\n\n")

# Define the stocks that are of interest
stocks = tickers
end_date = datetime.today()  # Set the end date for the data to today
start_date = end_date - relativedelta(years=10)  # Set the start date for the data to 10 years ago from today

# Initialize an empty DataFrame to store the adjusted close prices of the stocks
adj_close_df = pd.DataFrame()
# Loop through each stock symbol to download its data
for stock in stocks:
    try:
        # Download stock data from Yahoo Finance from start_date to end_date
        data = yf.download(stock, start=start_date, end=end_date)
        # Extract the 'Adjusted Close' prices and add them to the DataFrame
        adj_close_df[stock] = data["Adj Close"]
    except Exception as e:
        # If there's an error during download, print the error
        print(f"Error downloading {stock}: {e}")
        
# Input loop to capture and validate the user-provided risk-free rate
while True:
    try:
        # Request input from the user for the risk-free rate in decimal form
        risk_free_rate = float(input("\nPlease insert the risk-free rate you would like to use (e.g., 0.04 for 4%): "))
        # Validate if the input rate is within the logical range (-1, 1) -> risk-free rate might be negative
        if not -1 < risk_free_rate < 1:
            print("Please enter a rate between -1 and 1.")
            continue  # Restart the loop if input is not valid
        break  # Break the loop if the input is valid
    except ValueError:
        # Handle the case where input cannot be converted to float
        print("Invalid input. Please enter a valid risk-free rate in decimal form (e.g., 0.04).")

# Calculate log returns of the stocks
log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()  # Divide one day's stock price by the previous day's stock price, drop NA values

# Calculate the covariance matrix of the log returns to understand how stocks move together
cov_matrix = log_returns.cov() * 252  # Annualize the covariance by multiplying by the number of trading days

# Define a function to calculate the standard deviation of portfolio returns, which is a measure of risk
def standard_deviation(weights, cov_matrix):
    return np.sqrt(weights.T @ cov_matrix @ weights)

# Define a function to calculate the expected return of the portfolio
def expected_return(weights, log_returns):
    return np.sum(log_returns.mean() * weights) * 252  # Annualize by multiplying by the number of trading days

# Define a function to calculate the Sharpe ratio, which measures the performance of an investment compared to a risk-free asset
def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

# Define the objective function to be minimized (negative Sharpe Ratio)
def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

# Constraints to ensure that the sum of portfolio weights is 1
constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
# Set bounds for the weights of each stock in the portfolio (no short selling, max 40% allocation to any stock)
bounds = [(0, 0.4) for _ in range(len(stocks))]
# Initial guess for the weights
initial_weights = np.array([1/len(stocks)] * len(stocks))

# Perform the optimization to maximize the Sharpe Ratio (minimize the negative Sharpe Ratio)
optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, risk_free_rate),
                             method="SLSQP", constraints=constraints, bounds=bounds)

# Retrieve the optimal weights from the optimization results
optimal_weights = optimized_results.x

# Porfolio Performance Metrics Calculation
optimal_portfolio_return = expected_return(optimal_weights, log_returns)  # Calculate the expected return of the portfolio  
optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)  # Compute the portfolio's volatility (standard deviation)
optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)  # Determine the Sharpe Ratio of the portfolio

# Check if the optimization was successful, show performance metrics and weigths allocated to each stock
if optimized_results.success:
    print("\n\nThe following weights were allocated to each stock (in decimals):")
    for stock, weight in zip(stocks, optimal_weights): 
        print(f"{stock}: {weight: .4f}")
    
    # Visualize the optimal portfolio weights for each stock
    plt.figure(figsize=(10, 6))
    plt.bar(stocks, optimal_weights)  # Create a bar chart
    plt.xlabel("Stocks")  # Label for the x-axis
    plt.ylabel("Optimal Weights")  # Label for the y-axis
    plt.title("Optimal Portfolio Weights")  # Title of the chart
    plt.show()  # Display the chart 

    print("Risky Asset:")     
    print(f"\n\nThe Expected Annual Return of the Risky Asset is: {optimal_portfolio_return:.4f}")
    print(f"The Expected Volatility of the Risky Asset is: {optimal_portfolio_volatility:.4f}")
    print(f"The Optimal Sharpe Ratio of the Risky Asset is: {optimal_sharpe_ratio:.4f}")
else:
    print("\n\nOptimization did not converge.")


# Step 2: Implement the coefficient or risk aversion to determine the optimal allocation in the risky and risk-free asset
# Source: Bodie et al., 2014, p. 182

# Initialize variables for the expected return and standard deviation (volatility) of the risky asset
risky_asset_return = optimal_portfolio_return 
risky_asset_sd = optimal_portfolio_volatility

# Here we use the coefficient 'A' of risk aversion from the first module

# Calculate the optimal weight for investment in the risky asset based on utility maximization
# where utility U = risk_free_rate + weight_risky_asset * (E(r) - rf) - 0.5 * A * sd_risky_asset**2 * weight_risky_asset**2
# Derive utility function, set to zero and solve for weight_risky_asset
weight_risky_asset = ((risky_asset_return - risk_free_rate) / (A * risky_asset_sd**2)) * 100  

# Ensure that the calculated weight for the risky asset does not exceed 100%
if weight_risky_asset > 100:
    weight_risky_asset = 100  # Cap the investment at 100% of total wealth

# Calculate the remaining weight for the risk-free asset
weight_risk_free_asset = 100 - weight_risky_asset

# Calculate the return and standard deviation of the two-asset portfolio
portfolio_return = weight_risky_asset * risky_asset_return + weight_risk_free_asset * risk_free_rate
portfolio_sd = weight_risky_asset * risky_asset_sd

# Print the optimal allocations for both the risky and risk-free assets
print(f"\n\nGiven your degree of risk aversion, the optimal percentage of your wealth allocated in the Risky Asset is: {weight_risky_asset:.2f} %.")
print(f"Consequently, the optimal percentage of your wealth allocated in the Risk-Free Asset is: {weight_risk_free_asset:.2f} %.")

# Print portfolio return and standard deviation
print(f"\n\nYour final portfolio will yield {portfolio_return:.2f} % in return and has a volatility of {portfolio_sd:.2f} %.")

# Print Farewell message
print("\n\nThank you very much for using our program.")

