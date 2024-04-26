# Import Required Libraries

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize
import math


# Section 1: Define Tickers and Time Range
tickers = ["AFX.DE", "NESN.SW", "LIN"]

# Set the end date to today
end_date = datetime.today() # End date is today

# Set the start date to 15 years ago
start_date = end_date - timedelta(days = 15*365) # Set time range to 15 years
print(start_date)


# Section 2: Download Adjusted Close Prices

# Create an empty DataFrame to store the adjusted close prices
# Portfolio Optimization would underweigh a stock that pays more dividends with closed prices
adj_close_df = pd.DataFrame()

# Download the close prices for each ticker
for ticker in tickers:
    data = yf.download(ticker, start = start_date, end = end_date)
    adj_close_df[ticker] = data["Adj Close"]

print(adj_close_df)


# Section 3: Calculate Lognormal Returns

# Calculate the lognormal returns for each ticker
log_returns = np.log(adj_close_df/adj_close_df.shift(1)) # Divide one day's price by the previous day's price
log_returns = log_returns.dropna() # Check for missing values


# Section 4: Calculate Covariance Matrix
# Covariance Matrix measures the total risk of the total portfolio; each Asset has a certain covariance and a certain correlation with the other assets; calulate standard deviation in the obvious way possible
cov_matrix = log_returns.cov()*252

print(cov_matrix)


# Section 5: Define Porfolio Performance Metrics

# Calculate the portfolio
def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

# Calculate the expected return
def expected_return (weights, log_returns):
    return np.sum(log_returns.mean()*weights)*252

# Calculate the Sharpe Ratio
def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return(expected_return(weights, log_returns) - risk_free_rate) / standard_deviation (weights, cov_matrix)


# Section 6: Portfolio Optimization

# Get the 10 year risk-free rate for treasury bonds (American)
from fredapi import Fred
fred = Fred(api_key = "738b3333bab61b45fc240ccce553da96")
ten_year_treasury_rate = fred.get_series_latest_release("GS10") / 100


# Set the risk-free rate
risk_free_rate = ten_year_treasury_rate.iloc[-1] 
print(risk_free_rate)

# Define the function to minimize (negative Sharpe Ratio)

# Takes the Sharpe Ratio and gets the negative vale of that function
def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

# Set the constraints and bounds
# Constraints in this context are conditions that must be met by the solution during the optimization process.
# In this case, the contraint s that hte sum of all portfolio weigths must be equal to 1.
# The constraints vairable is a dictinoary with two keys: "type" and "fun". 
# "Type" is set to "eq", which means "equality constraint", and "fun" is assigned the function check_sum, which checks if the sum of the portfolio weights equals 1.
# Bounds are the limits placed on the variales during the optimization process. In this case, the variables ar eth portfolio weights, and each weight should be betwen 0 and 1.

constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

# This means that we cannot go short in any of these assets and that one stock cannot take more than 50 % of our portfolio
bounds = [(0, 0.5) for _ in range(len(tickers))]

# Set the initial weights

initial_weights = np.array([1/len(tickers)]*len(tickers))
print(initial_weights)

# Optimize the weights to maximize Sharpe ratio

# SLSQP stands for Sequential Least Squares Quadratic, which is a numerical optimization technique suitable for solving nonlinear optimization problems with contraints 
optimized_results = minimize(neg_sharpe_ratio, initial_weights, args =(log_returns, cov_matrix, risk_free_rate), method = "SLSQP", constraints= constraints, bounds=bounds)

# Get the optimal weights
optimal_weights = optimized_results.x


# Section 7: Analyze the Optimal Portfolio

# Display analytics of the optimal portfolio
print("Optimal Weights:")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight: .4f}")

print()

optimal_portfolio_return = expected_return(optimal_weights, log_returns)
optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)

print(f"Expected Annual Return: {optimal_portfolio_return:.4f}")
print(f"Expected Volatility: {optimal_portfolio_return:.4f}")
print(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")

import matplotlib.pyplot as plt

# Create a bar chart of the optimal weights
plt.figure(figsize=(10, 6))
plt.bar(tickers, optimal_weights)

# Add labels and a title
plt.xlabel("Assets")
plt.ylabel("Optimal Weights")
plt.title("Optimal Portfolio Weights")

# Display the chart
plt.show()













































