import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import pymc as pm
import arviz as az
from tabulate import tabulate
import random
import seaborn as sns
import logging

# Suppress PyMC3 logging messages
logger = logging.getLogger("pymc")
logger.propagate = False 
logger.setLevel(logging.ERROR)

def fit_pymc(samples, success):
  '''
  Creates a PyMC model to estimate the sharpness parameter of a strategy
  over a given period of time.
  '''
  with pm.Model() as model:
    p = pm.Uniform('p', lower=0, upper=1) # Prior
    x = pm.Binomial('x', n=samples, p=p, observed=success) # Likelihood

  with model:
    inference = pm.sample(progressbar=False, chains = 4, draws = 2000)

  # Stores key variables
  mean = az.summary(inference, hdi_prob = 0.95)['mean'].values[0]
  lower = az.summary(inference, hdi_prob = 0.95)['hdi_2.5%'].values[0]
  upper = az.summary(inference, hdi_prob = 0.95)['hdi_97.5%'].values[0]
  posterior_samples = inference.posterior['p'].values.flatten()

  #print(f'  PyMC results for p: {mean} ({lower}, {upper})\n')
  return mean, [lower, upper], posterior_samples

def summarize_season_results(seasons, betting_log):
  '''
  Using the log of bets placed with a strategy, calculates the strategy's performance
  using multiple metrics for each season separately
  '''
  data = []

  for season in seasons:
    # Isolating each season
    betting_log_season = betting_log[betting_log['Season'] == season]

    # Extracting relevant variables
    bets_placed = len(betting_log_season)
    successful_bets = sum(betting_log_season['Successful bet'].values)
    season_pl = sum(betting_log_season['Profit/Loss'])
    median_odds = np.median(betting_log_season['Odds locked'].values)

    # Performing additional calculations
    profit = (season_pl/bets_placed) * 100
    RBEP = 1/median_odds
    sharpness_mean, sharpness_ci, _ = fit_pymc(bets_placed, successful_bets)

    # Storing data for each season
    data.append([season, bets_placed, successful_bets, sharpness_ci, round(sharpness_mean, 2), round(RBEP, 2), round(profit, 2)])

  # Reporting all results
  headers = ['Season', 'Bets Placed', 'Successful Bets', 'Sharpness CI', 'Sharpness Mean', 'RBEP', 'Profit (%)']
  print(tabulate(data, headers=headers, tablefmt="fancy_grid", colalign=("center",) * len(headers)))


def summarize_complete_results(betting_log):
  '''
  Using a strategy's betting log, calculates its performance through the
  entire period when it was theoretically deployed in the backtest
  '''
  data = []

  # Extracting relevant variables
  bets_placed = len(betting_log)
  successful_bets = sum(betting_log['Successful bet'].values)
  total_pl = sum(betting_log['Profit/Loss'])
  median_odds = np.median(betting_log['Odds locked'].values)

  # Performing additional calculations
  profit = (total_pl/bets_placed)*100
  RBEP = 1/median_odds
  sharpness_mean, sharpness_ci, posterior_samples = fit_pymc(bets_placed, successful_bets)

  # Reporting results
  data.append(['Whole period', bets_placed, successful_bets, sharpness_ci, round(sharpness_mean, 2), round(RBEP, 2), round(profit, 2)])
  headers = ['Season', 'Bets Placed', 'Successful Bets', 'Sharpness CI', 'Sharpness Mean', 'RBEP', 'Profit (%)']
  print(tabulate(data, headers=headers, tablefmt="fancy_grid", colalign=("center",) * len(headers)))

  return posterior_samples

def evaluate_randomness(betting_log, plot_title):
    '''
    This function creates a Monte Carlo simulation over the bets placed during backtest by a strategy to evaluate the impact of
    randomness on possible financial outcomes. It uses Numpy arrays, rather than loops, to increase the speed of each trial,
    and simulates 5000 different trials of the same bets.
    '''

    # PERFORMS SIMULATIONS WITH NUMPY
    # 1. Extracts key variables
    trials = 10000
    bets_per_trial = len(betting_log)
    total_pl = betting_log['Profit/Loss'].sum()

    probabilities = betting_log['Pinnacle OIP'].values
    odds_locked = betting_log['Odds locked'].values

    profit_in_backtest = (total_pl / bets_per_trial)*100

    # 2. Simulates random outcomes and computes profits
    random_draws = np.random.rand(trials, bets_per_trial)
    wins = random_draws < probabilities
    profits = np.where(wins, odds_locked - 1, -1)
    final_profits = (profits.sum(axis=1) / bets_per_trial)*100  # Computes mean profit per trial for all trials

    # CALCULATES KEY METRICS
    # 1. Calculates mean and 95% confidence interval
    mean_simulation = np.mean(final_profits)
    lower_bound = np.percentile(final_profits, 2.5)
    upper_bound = np.percentile(final_profits, 97.5)

    # 2. Calculates key stats compared to backtest
    p_better_than_backtest = (sum(final_profits > profit_in_backtest)/trials) * 100
    p_loss = (sum(final_profits < 0)/trials) * 100
    p_greater_than_safe_investment = (sum(final_profits > 10)/trials) * 100

    # DISPLAYS RESULTS
    #1. Creates plot for easy visualization
    plt.figure(figsize=(10, 6))
    sns.histplot(final_profits, bins=30, kde=False, color="#15616D", alpha=0.6)
    plt.axvline(mean_simulation, color='#FF7D00', linestyle='dashed', linewidth=2, label=f'Average Profit: {mean_simulation:.2f}%')
    plt.axvline(profit_in_backtest, color='#07020D', linestyle='dashed', linewidth=2, label=f'Backtest Profit: {profit_in_backtest:.2f}%')
    plt.axvline(lower_bound, color='red', linestyle='dashed', linewidth=2, label=f'95% Lower Bound: {lower_bound:.2f}%')
    plt.axvline(upper_bound, color='red', linestyle='dashed', linewidth=2, label=f'95% Upper Bound: {upper_bound:.2f}%')
    plt.xlabel('Profit (%)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Distribution of Alternative Profits: {plot_title}', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. Prints outcomes
    print('\nSummary of random effect analysis')
    print(f'  Average profit over simulations: {mean_simulation:.2f}%')
    print(f'  95% interval of profit: [{lower_bound:.2f}%, {upper_bound:.2f}%]')
    print(f'  Probability of ending better off than in the backtest: {p_better_than_backtest:.2f}%')
    print(f'  Probability of ending with a loss: {p_loss:.2f}%')
    print(f'  Probability of ending better off than safe investment: {p_greater_than_safe_investment:.2f}%')

def summarize_results(seasons, betting_log, strategy_name):
  '''
  This function gathers the call of each individual evaluation function, de-cluttering future lines of code
  '''

  print('Season-specific results')
  summarize_season_results(seasons, betting_log)

  print('\nOverall results')
  posterior_samples = summarize_complete_results(betting_log)

  print()
  evaluate_randomness(betting_log, strategy_name)

  return posterior_samples

def test_script():
    print('Script containing strategy evaluation functions were imported successfully!')
