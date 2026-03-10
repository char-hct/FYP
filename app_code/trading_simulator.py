"""
Trading Simulator Module
Simulates buying/selling Bitcoin based on model predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime


def simulate_trading(price_df, predictions_df, initial_capital=100000, transaction_cost=0.001):
    """
    Simulate trading strategy based on model predictions.
    
    Parameters:
    -----------
    price_df : pd.DataFrame
        DataFrame with columns: 'DATE', 'Close', 'Open', etc.
    predictions_df : pd.DataFrame
        DataFrame with columns: 'Published' (or 'DATE'), 'Predicted_Label' (1=buy, -1=sell, 0=hold)
    initial_capital : float
        Starting capital in USD
    transaction_cost : float
        Transaction cost as percentage (0.001 = 0.1%)
    
    Returns:
    --------
    dict : Simulation results with portfolio history and final metrics
    """
    
    # Prepare price data
    price_df = price_df.copy()
    if 'DATE' in price_df.columns:
        price_df['date'] = pd.to_datetime(price_df['DATE']).dt.date
    else:
        price_df['date'] = pd.to_datetime(price_df.index).dt.date
    
    # Prepare predictions
    pred_df = predictions_df.copy()
    if 'Published' in pred_df.columns:
        pred_df['date'] = pd.to_datetime(pred_df['Published']).dt.date
    elif 'DATE' in pred_df.columns:
        pred_df['date'] = pd.to_datetime(pred_df['DATE']).dt.date
    elif 'date' in pred_df.columns:
        # Date column already exists
        pred_df['date'] = pd.to_datetime(pred_df['date']).dt.date if not isinstance(pred_df['date'].iloc[0], (pd.Timestamp, type(None))) else pred_df['date']
    else:
        # Handle index case
        idx_dates = pd.to_datetime(pred_df.index)
        if isinstance(idx_dates, pd.DatetimeIndex):
            pred_df['date'] = idx_dates.date
        else:
            pred_df['date'] = idx_dates.dt.date
    
    # Initialize portfolio
    cash = initial_capital
    btc_holdings = 0.0
    portfolio_history = []
    
    # Track unique dates and their predictions
    price_dates = sorted(price_df['date'].unique())
    pred_dict = {}
    
    # Build prediction dictionary (most recent prediction per date)
    for _, row in pred_df.iterrows():
        date = row['date']
        # Use AI_Predicted_Label or Predicted_Label
        pred = row.get('Predicted_Label', 0)
        if pd.isna(pred):
            pred = 0
        pred_dict[date] = int(pred)
    
    # Simulate trading day by day
    for current_date in price_dates:
        # Get current price
        day_prices = price_df[price_df['date'] == current_date]
        if day_prices.empty:
            continue
        
        current_price = float(day_prices['Close'].iloc[0])
        
        # Get prediction for this date
        prediction = pred_dict.get(current_date, 0)
        
        # Execute trading logic
        actual_action = 0  # 0=HOLD by default
        if prediction == 1:  # BUY signal
            if cash > 0:
                # Buy as much Bitcoin as possible with remaining cash
                amount_to_buy = cash / current_price
                transaction_fee = amount_to_buy * current_price * transaction_cost
                amount_to_buy_after_fee = (cash - transaction_fee) / current_price
                btc_holdings += amount_to_buy_after_fee
                cash -= (amount_to_buy_after_fee * current_price + transaction_fee)
                actual_action = 1  # Actually bought
            # else: signal was BUY but no cash → actual_action stays 0
        
        elif prediction == -1:  # SELL signal
            if btc_holdings > 0:
                # Sell all Bitcoin
                sale_amount = btc_holdings * current_price
                transaction_fee = sale_amount * transaction_cost
                cash += (sale_amount - transaction_fee)
                btc_holdings = 0.0
                actual_action = -1  # Actually sold
            # else: signal was SELL but no BTC → actual_action stays 0
        
        # Calculate portfolio value
        portfolio_value = cash + (btc_holdings * current_price)
        
        portfolio_history.append({
            'date': current_date,
            'price': current_price,
            'prediction': prediction,
            'actual_action': actual_action,
            'btc_holdings': btc_holdings,
            'cash': cash,
            'portfolio_value': portfolio_value
        })
    
    # Calculate metrics
    if portfolio_history:
        # Add BTC value over time for each date (buy-and-hold strategy)
        first_price = portfolio_history[0]['price']
        for item in portfolio_history:
            btc_value_over_time = (initial_capital / first_price) * item['price']
            item['btc_value_over_time'] = btc_value_over_time
        
        final_portfolio = portfolio_history[-1]
        final_value = final_portfolio['portfolio_value']
        profit_loss = final_value - initial_capital
        roi = (profit_loss / initial_capital) * 100
        
        # Calculate buy-and-hold benchmark
        last_price = portfolio_history[-1]['price']
        btc_bought_at_start = initial_capital / first_price
        buy_hold_value = btc_bought_at_start * last_price
        buy_hold_profit = buy_hold_value - initial_capital
        buy_hold_roi = (buy_hold_profit / initial_capital) * 100
        
        results = {
            'portfolio_history': pd.DataFrame(portfolio_history),
            'final_value': final_value,
            'initial_capital': initial_capital,
            'profit_loss': profit_loss,
            'roi_percent': roi,
            'final_btc_holdings': btc_holdings,
            'final_cash': cash,
            'num_trades': sum(1 for h in portfolio_history if h['prediction'] != 0),
            'buy_hold_value': buy_hold_value,
            'buy_hold_roi': buy_hold_roi,
            'strategy_outperformance': roi - buy_hold_roi,
            'price_range': {
                'min': min(h['price'] for h in portfolio_history),
                'max': max(h['price'] for h in portfolio_history),
                'start': portfolio_history[0]['price'],
                'end': portfolio_history[-1]['price']
            }
        }
    else:
        results = {
            'portfolio_history': pd.DataFrame(),
            'final_value': initial_capital,
            'initial_capital': initial_capital,
            'profit_loss': 0,
            'roi_percent': 0,
            'final_btc_holdings': 0,
            'final_cash': initial_capital,
            'num_trades': 0,
            'buy_hold_value': initial_capital,
            'buy_hold_roi': 0,
            'strategy_outperformance': 0,
            'error': 'No data available for simulation'
        }
    
    return results


def simulate_multiple_models(price_df, predictions_dict, initial_capital=100000):
    """
    Simulate trading for multiple models and compare results.
    
    Parameters:
    -----------
    price_df : pd.DataFrame
        Price data
    predictions_dict : dict
        Dictionary of {model_name: predictions_df}
    initial_capital : float
        Starting capital
    
    Returns:
    --------
    dict : Simulation results for all models
    """
    results = {}
    for model_name, pred_df in predictions_dict.items():
        try:
            results[model_name] = simulate_trading(price_df, pred_df, initial_capital)
        except Exception as e:
            results[model_name] = {'error': str(e)}
    
    return results
