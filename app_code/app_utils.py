# --- Helper: get week range (Saturday to Friday) for a given Friday date ---
def get_week_range_from_friday(friday_date):
    """
    Given a Friday date (as string or pd.Timestamp), return (start_date, end_date) where start_date is the previous Saturday and end_date is the Friday.
    """
    import pandas as pd
    if not isinstance(friday_date, pd.Timestamp):
        friday_date = pd.to_datetime(friday_date)
    start_date = friday_date - pd.Timedelta(days=6)
    return start_date.date(), friday_date.date()
# --- Combined raw and transformed data loader for website ---
def get_weekly_data_combined(week_date="2025-08-29"):
    """
    Return both raw and transformed (normalized) weekly data for the given week.
    Returns a dict: {'raw': ..., 'transformed': ...}
    """
    raw = get_latest_raw_weekly_data(week_date)
    # Transformed data
    transformed, _ = get_newest_week_data() if week_date == "2025-08-29" else (None, None)
    # If you want to support arbitrary week_date for transformed, you can add logic to load that specific week from the transformed files
    # Add week range info for display, and label as 'Latest Week'
    week_start, week_end = get_week_range_from_friday(week_date)
    return {
        'raw': raw,
        'transformed': transformed,
        'week_range': {'start': str(week_start), 'end': str(week_end)},
        'label': f"Latest Week ({week_end})"
    }
# --- Raw weekly data loader for website ---
def get_latest_raw_weekly_data(week_date="2025-08-29"):
    """
    Load the raw weekly data for BTC, macro, and more variables for a given week (default: 2025-08-29 for testing).
    Returns a dict with keys: btc, macro, more, each containing a dict of column:value for that week.
    """
    import pandas as pd
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_dir = BASE_DIR / "data" / "cleaned_v2"
    result = {}
    
    # Convert week_date to datetime for comparison
    week_date_dt = pd.to_datetime(week_date) if isinstance(week_date, str) else week_date
    
    # BTC
    btc_path = data_dir / "BTC-USD_weekly.csv"
    if btc_path.exists():
        btc_df = pd.read_csv(btc_path)
        btc_df['DATE'] = pd.to_datetime(btc_df['DATE'])
        row = btc_df[btc_df['DATE'].dt.date == week_date_dt.date()]
        if not row.empty:
            result['btc'] = row.iloc[0].to_dict()
        else:
            result['btc'] = btc_df.iloc[-1].to_dict()
    
    # Macro (uses 'Unnamed: 0' as date column)
    macro_path = data_dir / "macro" / "macro_fri.csv"
    if macro_path.exists():
        macro_df = pd.read_csv(macro_path)
        macro_df['Unnamed: 0'] = pd.to_datetime(macro_df['Unnamed: 0'])
        row = macro_df[macro_df['Unnamed: 0'].dt.date == week_date_dt.date()]
        if not row.empty:
            result['macro'] = row.iloc[0].to_dict()
        else:
            result['macro'] = macro_df.iloc[-1].to_dict()
    
    # More (uses 'Unnamed: 0' as date column)
    more_path = data_dir / "more" / "more_fri.csv"
    if more_path.exists():
        more_df = pd.read_csv(more_path)
        more_df['Unnamed: 0'] = pd.to_datetime(more_df['Unnamed: 0'])
        row = more_df[more_df['Unnamed: 0'].dt.date == week_date_dt.date()]
        if not row.empty:
            result['more'] = row.iloc[0].to_dict()
        else:
            result['more'] = more_df.iloc[-1].to_dict()
    
    return result
def get_latest_raw_feature_values(feature_names):
    """
    Given a list of transformed feature names, fetch the latest available raw value for each.
    Returns: dict {feature_name: latest_raw_value}
    """
    import re
    raw_data = {}
    # Helper to extract base name and lag from feature name
    def parse_feature_name(name):
        m = re.match(r'(btc|macro|more)__(.+?)(?:__lag(\d+))?$', name)
        if m:
            group, base, lag = m.groups()
            return group, base, int(lag) if lag else None
        return None, None, None

    # Load raw macro/more/btc dicts for the fixed week (2025-08-29)
    from app_code.app_utils import get_weekly_data_combined
    raw_dict = get_weekly_data_combined("2025-08-29")['raw']
    macro_raw = raw_dict.get('macro', {})
    more_raw = raw_dict.get('more', {})
    btc_raw = raw_dict.get('btc', {})

    for feat in feature_names:
        group, base, lag = parse_feature_name(feat)
        value = None
        if group == 'macro' and base:
            value = macro_raw.get(base, None)
        elif group == 'more' and base:
            value = more_raw.get(base, None)
        elif group == 'btc' and base:
            value = btc_raw.get(base, None)
        raw_data[feat] = value
    return raw_data
"""
Utility functions for Streamlit app
- Load newest week data
- Load trained models
- Make predictions
- Load metrics
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PERIOD_START = "2020-09-01"
# Set PERIOD_END to last COMPLETED week (last Friday)
from datetime import datetime, timedelta
today = datetime.today()
# If today IS Friday, use PREVIOUS Friday (last week's Friday is completed)
days_since_friday = (today.weekday() - 4) % 7
if days_since_friday == 0:
    # Today IS Friday, use PREVIOUS Friday
    last_friday = today - timedelta(days=7)
else:
    # Use the most recent completed Friday
    last_friday = today - timedelta(days=days_since_friday)
PERIOD_END = last_friday.strftime("%Y-%m-%d")
LAGS = (1, 2, 3, 4, 5, 6)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models" / "latest"
METRICS_DIR = BASE_DIR / "metrics"

def read_csv(path, index_col=0):
    """Read CSV with flexible date parsing for the index"""
    df = pd.read_csv(path, index_col=index_col)
    # Try to parse index as datetime with flexible format
    try:
        df.index = pd.to_datetime(df.index, format='mixed', dayfirst=False, errors='coerce')
    except:
        try:
            df.index = pd.to_datetime(df.index, errors='coerce')
        except:
            pass  # Keep original index if parsing fails
    return df

def get_newest_week_data():
    """
    Extract newest week data from all available sources
    Returns a dict with all features for the latest week
    """
    try:
        # Load BTC data (main time series)
        btc_path = DATA_DIR / "trans" / "cleaned_v1" / "BTC-USD_weekly_transformed.csv"
        if not btc_path.exists():
            return None
        
        btc_df = read_csv(btc_path).loc[PERIOD_START:PERIOD_END].sort_index()
        newest_date = btc_df.index[-1]
        newest_week = btc_df.loc[[newest_date]].copy()
        
        # Extract just the features (not target)
        features = {}
        features['date'] = newest_date
        features['btc_data'] = newest_week.to_dict('records')[0] if len(newest_week) > 0 else {}
        
        # Load macro data
        macro_path = DATA_DIR / "trans" / "cleaned_v1" / "macro" / "macro_avg_transformed.csv"
        if macro_path.exists():
            macro_df = read_csv(macro_path).loc[PERIOD_START:PERIOD_END].sort_index()
            if newest_date in macro_df.index:
                features['macro_data'] = macro_df.loc[[newest_date]].to_dict('records')[0]
        
        # Load more data (market metrics)
        more_path = DATA_DIR / "trans" / "cleaned_v1" / "more" / "more_avg_transformed.csv"
        if more_path.exists():
            more_df = read_csv(more_path).loc[PERIOD_START:PERIOD_END].sort_index()
            if newest_date in more_df.index:
                features['more_data'] = more_df.loc[[newest_date]].to_dict('records')[0]
        
        # Load sentiment data
        sentiment_path = DATA_DIR / "sentiment" / "weekly_sentiment.csv"
        if sentiment_path.exists():
            sentiment_df = pd.read_csv(sentiment_path, parse_dates=["date"], dayfirst=True)
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], errors='coerce')
            newest_sentiment = sentiment_df[sentiment_df['date'].dt.date == newest_date.date()]
            if not newest_sentiment.empty:
                features['sentiment_data'] = newest_sentiment.iloc[-1].to_dict()
        
        return features, newest_date
    
    except Exception as e:
        print(f"Error loading newest week data: {e}")
        return None, None

def get_newest_week_df():
    """
    Get newest week as a single row DataFrame (TRANSFORMED - normalized)
    Ready for model prediction
    """
    try:
        btc_path = DATA_DIR / "trans" / "cleaned_v1" / "BTC-USD_weekly_transformed.csv"
        if not btc_path.exists():
            return None, None
        
        btc_df = read_csv(btc_path).loc[PERIOD_START:PERIOD_END].sort_index()
        newest_date = btc_df.index[-1]
        newest_week_df = btc_df.loc[[newest_date]].copy()
        
        return newest_week_df, newest_date
    except Exception as e:
        print(f"Error getting newest week DataFrame: {e}")
        return None, None

def get_newest_week_raw_ohlcv():
    """
    Get newest week raw OHLCV data from daily BTC prices
    Returns weekly aggregation with Close, High, Low, Open, Volume
    """
    try:
        btc_daily_path = DATA_DIR / "BTC-USD.csv"
        if not btc_daily_path.exists():
            return None, None
        
        # Load daily data, skip header rows
        daily_df = pd.read_csv(btc_daily_path, skiprows=2)
        daily_df.columns = ['DATE', 'Close', 'High', 'Low', 'Open', 'Volume']
        daily_df['DATE'] = pd.to_datetime(daily_df['DATE'])
        daily_df.set_index('DATE', inplace=True)
        
        # Create weekly aggregation (Friday close)
        weekly_df = daily_df.resample('W-FRI').agg({
            'Close': 'last',
            'High': 'max',
            'Low': 'min',
            'Open': 'first',
            'Volume': 'sum'
        })
        
        # Filter by period
        weekly_df = weekly_df.loc[PERIOD_START:PERIOD_END]
        
        if len(weekly_df) == 0:
            return None, None
        
        newest_date = weekly_df.index[-1]
        newest_week_df = weekly_df.loc[[newest_date]].copy()
        newest_week_df.reset_index(inplace=True)
        newest_week_df.columns = ['Date', 'Close ($)', 'High ($)', 'Low ($)', 'Open ($)', 'Volume']
        
        return newest_week_df, newest_date
    except Exception as e:
        print(f"Error getting newest week raw OHLCV: {e}")
        return None, None

def get_all_training_features():
    """
    Get all available features from training data (newest week)
    Combines BTC prices with lags, macro data, and compare data
    Returns: DataFrame with all features
    """
    try:
        # Start with BTC transformed data (includes lags)
        btc_path = DATA_DIR / "trans" / "cleaned_v1" / "BTC-USD_weekly_transformed_ffill.csv"
        if not btc_path.exists():
            return None
        
        all_data = pd.read_csv(btc_path)
        
        # Add macro data if available (use only fri version to avoid duplicates)
        macro_dir = DATA_DIR / "trans" / "cleaned_v1" / "macro"
        if macro_dir.exists():
            macro_file = macro_dir / "macro_fri_transformed_ffill.csv"
            if macro_file.exists():
                try:
                    macro_df = pd.read_csv(macro_file)
                    if 'DATE' in macro_df.columns and 'DATE' in all_data.columns:
                        # Drop duplicate columns before merge
                        existing_cols = set(all_data.columns)
                        new_cols = [col for col in macro_df.columns if col not in existing_cols or col == 'DATE']
                        macro_df = macro_df[new_cols]
                        # Merge on DATE
                        all_data = all_data.merge(macro_df, on='DATE', how='left')
                except Exception as e:
                    print(f"Warning: Could not load {macro_file.name}: {e}")
        
        # Add more data if available (use only fri version to avoid duplicates)
        more_dir = DATA_DIR / "trans" / "cleaned_v1" / "more"
        if more_dir.exists():
            more_file = more_dir / "more_fri_transformed_ffill.csv"
            if more_file.exists():
                try:
                    more_df = pd.read_csv(more_file)
                    if 'DATE' in more_df.columns and 'DATE' in all_data.columns:
                        # Drop duplicate columns before merge
                        existing_cols = set(all_data.columns)
                        new_cols = [col for col in more_df.columns if col not in existing_cols or col == 'DATE']
                        more_df = more_df[new_cols]
                        # Merge on DATE
                        all_data = all_data.merge(more_df, on='DATE', how='left')
                except Exception as e:
                    print(f"Warning: Could not load {more_file.name}: {e}")
        
        # Add compare data if available
        compare_dir = DATA_DIR / "trans" / "cleaned_v1" / "compare"
        if compare_dir.exists():
            compare_file = compare_dir / "compare_fri_transformed.csv"
            if compare_file.exists():
                try:
                    compare_df = pd.read_csv(compare_file)
                    if 'DATE' in compare_df.columns and 'DATE' in all_data.columns:
                        # Drop duplicate columns before merge
                        existing_cols = set(all_data.columns)
                        new_cols = [col for col in compare_df.columns if col not in existing_cols or col == 'DATE']
                        compare_df = compare_df[new_cols]
                        # Merge on DATE
                        all_data = all_data.merge(compare_df, on='DATE', how='left')
                except Exception as e:
                    print(f"Warning: Could not load {compare_file.name}: {e}")
        
        # Add sentiment features if available
        sentiment_path = DATA_DIR / "sentiment" / "weekly_sentiment.csv"
        if sentiment_path.exists():
            try:
                sentiment_df = pd.read_csv(sentiment_path)
                # Rename date column to DATE for consistency
                if 'date' in sentiment_df.columns:
                    sentiment_df.rename(columns={'date': 'DATE'}, inplace=True)
                if 'DATE' in sentiment_df.columns:
                    sentiment_df['DATE'] = pd.to_datetime(sentiment_df['DATE'])
                    all_data['DATE'] = pd.to_datetime(all_data['DATE'])
                    all_data = all_data.merge(sentiment_df, on='DATE', how='left')
            except Exception as e:
                print(f"Warning: Could not load sentiment data: {e}")
        
        # Get newest week only
        if len(all_data) > 0:
            newest_data = all_data.iloc[-1:].copy()
            print("\nDEBUG: Newest row in all_training_features (transformed):")
            print(newest_data.to_dict('records')[0])
            return newest_data
        
        return None
    except Exception as e:
        print(f"Error getting all training features: {e}")
        return None

def categorize_feature(feature_name):
    """
    Categorize a feature name into its data kind
    Returns: (category, display_name)
    """
    feature_lower = feature_name.lower()
    
    # Bitcoin (includes target, price, lags, and all BTC-related features)
    if 'btc' in feature_lower or 'bitcoin' in feature_lower:
        return ('Bitcoin', feature_name)
    
    # News Sentiment - including sentiment proportions and counts
    if any(sent in feature_lower for sent in ['sentiment', 'polarity', 'vader', 'ai_',
                                                'neg_prop', 'neu_prop', 'pos_prop',
                                                'neg_count', 'neu_count', 'pos_count',
                                                'neg_', 'neu_', 'pos_', 'max_sentiment',
                                                'min_sentiment', 'weighted_avg', 'finbert']):
        return ('News Sentiment', feature_name)
    
    # Macro Economic Indicators (includes indices, forex, commodities, etc.)
    if any(macro in feature_lower for macro in 
        ['macro', 'cpi', 'ppi', 'unrate', 'indpro', 'payems', 'houst', 'rsxfs', 'walcl', 'wm2ns',
         'effr', 'dgs10', 'pcepi', 'close_', 'gspc', 'vix', 'dxy', 'dxusd', 'gcf', 
         'ixic', 'clf', 'crbl', 'n225', 'ftse', 'eur', 'gbp', 'jpy']):
        return ('Macro Economic', feature_name)
    
    # Default: Other (technical indicators, on-chain, volume, etc.)
    return ('Other', feature_name)

def get_features_by_category():
    """
    Get all training features organized by category
    Returns: dict with category -> list of (feature_name, value) tuples
    """
    try:
        all_features_df = get_all_training_features()
        if all_features_df is None:
            return None
        
        # Convert to dictionary
        features_dict = all_features_df.iloc[0].to_dict()
        
        # Categorize features
        categorized = {}
        for feature_name, value in features_dict.items():
            if feature_name == 'DATE':
                continue
            
            category, display_name = categorize_feature(feature_name)
            
            if category not in categorized:
                categorized[category] = []
            
            # Format value
            if isinstance(value, (int, float)):
                if pd.isna(value):
                    formatted_value = "N/A"
                elif abs(value) < 0.001 and value != 0:
                    formatted_value = f"{value:.6e}"
                else:
                    formatted_value = f"{value:.6f}".rstrip('0').rstrip('.')
            else:
                formatted_value = str(value)
            
            categorized[category].append((display_name, formatted_value))
        
        # Sort categories in a logical order
        category_order = [
            'Bitcoin',
            'Macro Economic',
            'News Sentiment',
            'Other'
        ]
        
        sorted_categorized = {}
        for cat in category_order:
            if cat in categorized:
                sorted_categorized[cat] = sorted(categorized[cat], key=lambda x: x[0])
        
        # Add any remaining categories not in the order
        for cat in sorted(categorized.keys()):
            if cat not in sorted_categorized:
                sorted_categorized[cat] = sorted(categorized[cat], key=lambda x: x[0])
        
        return sorted_categorized
    except Exception as e:
        print(f"Error getting features by category: {e}")
        return None

def load_model(model_name):
    """
    Load trained model from pickle file
    model_name: 'lin_base', 'lin_model2', 'lin_model3', 'rf_base', etc.
    """
    try:
        model_path = MODELS_DIR / f"{model_name}.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        else:
            return None
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None

def load_all_models():
    """Load all 9 trained models"""
    models = {}
    model_names = [
        'lin_base', 'lin_model2', 'lin_model3',
        'rf_base', 'rf_model2', 'rf_model3',
        'xgb_base', 'xgb_model2', 'xgb_model3'
    ]
    
    for name in model_names:
        model = load_model(name)
        if model:
            models[name] = model
    
    return models

def load_metrics():
    """Load training metrics from JSON"""
    try:
        metrics_path = METRICS_DIR / "latest_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return metrics
        else:
            return {}
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return {}

def load_feature_columns(model_type='lin'):
    """
    Load feature column names for each model type
    Used for ensuring prediction data has correct features
    """
    try:
        features_path = METRICS_DIR / f"{model_type}_features.json"
        if features_path.exists():
            with open(features_path, 'r') as f:
                features = json.load(f)
            return features
        else:
            return {}
    except Exception as e:
        print(f"Error loading features for {model_type}: {e}")
        return {}

def get_sentiment_summary():
    """
    Get latest week sentiment summary from weekly_sentiment.csv
    Uses avg_sentiment_weekly (human) and AI_avg_sentiment_weekly (AI predictions)
    Returns: dict with summary text, sentiment label, and stats
    """
    try:
        sentiment_path = DATA_DIR / "sentiment" / "weekly_sentiment.csv"
        if not sentiment_path.exists():
            return None
        
        sentiment_df = pd.read_csv(sentiment_path, parse_dates=["date"])
        newest_sentiment = sentiment_df.iloc[-1]
        
        # Get sentiment scores from the latest week
        human_sentiment = newest_sentiment.get('avg_sentiment_weekly', 0)
        ai_sentiment = newest_sentiment.get('AI_avg_sentiment_weekly', 0)
        avg_sentiment = (human_sentiment + ai_sentiment) / 2
        
        # Determine sentiment label
        if avg_sentiment > 0.05:
            sentiment_label = "POSITIVE 😊"
            sentiment_color = "green"
        elif avg_sentiment < -0.05:
            sentiment_label = "NEGATIVE 😞"
            sentiment_color = "red"
        else:
            sentiment_label = "NEUTRAL 😐"
            sentiment_color = "gray"
        
        return {
            'date': newest_sentiment['date'],
            'vader_sentiment': human_sentiment,
            'ai_sentiment': ai_sentiment,
            'avg_sentiment': avg_sentiment,
            'label': sentiment_label,
            'color': sentiment_color
        }
    except Exception as e:
        print(f"Error getting sentiment summary: {e}")
        return None

def get_news_summary():
    """
    Get latest news summary
    Returns: list of recent news items
    """
    try:
        news_path = DATA_DIR / "sentiment" / "predicted_news.csv"
        if not news_path.exists():
            return []

        news_df = pd.read_csv(news_path)
        news_df.columns = news_df.columns.astype(str).str.strip()
        if "Date" in news_df.columns:
            news_df["Date"] = pd.to_datetime(news_df["Date"], errors="coerce")
        else:
            return []

        # Get newest week news
        newest_date = news_df["Date"].max()
        week_start = newest_date - pd.Timedelta(days=7)

        recent_news = news_df[
            (news_df["Date"] >= week_start) &
            (news_df["Date"] <= newest_date)
        ].sort_values("Date", ascending=False)

        return recent_news.head(20).to_dict("records")
    except Exception as e:
        print(f"Error getting news summary: {e}")
        return []

def save_model(model, model_name):
    """Save trained model to pickle"""
    try:
        model_path = MODELS_DIR / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved model: {model_path}")
    except Exception as e:
        print(f"Error saving model {model_name}: {e}")

def save_metrics(metrics):
    """Save metrics to JSON"""
    try:
        metrics_path = METRICS_DIR / "latest_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"Saved metrics: {metrics_path}")
    except Exception as e:
        print(f"Error saving metrics: {e}")

def save_features(features, model_type):
    """Save feature column names"""
    try:
        features_path = METRICS_DIR / f"{model_type}_features.json"
        with open(features_path, 'w') as f:
            json.dump(features, f, indent=2)
        print(f"Saved features for {model_type}")
    except Exception as e:
        print(f"Error saving features: {e}")


# ========== NEW FUNCTIONS FOR DYNAMIC WEEK DISPLAY ==========

def get_newest_completed_week():
    """
    Get the newest completed week (Friday date) from the cleaned data files.
    Returns: pd.Timestamp of the latest Friday or None if data unavailable
    """
    import pandas as pd
    from pathlib import Path
    
    try:
        BASE_DIR = Path(__file__).resolve().parent.parent
        btc_path = BASE_DIR / "data" / "cleaned_v2" / "BTC-USD_weekly.csv"
        
        if btc_path.exists():
            btc_df = pd.read_csv(btc_path)
            btc_df['DATE'] = pd.to_datetime(btc_df['DATE'])
            newest_date = btc_df['DATE'].max()
            return newest_date
    except Exception as e:
        print(f"Error getting newest completed week: {e}")
    
    return None


def get_week_comparison_data(week_date=None):
    """
    Get comprehensive data for a given week including:
    - Raw values for all variables (latest Friday data)
    - Previous week raw values
    - Percentage changes with metadata
    
    If week_date is None, uses the newest completed week.
    
    Returns: dict with structure:
    {
        'week_date': pd.Timestamp,
        'week_range': {'start': date, 'end': date},
        'variables': {
            'variable_name': {
                'raw_value': float,
                'raw_date': date,
                'prev_value': float,
                'prev_date': date,
                'pct_change': float,
                'change_desc': str (e.g. "from 2025-08-22 to 2025-08-29"),
                'symbol': str (↑ or ↓),
                'direction': str ('up' or 'down')
            }
        }
    }
    """
    import pandas as pd
    from pathlib import Path
    import numpy as np
    
    try:
        # Get newest week if not specified
        if week_date is None:
            week_date = get_newest_completed_week()
        
        if week_date is None:
            return None
        
        # Convert to pandas Timestamp
        if not isinstance(week_date, pd.Timestamp):
            week_date = pd.to_datetime(week_date)
        
        BASE_DIR = Path(__file__).resolve().parent.parent
        data_dir = BASE_DIR / "data" / "cleaned_v2"
        
        # Week range
        week_start, week_end = get_week_range_from_friday(week_date)
        
        # Load all data
        btc_df = pd.read_csv(data_dir / "BTC-USD_weekly.csv")
        btc_df['DATE'] = pd.to_datetime(btc_df['DATE'])
        btc_df = btc_df.sort_values('DATE').set_index('DATE')
        
        macro_df = pd.read_csv(data_dir / "macro" / "macro_fri.csv", index_col=0)
        macro_df.index = pd.to_datetime(macro_df.index)
        macro_df = macro_df.sort_index()
        
        more_df = pd.read_csv(data_dir / "more" / "more_fri.csv", index_col=0)
        more_df.index = pd.to_datetime(more_df.index)
        more_df = more_df.sort_index()
        
        # Build result
        result = {
            'week_date': week_date,
            'week_range': {'start': week_start, 'end': week_end},
            'variables': {}
        }
        
        # BTC variables (now using index)
        if week_date in btc_df.index:
            curr_btc_row = btc_df.loc[week_date]
            prev_btc_rows = btc_df[btc_df.index < week_date]
            
            for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
                if col in btc_df.columns and pd.notna(curr_btc_row[col]):
                    curr_val = curr_btc_row[col]
                    
                    if not prev_btc_rows.empty and col in prev_btc_rows.columns:
                        prev_val_series = prev_btc_rows[col].dropna()
                        if not prev_val_series.empty:
                            prev_val = prev_val_series.iloc[-1]
                            prev_date = prev_val_series.index[-1]
                            pct_change = ((curr_val - prev_val) / prev_val * 100) if prev_val != 0 else 0
                        else:
                            prev_val, prev_date, pct_change = None, None, 0
                    else:
                        prev_val, prev_date, pct_change = None, None, 0
                    
                    result['variables'][f'BTC {col}'] = {
                        'raw_value': float(curr_val),
                        'raw_date': week_date.date() if hasattr(week_date, 'date') else week_date,
                        'prev_value': float(prev_val) if prev_val else None,
                        'prev_date': prev_date.date() if prev_date and hasattr(prev_date, 'date') else prev_date,
                        'pct_change': float(pct_change),
                        'change_desc': f"from {prev_date.date() if prev_date and hasattr(prev_date, 'date') else prev_date} to {week_date.date() if hasattr(week_date, 'date') else week_date}",
                        'symbol': '↑' if pct_change >= 0 else '↓',
                        'direction': 'up' if pct_change >= 0 else 'down'
                    }
        
        # Macro variables (now using index)
        if week_date in macro_df.index:
            curr_macro_row = macro_df.loc[week_date]
            prev_macro_rows = macro_df[macro_df.index < week_date]
            
            for col in macro_df.columns:
                if pd.notna(curr_macro_row[col]):
                    curr_val = curr_macro_row[col]
                    
                    if not prev_macro_rows.empty and col in prev_macro_rows.columns:
                        prev_val_series = prev_macro_rows[col].dropna()
                        if not prev_val_series.empty:
                            prev_val = prev_val_series.iloc[-1]
                            prev_date = prev_val_series.index[-1]
                            pct_change = ((curr_val - prev_val) / prev_val * 100) if prev_val != 0 else 0
                        else:
                            prev_val, prev_date, pct_change = None, None, 0
                    else:
                        prev_val, prev_date, pct_change = None, None, 0
                    
                    result['variables'][f'Macro {col}'] = {
                        'raw_value': float(curr_val),
                        'raw_date': week_date.date() if hasattr(week_date, 'date') else week_date,
                        'prev_value': float(prev_val) if prev_val else None,
                        'prev_date': prev_date.date() if prev_date and hasattr(prev_date, 'date') else prev_date,
                        'pct_change': float(pct_change),
                        'change_desc': f"from {prev_date.date() if prev_date and hasattr(prev_date, 'date') else prev_date} to {week_date.date() if hasattr(week_date, 'date') else week_date}",
                        'symbol': '↑' if pct_change >= 0 else '↓',
                        'direction': 'up' if pct_change >= 0 else 'down'
                    }
        
        # More variables - only numeric columns (now using index)
        if week_date in more_df.index:
            curr_more_row = more_df.loc[week_date]
            prev_more_rows = more_df[more_df.index < week_date]
            
            for col in more_df.columns:
                # Skip non-numeric columns like 'symbol'
                if not pd.api.types.is_numeric_dtype(more_df[col]):
                    continue
                    
                if pd.notna(curr_more_row[col]):
                    curr_val = curr_more_row[col]
                    
                    if not prev_more_rows.empty and col in prev_more_rows.columns:
                        prev_val_series = prev_more_rows[col].dropna()
                        if not prev_val_series.empty:
                            prev_val = prev_val_series.iloc[-1]
                            prev_date = prev_val_series.index[-1]
                            try:
                                pct_change = ((curr_val - prev_val) / prev_val * 100) if prev_val != 0 else 0
                            except (TypeError, ValueError):
                                pct_change = 0
                        else:
                            prev_val, prev_date, pct_change = None, None, 0
                    else:
                        prev_val, prev_date, pct_change = None, None, 0
                    
                    try:
                        result['variables'][f'More {col}'] = {
                            'raw_value': float(curr_val),
                            'raw_date': week_date.date() if hasattr(week_date, 'date') else week_date,
                            'prev_value': float(prev_val) if prev_val else None,
                            'prev_date': prev_date.date() if prev_date and hasattr(prev_date, 'date') else prev_date,
                            'pct_change': float(pct_change),
                            'change_desc': f"from {prev_date.date() if prev_date and hasattr(prev_date, 'date') else prev_date} to {week_date.date() if hasattr(week_date, 'date') else week_date}",
                            'symbol': '↑' if pct_change >= 0 else '↓',
                            'direction': 'up' if pct_change >= 0 else 'down'
                        }
                    except (TypeError, ValueError):
                        pass
        
        # Sentiment variables
        try:
            sent_df = pd.read_csv(BASE_DIR / "data" / "sentiment" / "weekly_sentiment.csv")
            sent_df['date'] = pd.to_datetime(sent_df['date'], format='mixed', dayfirst=True, errors='coerce').dt.date
            
            # Filter sentiment data for the week date
            week_date_obj = week_date.date() if hasattr(week_date, 'date') else week_date
            sent_week = sent_df[sent_df['date'] == week_date_obj]
            
            if not sent_week.empty:
                sent_row = sent_week.iloc[0]
                
                # Get previous week's sentiment
                prev_sent = sent_df[sent_df['date'] < week_date_obj]
                
                for col in ['avg_sentiment_weekly', 'AI_avg_sentiment_weekly']:
                    if col in sent_df.columns and pd.notna(sent_row[col]):
                        curr_val = sent_row[col]
                        
                        if not prev_sent.empty and col in prev_sent.columns:
                            prev_val_series = prev_sent[col].dropna()
                            if not prev_val_series.empty:
                                prev_val = prev_val_series.iloc[-1]
                                prev_date = prev_sent.iloc[-1]['date']
                                try:
                                    pct_change = ((curr_val - prev_val) / abs(prev_val) * 100) if prev_val != 0 else 0
                                except (TypeError, ValueError):
                                    pct_change = 0
                            else:
                                prev_val, prev_date, pct_change = None, None, 0
                        else:
                            prev_val, prev_date, pct_change = None, None, 0
                        
                        # Friendly names
                        col_name = 'FinBERT Sentiment' if col == 'avg_sentiment_weekly' else 'AI Sentiment'
                        
                        result['variables'][col_name] = {
                            'raw_value': float(curr_val),
                            'raw_date': week_date_obj,
                            'prev_value': float(prev_val) if prev_val else None,
                            'prev_date': prev_date,
                            'pct_change': float(pct_change),
                            'change_desc': f"from {prev_date} to {week_date_obj}",
                            'symbol': '↑' if curr_val >= 0 else '↓',
                            'direction': 'up' if curr_val >= 0 else 'down'
                        }
        except Exception as e:
            # Sentiment data optional
            pass
        
        return result
    
    except Exception as e:
        print(f"Error getting week comparison data: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test loading
    print("Testing app_utils...")
    
    # Get newest week
    features, newest_date = get_newest_week_data()
    if features:
        print(f"Newest week: {newest_date}")
        print(f"Features available: {features.keys()}")
    
    # Get sentiment
    sentiment = get_sentiment_summary()
    if sentiment:
        print(f"Sentiment: {sentiment['label']}")
    
    # Get news
    news = get_news_summary()
    print(f"News items available: {len(news)}")
    
    print("app_utils test complete!")
