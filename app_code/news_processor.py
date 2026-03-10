"""
News summary and sentiment processing for the dashboard
"""

import pandas as pd
from pathlib import Path
from collections import Counter
import re

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def get_word_cloud_summary(max_words=30):
    """
    Generate word cloud from news titles (newest week)
    Returns: dict with word frequencies for visualization
    """
    try:
        news_path = DATA_DIR / "sentiment" / "predicted_news.csv"
        if not news_path.exists():
            return {}
        
        news_df = pd.read_csv(news_path)
        news_df.columns = news_df.columns.astype(str).str.strip()
        if "Date" in news_df.columns:
            news_df["Date"] = pd.to_datetime(news_df["Date"], errors="coerce")
        else:
            return {}
        
        # Get newest week news
        newest_date = news_df['Date'].max()
        week_start = newest_date - pd.Timedelta(days=7)
        
        recent_news = news_df[
            (news_df['Date'] >= week_start) & 
            (news_df['Date'] <= newest_date)
        ]
        
        if recent_news.empty:
            return {}
        
        # Extract words from titles
        all_words = []
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'be', 'been',
            'bitcoin', 'crypto', 'btc', 'news', 'market', 'price', 'trading'
        }
        
        for title in recent_news['Title'].fillna(''):
            # Clean and lowercase
            words = re.findall(r'\b[a-z]+\b', title.lower())
            # Remove stop words and short words
            words = [w for w in words if w not in stop_words and len(w) > 3]
            all_words.extend(words)
        
        # Count word frequencies
        word_freq = Counter(all_words).most_common(max_words)
        
        return dict(word_freq)
    
    except Exception as e:
        print(f"Error generating word cloud: {e}")
        return {}

def generate_news_paragraph(max_titles=5):
    """
    Generate a paragraph summary from top news titles
    """
    try:
        news_path = DATA_DIR / "sentiment" / "predicted_news.csv"
        if not news_path.exists():
            return "No news data available"
        
        news_df = pd.read_csv(news_path)
        news_df.columns = news_df.columns.astype(str).str.strip()
        if "Date" in news_df.columns:
            news_df["Date"] = pd.to_datetime(news_df["Date"], errors="coerce")
        else:
            return "No news data available"
        
        # Get newest week news
        newest_date = news_df['Date'].max()
        week_start = newest_date - pd.Timedelta(days=7)
        
        recent_news = news_df[
            (news_df['Date'] >= week_start) & 
            (news_df['Date'] <= newest_date)
        ].sort_values('Date', ascending=False)
        
        if recent_news.empty:
            return "No news available for this week"
        
        # Get top titles by relevance (most recent first)
        top_titles = recent_news.head(max_titles)['Title'].tolist()
        
        # Create paragraph
        paragraph = "**Latest Bitcoin News Summary (This Week):**\n\n"
        for i, title in enumerate(top_titles, 1):
            paragraph += f"{i}. {title}\n"
        
        return paragraph
    
    except Exception as e:
        print(f"Error generating news paragraph: {e}")
        return "Error loading news data"

def get_sentiment_distribution():
    """
    Get sentiment distribution for newest week
    Returns: dict with counts and percentages
    """
    try:
        news_path = DATA_DIR / "sentiment" / "predicted_news.csv"
        if not news_path.exists():
            return {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0}
        
        news_df = pd.read_csv(news_path)
        news_df.columns = news_df.columns.astype(str).str.strip()
        if "Date" in news_df.columns:
            news_df["Date"] = pd.to_datetime(news_df["Date"], errors="coerce")
        else:
            return {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0}
        
        # Get newest week news
        newest_date = news_df['Date'].max()
        week_start = newest_date - pd.Timedelta(days=7)
        
        recent_news = news_df[
            (news_df['Date'] >= week_start) & 
            (news_df['Date'] <= newest_date)
        ]
        
        if recent_news.empty:
            return {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0}
        
        # Count sentiments (assuming column name is 'Sentiment' or similar)
        # Adjust based on your actual CSV structure
        sentiment_col = None
        preferred_cols = ["AI_Predicted_Label", "Predicted Label", "Predicted_Label"]
        for col in preferred_cols:
            if col in recent_news.columns:
                sentiment_col = col
                break
        if sentiment_col is None:
            # fallback: any column containing 'label'
            for col in recent_news.columns:
                if 'label' in col.lower():
                    sentiment_col = col
                    break
        
        if sentiment_col is None:
            return {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0}
        
        total = len(recent_news)
        labels = recent_news[sentiment_col].astype(str).str.strip().str.lower()
        positive = (labels == 'positive').sum()
        negative = (labels == 'negative').sum()
        neutral = total - positive - negative
        
        return {
            'positive': positive,
            'neutral': neutral,
            'negative': negative,
            'total': total
        }
    
    except Exception as e:
        print(f"Error getting sentiment distribution: {e}")
        return {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0}

def get_news_by_source():
    """
    Get news distribution by source
    """
    try:
        news_path = DATA_DIR / "sentiment" / "predicted_news.csv"
        if not news_path.exists():
            return {}
        
        news_df = pd.read_csv(news_path)
        news_df.columns = news_df.columns.astype(str).str.strip()
        if "Date" in news_df.columns:
            news_df["Date"] = pd.to_datetime(news_df["Date"], errors="coerce")
        else:
            return {}
        
        # Get newest week news
        newest_date = news_df['Date'].max()
        week_start = newest_date - pd.Timedelta(days=7)
        
        recent_news = news_df[
            (news_df['Date'] >= week_start) & 
            (news_df['Date'] <= newest_date)
        ]
        
        # Count by source
        if 'Source' in recent_news.columns:
            source_counts = recent_news['Source'].value_counts().to_dict()
            return source_counts
        else:
            return {}
    
    except Exception as e:
        print(f"Error getting news by source: {e}")
        return {}

if __name__ == "__main__":
    # Test
    print("Testing news_processor...")
    
    print("\n1. Word Cloud Summary:")
    word_freq = get_word_cloud_summary()
    print(f"Top words: {list(word_freq.items())[:10]}")
    
    print("\n2. News Paragraph:")
    para = generate_news_paragraph()
    print(para)
    
    print("\n3. Sentiment Distribution:")
    sentiment = get_sentiment_distribution()
    print(sentiment)
    
    print("\n4. News by Source:")
    sources = get_news_by_source()
    print(sources)
    
    print("\nNews processor test complete!")
def get_all_sources():
    """
    Get all unique news sources used in the dataset
    Returns: sorted list of sources
    """
    try:
        news_path = DATA_DIR / "sentiment" / "predicted_news.csv"
        if not news_path.exists():
            return []
        
        news_df = pd.read_csv(news_path)
        news_df.columns = news_df.columns.astype(str).str.strip()
        
        if "Source" in news_df.columns:
            all_sources = sorted(news_df['Source'].dropna().unique().tolist())
            return all_sources
        
        return []
    except Exception as e:
        print(f"Error getting all sources: {e}")
        return []