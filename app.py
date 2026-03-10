"""
Bitcoin Price Analysis
Interactive Streamlit app for viewing model performance and predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
from PIL import Image
import sys
import os
import datetime

# Add app_code to path
sys.path.insert(0, str(Path(__file__).parent / "app_code"))

from app_code.app_utils import (
    get_newest_week_data, get_sentiment_summary, get_news_summary,
    load_metrics, load_all_models, get_newest_week_df, get_newest_week_raw_ohlcv,
    get_all_training_features, get_features_by_category, get_weekly_data_combined, get_latest_raw_feature_values,
    get_newest_completed_week, get_week_range_from_friday
)
from app_code.news_processor import (
    generate_news_paragraph, get_word_cloud_summary, 
    get_sentiment_distribution, get_news_by_source, get_all_sources
)

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Weekly Bitcoin Price Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS STYLING ==========
st.markdown('''
<style>
    /* ---- Global font size boost ---- */
    html, body, [class*="css"] {
        font-size: 18px !important;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {
        font-size: 1.05rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem !important;
    }
    .stRadio label, .stSelectbox label, .stMultiSelect label, .stCheckbox label {
        font-size: 1.05rem !important;
    }
    .stDataFrame, .stTable {
        font-size: 1.02rem !important;
    }
    h1 { font-size: 2.2rem !important; }
    h2 { font-size: 1.8rem !important; }
    h3 { font-size: 1.5rem !important; }
    h4 { font-size: 1.3rem !important; }

    .main-header {
        font-size: 48px;
        font-weight: bold;
        color: #1f77b4;
    }
    .sticky-header {
        position: sticky;
        top: 0;
        background-color: white;
        z-index: 999;
        padding: 15px 0;
        border-bottom: 2px solid #1f77b4;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive { color: #2ecc71; font-weight: bold; }
    .negative { color: #e74c3c; font-weight: bold; }
    .neutral { color: #95a5a6; font-weight: bold; }
</style>
''', unsafe_allow_html=True)

# ========== SIDEBAR ==========
st.sidebar.title("📋 Navigation")
st.sidebar.markdown("---")

# Tab selection in sidebar
selected_tab = st.sidebar.radio(
    "Select Page:",
    [
        "📊 Dashboard",
        "📰 News & Sentiment",
        "📈 Latest data",
        "🔵 Logistic Regression",
        "🌳 Random Forest",
        "⚡ XGBoost",
        "🧠 LSTM",
        "🎯 Model Comparison",
        "💵 Trading Simulation"
    ]
)
st.sidebar.markdown("---")


# Load data for newest completed week dynamically
from app_code.app_utils import get_weekly_data_combined, get_newest_completed_week
@st.cache_data
def load_all_data():
    newest_week = get_newest_completed_week()
    if newest_week is None:
        # Fallback to a fixed date if no data available
        newest_week = "2025-08-29"
    week_date_str = newest_week.strftime('%Y-%m-%d') if hasattr(newest_week, 'strftime') else str(newest_week)
    week_data = get_weekly_data_combined(week_date_str)
    return week_data

try:
    data = load_all_data()
    week_label = data.get('label', 'Latest Week')
    week_range = data.get('week_range', {})
    sentiment_info = data['raw'].get('sentiment', {})
except Exception as e:
    st.sidebar.error(f"Error loading data: {e}")
    st.stop()

# ========== HEADER ==========
st.markdown('<h1 class="main-header">📊 Weekly Bitcoin Price Analysis</h1>', unsafe_allow_html=True)
# Show coming Friday close
st.markdown(f"**Coming Friday close: {data.get('coming_friday', '2026-03-06')}**")
# Show latest data range
if week_range:
    st.markdown(f"**Latest Data: {week_range['start']} to {week_range['end']}**")
st.markdown("---")

# ========== HELPER FUNCTION: DISPLAY MODEL METRICS ==========
def display_model_metrics(metrics_dict, model_name, feature_importance_path=None, col_count=3):
    """Display metrics for a single model"""
    if not metrics_dict:
        st.warning(f"No metrics available for {model_name}")
        return

    st.subheader(f"{model_name} - Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        acc = metrics_dict.get('accuracy')
        if acc is not None:
            st.metric("Accuracy", f"{acc:.4f}", delta=f"{acc*100:.2f}%")

    with col2:
        auc = metrics_dict.get('auc')
        if auc is not None:
            st.metric("AUC", f"{auc:.4f}")

    with col3:
        f1 = metrics_dict.get('f1')
        if f1 is not None:
            st.metric("F1 Score", f"{f1:.4f}")

    # Model details
    with st.expander("📋 Model Details", expanded=False):
        st.write(f"**Type:** {metrics_dict.get('model_type', 'Unknown')}")
        st.write(f"**Features Used:** {metrics_dict.get('features', 'Unknown')}")
        if 'accuracy' in metrics_dict:
            st.json({k: v for k, v in metrics_dict.items() if k != 'model_type'})
        # Feature importance plot
        if feature_importance_path and Path(feature_importance_path).exists():
            st.image(str(feature_importance_path), use_container_width=True)

# ========== DASHBOARD TAB ==========
if selected_tab == "📊 Dashboard":
    # Reduce default padding for a compact dashboard
    st.markdown("""<style>
        .block-container {padding-top:1rem; padding-bottom:0rem;}
        [data-testid="stMetricValue"] {font-size:1.7rem;}
        [data-testid="stMetricLabel"] {font-size:1.1rem;}
        [data-testid="stMetricDelta"] {font-size:1.05rem;}
    </style>""", unsafe_allow_html=True)

    # ---- Load all dashboard data ----
    btc_weekly_path = Path("data/cleaned_v2/BTC-USD_weekly.csv")
    btc_weekly = pd.read_csv(btc_weekly_path, parse_dates=["DATE"]) if btc_weekly_path.exists() else pd.DataFrame()

    sent_path = Path("data/sentiment/weekly_sentiment.csv")
    if sent_path.exists():
        sent_df = pd.read_csv(sent_path)
        sent_df['date'] = pd.to_datetime(sent_df['date'], format='mixed', dayfirst=True, errors='coerce')
        sent_df = sent_df.sort_values('date')
    else:
        sent_df = pd.DataFrame()

    pred_files = {
        "Logistic Reg.": Path("data/predictions/predictions_linear_regression.csv"),
        "Random Forest": Path("data/predictions/predictions_random_forest.csv"),
        "XGBoost": Path("data/predictions/predictions_xgboost.csv"),
        "LSTM": Path("data/predictions/predictions_lstm.csv"),
    }
    model_preds = {}
    for name, p in pred_files.items():
        if p.exists():
            df = pd.read_csv(p)
            if not df.empty:
                last = df.iloc[-1]
                model_preds[name] = {
                    'date': last['date'],
                    'direction': 'UP' if last['y_pred_binary'] == 1 else 'DOWN',
                    'confidence': last.get('confidence', None),
                }

    # Add LSTM metrics to dashboard
    metrics_path = Path("metrics/latest_metrics.json")
    lstm_metrics = None
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            all_metrics = json.load(f)
        lstm_metrics = all_metrics.get('models', {}).get('lstm_model3', None)
    # Fallback: load from lstm_model3_metrics.json if not found
    if lstm_metrics is None:
        lstm_metrics_path = Path("metrics/lstm_model3_metrics.json")
        if lstm_metrics_path.exists():
            with open(lstm_metrics_path, 'r') as f:
                lstm_metrics = json.load(f)

    # ===== ROW 1: Price metrics + Sentiment + Predictions (compact top bar) =====
    if not btc_weekly.empty:
        latest_price = btc_weekly['Close'].iloc[-1]
        prev_price = btc_weekly['Close'].iloc[-2] if len(btc_weekly) > 1 else latest_price
        price_change = ((latest_price - prev_price) / prev_price) * 100
    else:
        latest_price = prev_price = price_change = 0

    if not btc_weekly.empty:
        latest_date = btc_weekly['DATE'].iloc[-1]
        st.markdown(f"**📈 Latest Weekly Price ({latest_date.strftime('%Y-%m-%d')})**")
    else:
        st.markdown("**📈 Latest Weekly Price**")
    top1, top2, top3, top4 = st.columns([1.2, 1, 1, 3.4])
    with top1:
        st.metric("BTC Close", f"${latest_price:,.0f}", f"{price_change:+.2f}%")
    with top2:
        if not btc_weekly.empty:
            st.metric("High", f"${btc_weekly['High'].iloc[-1]:,.0f}")
    with top3:
        if not btc_weekly.empty:
            st.metric("Low", f"${btc_weekly['Low'].iloc[-1]:,.0f}")
    with top4:
        # --- Build Predictions HTML (left side) ---
        pred_html = ""
        if model_preds:
            pred_parts = []
            for name, pred in model_preds.items():
                icon = '🟢↑' if pred['direction'] == 'UP' else '🔴↓'
                c = pred['confidence']
                c_str = f"Prob: {c*100:.0f}%" if c is not None and pd.notna(c) else ""
                pred_parts.append(f"<div style='padding:3px 0;'><span style='font-size:1.2em;'><b>{name}</b> {icon}</span> <span style='color:#555;font-size:1.05em'>{c_str}</span></div>")
            pred_html = (
                f"<div style='font-size:1.2em;color:#1a73e8;font-weight:700;margin-bottom:6px;'>Predictions for 2026-03-06 (predicting Friday closing price)</div>"
                + "".join(pred_parts)
            )
        # --- Build Model Accuracy HTML (right side) ---
        acc_section_html = ""
        metrics_path = Path("metrics/latest_metrics.json")
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                all_metrics = json.load(f)
            models_meta = all_metrics.get('models', {})
            best_models = {}
            for key, m in models_meta.items():
                mtype = m.get('model_type', key)
                base_type = mtype.split('(')[0].strip()
                acc = m.get('accuracy', 0)
                if base_type not in best_models or acc > best_models[base_type]['accuracy']:
                    best_models[base_type] = {'accuracy': acc, 'auc': m.get('auc', 0), 'f1': m.get('f1', 0), 'variant': mtype}
            acc_html = ""
            for bt in ["Logistic Regression", "Random Forest", "XGBoost", "LSTM"]:
                if bt in best_models:
                    bm = best_models[bt]
                    ap = bm['accuracy'] * 100
                    ac = '#2ecc71' if ap >= 60 else ('#e67e22' if ap >= 50 else '#e74c3c')
                    short_name = bt.replace("Logistic Regression", "Logistic Reg.").replace("Random Forest", "RF").replace("LSTM", "LSTM")
                    acc_html += (
                        f"<div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #eee;font-size:1.05em;'>"
                        f"<span><b>{short_name}</b></span>"
                        f"<span style='color:{ac};font-weight:700;'>Acc {ap:.1f}%</span>"
                        f"<span>AUC {bm['auc']:.2f}</span>"
                        f"<span>F1 {bm['f1']:.2f}</span>"
                        f"</div>"
                    )
            # Always show LSTM metrics row, even if missing
            if lstm_metrics:
                ap = lstm_metrics.get('accuracy', 0) * 100
                ac = '#2ecc71' if ap >= 60 else ('#e67e22' if ap >= 50 else '#e74c3c')
                auc_val = lstm_metrics.get('auc', 0)
                f1_val = lstm_metrics.get('f1', 0)
            else:
                ap = 0
                ac = '#e74c3c'
                auc_val = 0
                f1_val = 0
            acc_html += (
                f"<div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #eee;font-size:1.05em;'>"
                f"<span><b>LSTM</b></span>"
                f"<span style='color:{ac};font-weight:700;'>Acc {ap:.1f}%</span>"
                f"<span>AUC {auc_val:.2f}</span>"
                f"<span>F1 {f1_val:.2f}</span>"
                f"</div>"
            )
            acc_section_html = (
                "<div style='font-size:1.1em;font-weight:700;margin-bottom:4px;'>Model Performance<br>(Accuracy, AUC, F1 score)</div>"
                + acc_html
            )
        # --- Combine into one rectangle ---
        st.markdown(
            f"<div style='border:2.5px solid #1a73e8;border-radius:10px;padding:12px 16px;background:linear-gradient(135deg,#e8f0fe,#d2e3fc);box-shadow:0 2px 8px rgba(26,115,232,0.15);'>"
            f"<div style='display:flex;gap:24px;'>"
            f"<div style='flex:1;border-right:1.5px solid #b0c4de;padding-right:16px;'>{pred_html}</div>"
            f"<div style='flex:1;padding-left:8px;'>{acc_section_html}</div>"
            f"</div></div>",
            unsafe_allow_html=True
        )

    # ===== Period selector (controls all charts below) =====
    time_options = ["3M", "6M", "1Y", "2Y", "All"]
    selected_range = st.radio("Period:", time_options, index=4, horizontal=True, key="dashboard_time_range")
    range_map = {"3M": 90, "6M": 180, "1Y": 365, "2Y": 730, "All": None}
    days = range_map[selected_range]

    # ===== ROW 2: Price chart (left) + Weekly Returns (middle) + Sentiment Trend (right) =====
    chart_col, ret_col, trend_col = st.columns([1.8, 1.5, 1.1], gap="small")

    with chart_col:
        st.markdown("**Bitcoin Weekly Close Price**")
        if not btc_weekly.empty:
            max_date = btc_weekly['DATE'].max()
            btc_display = btc_weekly[btc_weekly['DATE'] >= max_date - pd.Timedelta(days=days)] if days else btc_weekly

            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=btc_display['DATE'], y=btc_display['Close'],
                mode='lines', name='BTC Close',
                line=dict(color='#F7931A', width=2),
                fill='tozeroy', fillcolor='rgba(247,147,26,0.08)'
            ))
            fig_price.update_layout(
                height=400,
                margin=dict(l=5, r=5, t=5, b=5),
                xaxis=dict(title='', showgrid=False, tickfont=dict(size=12)),
                yaxis=dict(title='USD', showgrid=True, gridcolor='rgba(128,128,128,0.15)', tickfont=dict(size=12), titlefont=dict(size=13)),
                plot_bgcolor='rgba(0,0,0,0)',
                hovermode='x unified',
            )
            st.plotly_chart(fig_price, use_container_width=True, key="dashboard_btc_price")

    with ret_col:
        st.markdown("**Weekly Returns**")
        if not btc_weekly.empty and len(btc_weekly) > 1:
            max_date = btc_weekly['DATE'].max()
            returns_all = btc_weekly.copy()
            returns_all['Return'] = returns_all['Close'].pct_change() * 100
            returns_all = returns_all.dropna(subset=['Return'])
            if days:
                returns_df = returns_all[returns_all['DATE'] >= max_date - pd.Timedelta(days=days)]
            else:
                returns_df = returns_all
            colors = ['#2ecc71' if r >= 0 else '#e74c3c' for r in returns_df['Return']]
            fig_ret = go.Figure(data=[go.Bar(
                x=returns_df['DATE'], y=returns_df['Return'],
                marker_color=colors,
            )])
            fig_ret.update_layout(
                height=400,
                margin=dict(l=5, r=5, t=5, b=5),
                xaxis=dict(title='', showgrid=False, tickformat='%b %d', tickfont=dict(size=10)),
                yaxis=dict(title='%', showgrid=True, gridcolor='rgba(128,128,128,0.15)', zeroline=True, zerolinecolor='gray', tickfont=dict(size=11)),
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_ret, use_container_width=True, key="dashboard_weekly_returns")

    with trend_col:
        st.markdown("**Sentiment Trend**")
        if not sent_df.empty:
            max_sent_date = sent_df['date'].max()
            if days:
                recent_sent = sent_df[sent_df['date'] >= max_sent_date - pd.Timedelta(days=days)]
            else:
                recent_sent = sent_df
            if len(recent_sent) > 1:
                fig_sent = go.Figure()
                fig_sent.add_trace(go.Scatter(
                    x=recent_sent['date'], y=recent_sent['avg_sentiment_weekly'],
                    mode='lines+markers', name='FinBERT',
                    line=dict(color='#3498db', width=2), marker=dict(size=3)
                ))
                fig_sent.add_trace(go.Scatter(
                    x=recent_sent['date'], y=recent_sent['AI_avg_sentiment_weekly'],
                    mode='lines+markers', name='AI',
                    line=dict(color='#e67e22', width=2), marker=dict(size=3)
                ))
                fig_sent.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.5)
                fig_sent.update_layout(
                    height=400,
                    margin=dict(l=5, r=5, t=5, b=5),
                    xaxis=dict(showticklabels=True, showgrid=False, tickfont=dict(size=10)),
                    yaxis=dict(showticklabels=True, showgrid=False, tickfont=dict(size=11)),
                    plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(orientation='h', yanchor='bottom', y=1, xanchor='center', x=0.5, font=dict(size=10)),
                )
                st.plotly_chart(fig_sent, use_container_width=True, key="dashboard_sentiment_trend")

    # ===== ROW 3: Sentiment Scores + AI Weekly News Summary =====
    bot1, bot2 = st.columns([1, 2], gap="small")

    with bot1:
        st.markdown("**Sentiment**")
        if not sent_df.empty:
            latest_sent = sent_df.iloc[-1]
            fb = latest_sent.get('avg_sentiment_weekly', None)
            ai = latest_sent.get('AI_avg_sentiment_weekly', None)
            fb_str = f"{fb:.3f}" if fb is not None and pd.notna(fb) else "N/A"
            ai_str = f"{ai:.3f}" if ai is not None and pd.notna(ai) else "N/A"
            fb_color = '#2ecc71' if fb and fb > 0 else ('#e74c3c' if fb and fb < 0 else '#95a5a6')
            ai_color = '#2ecc71' if ai and ai > 0 else ('#e74c3c' if ai and ai < 0 else '#95a5a6')
            fb_label = 'Bullish' if fb and fb > 0 else ('Bearish' if fb and fb < 0 else 'Neutral')
            ai_label = 'Bullish' if ai and ai > 0 else ('Bearish' if ai and ai < 0 else 'Neutral')
            st.markdown(
                f"<div style='border:1.5px solid #d0d0d0;border-radius:8px;padding:12px 16px;background:#f9f9f9;'>"
                f"<div style='display:flex;justify-content:space-around;'>"
                f"<div style='text-align:center;'><div style='font-size:1.1em;color:gray;'>FinBERT</div>"
                f"<div style='font-size:1.5em;font-weight:700;color:{fb_color};'>{fb_str}</div>"
                f"<div style='font-size:1.0em;color:{fb_color};font-weight:600;'>{fb_label}</div></div>"
                f"<div style='text-align:center;'><div style='font-size:1.1em;color:gray;'>AI (Llama)</div>"
                f"<div style='font-size:1.5em;font-weight:700;color:{ai_color};'>{ai_str}</div>"
                f"<div style='font-size:1.0em;color:{ai_color};font-weight:600;'>{ai_label}</div></div>"
                f"</div></div>",
                unsafe_allow_html=True
            )
        else:
            st.info("No sentiment data.")

    with bot2:
        st.markdown("**AI Weekly News Summary**")
        summary_csv = Path("data/sentiment/weekly_news_summary.csv")
        if summary_csv.exists():
            try:
                sum_df = pd.read_csv(summary_csv)
                if not sum_df.empty:
                    latest_sum = sum_df.iloc[-1]
                    summary_text = str(latest_sum['summary'])
                    st.markdown(
                        f"<div style='border:1.5px solid #d0d0d0;border-radius:8px;padding:10px 14px;background:#f9f9f9;"
                        f"font-size:1.0em;line-height:1.5;max-height:130px;overflow-y:auto;'>"
                        f"{summary_text}</div>",
                        unsafe_allow_html=True
                    )
                    st.caption(f"{latest_sum['week_start']} – {latest_sum['week_end']} · {int(latest_sum['headline_count'])} headlines")
                else:
                    st.info("No summary yet.")
            except Exception:
                st.info("No summary yet.")
        else:
            st.info("Summary generated during weekly update.")

# ========== MAIN PAGE TAB ==========
elif selected_tab == "🏠 Overall":
    # ...existing code...
    date_step = max(1, len(all_dates) // 50)  # Show ~50 labels max
    date_labels = [d.strftime('%d/%m/%Y') if i % date_step == 0 else d.strftime('%d/%m/%Y') for i, d in enumerate(all_dates)]

    # Display date labels above
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Start Date:**")
    with col2:
        st.write("**End Date:**")

    # Display selected date range in DD/MM/YYYY format (initialize with defaults first)
    col1, col2 = st.columns(2)
    col1_placeholder = col1.empty()
    col2_placeholder = col2.empty()

    # Use select_slider with actual date strings
    selected_range = st.select_slider(
        "📅 Select date range:",
        options=[d.strftime('%d/%m/%Y') for d in all_dates],
        value=(all_dates[0].strftime('%d/%m/%Y'), all_dates[-1].strftime('%d/%m/%Y'))
    )
    
    # Convert selected dates back to date objects
    start_date_obj = datetime.datetime.strptime(selected_range[0], '%d/%m/%Y').date()
    end_date_obj = datetime.datetime.strptime(selected_range[1], '%d/%m/%Y').date()
    start_date = pd.to_datetime(start_date_obj)
    end_date = pd.to_datetime(end_date_obj)

    # Now display the selected dates
    col1_placeholder.markdown(f"## {start_date_obj.strftime('%d/%m/%Y')}")
    col2_placeholder.markdown(f"## {end_date_obj.strftime('%d/%m/%Y')}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Show/hide prediction overlay
    show_pred = st.checkbox("Show prediction overlay on price chart", value=True)

    # Filter price data by selected date range
    price_df = price_df_all[(price_df_all["DATE"] >= start_date) & (price_df_all["DATE"] <= end_date)]
    
    # Load prediction data
    pred_path = Path("data/sentiment/predicted_news.csv")

    # Try to load prediction direction CSV
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        pred_df["Published"] = pd.to_datetime(pred_df["Published"], format='mixed', dayfirst=True, errors='coerce')
        pred_df = pred_df[(pred_df["Published"] >= start_date) & (pred_df["Published"] <= end_date)]
    else:
        pred_df = pd.DataFrame()

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        latest_price = price_df["Close"].iloc[-1] if not price_df.empty else None
        st.metric("Latest Price", f"${latest_price:,.2f}" if latest_price else "N/A")
    with col2:
        # Show latest prediction direction
        if not pred_df.empty:
            latest_pred = pred_df["Predicted_Label"].iloc[-1]
            latest_label = pred_df["AI_Predicted_Label"].iloc[-1]
            conf = pred_df["Predicted_Probability"].iloc[-1] if "Predicted_Probability" in pred_df.columns else None
            arrow = "⬆️" if latest_pred == 1 else ("⬇️" if latest_pred == -1 else "↔️")
            label_text = f"{latest_label}" if pd.notna(latest_label) else "N/A"
            st.metric("Sentiment", f"{arrow} {label_text}" if label_text != "N/A" else arrow)
        else:
            st.metric("Sentiment", "N/A")
    with col3:
        # Prediction confidence
        if not pred_df.empty and "Predicted_Probability" in pred_df.columns:
            avg_conf = pred_df["Predicted_Probability"].mean()
            st.metric("Avg Confidence", f"{avg_conf*100:.1f}%" if pd.notna(avg_conf) else "N/A")
        else:
            st.metric("Avg Confidence", "N/A")
    with col4:
        # Volatility (std dev of Close)
        vol = price_df["Close"].std() if not price_df.empty else None
        st.metric("Volatility", f"{vol:.2f}" if vol else "N/A")
    with col5:
        # Trading volume
        vol_val = price_df["Volume"].iloc[-1] if not price_df.empty else None
        st.metric("Volume", f"{vol_val:,.0f}" if vol_val else "N/A")

    st.markdown("---")

    # Historical price chart with sentiment overlay
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_df['DATE'], y=price_df['Close'], mode='lines', name='BTC Price'))
    if show_pred and not pred_df.empty:
        # Overlay sentiment markers
        up_mask = pred_df['Predicted_Label'] == 1
        down_mask = pred_df['Predicted_Label'] == -1

        # Create price lookup by date (convert index to dates for proper matching)
        price_by_date = price_df.set_index(pd.to_datetime(price_df['DATE']).dt.date).rename_axis('date_index')

        # Helper function to get price for a given date
        def get_price_for_date(d):
            try:
                if d.date() in price_by_date.index:
                    return price_by_date.loc[d.date(), 'Close']
            except:
                pass
            return None

        # Get prices for up and down sentiments
        up_prices = pred_df['Published'][up_mask].map(get_price_for_date)
        down_prices = pred_df['Published'][down_mask].map(get_price_for_date)

        # Only plot markers where we have valid price data
        up_valid = up_prices.notna()
        down_valid = down_prices.notna()

        fig.add_trace(go.Scatter(
            x=pred_df['Published'][up_mask][up_valid],
            y=up_prices[up_valid],
            mode='markers',
            marker=dict(color='green', symbol='arrow-up', size=10),
            name='Positive Sentiment'
        ))
        fig.add_trace(go.Scatter(
            x=pred_df['Published'][down_mask][down_valid],
            y=down_prices[down_valid],
            mode='markers',
            marker=dict(color='red', symbol='arrow-down', size=10),
            name='Negative Sentiment'
        ))
    fig.update_layout(title='Bitcoin Price (with Prediction Overlay)', yaxis=dict(title='Price'), xaxis=dict(title='Date'))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Model performance section (4 models)
    st.subheader("Model Performance Overview")
    model_names = ["Logistic", "Random Forest", "XGBoost", "LSTM"]
    model_metrics = [
        {"accuracy": 0.65, "precision": 0.7, "recall": 0.6},
        {"accuracy": 0.68, "precision": 0.72, "recall": 0.63},
        {"accuracy": 0.70, "precision": 0.75, "recall": 0.65},
        None  # LSTM not available
    ]
    cols = st.columns(4)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**{model_names[i]} Model**")
            if model_metrics[i]:
                st.metric("Accuracy", f"{model_metrics[i]['accuracy']*100:.1f}%")
                st.metric("Precision", f"{model_metrics[i]['precision']*100:.1f}%")
                st.metric("Recall", f"{model_metrics[i]['recall']*100:.1f}%")
                st.markdown("Confusion Matrix:")
                st.image("https://via.placeholder.com/150x100?text=Confusion+Matrix")
                st.markdown("Feature Importance:")
                st.image("https://via.placeholder.com/150x100?text=Feature+Importance")
            else:
                st.info("Model not available on this computer.")

    st.markdown("---")

    # Technical indicators chart (example)
    st.subheader("Technical Indicators")
    # Example: 50-day and 200-day moving averages
    if not price_df.empty:
        price_df['MA50'] = price_df['Close'].rolling(window=50).mean()
        price_df['MA200'] = price_df['Close'].rolling(window=200).mean()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=price_df['DATE'], y=price_df['Close'], mode='lines', name='BTC Price'))
        fig2.add_trace(go.Scatter(x=price_df['DATE'], y=price_df['MA50'], mode='lines', name='MA50'))
        fig2.add_trace(go.Scatter(x=price_df['DATE'], y=price_df['MA200'], mode='lines', name='MA200'))
        fig2.update_layout(title='BTC Price with Moving Averages', yaxis=dict(title='Price'), xaxis=dict(title='Date'))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Volume & Volatility chart
    st.subheader("Volume & Volatility")
    if not price_df.empty:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=price_df['DATE'], y=price_df['Volume'], name='Volume'))
        fig3.add_trace(go.Scatter(x=price_df['DATE'], y=price_df['Close'].rolling(window=14).std(), mode='lines', name='Volatility (14d std)'))
        fig3.update_layout(title='BTC Volume & Volatility', yaxis=dict(title='Volume / Volatility'), xaxis=dict(title='Date'))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # News & Sentiment Feed
    st.subheader("News & Sentiment")
    news_path = Path("data/sentiment/predicted_news.csv")
    if news_path.exists():
        news_df = pd.read_csv(news_path)
        news_df["Published"] = pd.to_datetime(news_df["Published"], format='mixed', dayfirst=True, errors='coerce')
        news_df = news_df[(news_df["Published"] >= start_date) & (news_df["Published"] <= end_date)]
        st.write("Recent News Headlines:")
        st.dataframe(news_df[["Published", "Title", "Source", "AI_Predicted_Label"]].tail(10), use_container_width=True)
        # Sentiment distribution
        if "AI_Predicted_Label" in news_df.columns:
            sent_counts = news_df["AI_Predicted_Label"].value_counts()
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(x=sent_counts.index, y=sent_counts.values, marker_color=['green','grey','red']))
            fig4.update_layout(title='Sentiment Distribution', yaxis=dict(title='Count'), xaxis=dict(title='Sentiment'))
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No news data available.")

    st.markdown("---")

    # Export/download button
    st.download_button(
        label="Download Dashboard Data (CSV)",
        data=price_df.to_csv(index=False),
        file_name="btc_dashboard_data.csv",
        mime="text/csv"
    )


# ========== TAB 1: NEWS & SENTIMENT ==========
elif selected_tab == "📰 News & Sentiment":
    st.title("📰 News & Sentiment Analysis")
    with st.expander("News sentiment only includes articles from these sources:"):
        st.markdown("""
        - Bloomberg.com
        - Yahoo Finance
        - The New York Times
        - Reuters
        - Forbes
        - CoinDesk
        - Decrypt
        - Cointelegraph
        - The Block
        - Bitcoin Magazine
        - CNBC
        - The Wall Street Journal
        - CoinShares
        - U.Today
        - CoinJournal
        - CryptoPotato
        - ForkLog
        - The Daily Hodl
        - CoinGape
        - Business Insider
        - WIRED
        - MIT Technology Review
        - BeInCrypto
        - Al Jazeera
        - Asia Times
        - SWI swissinfo.ch
        - The Straits Times
        - Greek City Times
        - nairametrics.com
        - techcentral.co.za
        - globaltimes.cn
        - TechAfrica News
        """)
    
    # Use the dynamic newest week range for all news & sentiment data
    try:
        newest_week = get_newest_completed_week()
        if newest_week:
            week_start, week_end = get_week_range_from_friday(newest_week)
            week_range = {'start': str(week_start), 'end': str(week_end)}
        else:
            week_range = {'start': '2026-02-21', 'end': '2026-02-27'}
    except Exception as e:
        st.error(f"Error getting week range: {e}")
        week_range = {'start': '2026-02-21', 'end': '2026-02-27'}
    
    st.markdown(f"**Latest Week: {week_range['start']} to {week_range['end']}**")

    # Initialize variables
    week_news = pd.DataFrame()
    
    week_start_dt = pd.to_datetime(week_range['start'])
    week_end_dt = pd.to_datetime(week_range['end']) + pd.Timedelta(days=1)

    # ---- Load all sentiment data from predicted_news.csv ----
    # This file has both:
    #   - Predicted_Label (-1/0/1) = FinBERT-based model predictions
    #   - AI_Predicted_Label (NEGATIVE/NEUTRAL/POSITIVE) = AI (Llama) predictions
    news_path = Path("data/sentiment/predicted_news.csv")
    if news_path.exists():
        try:
            all_df = pd.read_csv(news_path)
            all_df.columns = all_df.columns.astype(str).str.strip()
            all_df["Published"] = pd.to_datetime(all_df["Published"], format='mixed', dayfirst=True, errors="coerce")
            week_news = all_df[(all_df["Published"] >= week_start_dt) & (all_df["Published"] < week_end_dt)].copy()
            
            # Map Predicted_Label numeric to text for FinBERT display
            label_map = {-1: "NEGATIVE", -1.0: "NEGATIVE", 0: "NEUTRAL", 0.0: "NEUTRAL", 1: "POSITIVE", 1.0: "POSITIVE"}
            if "Predicted_Label" in week_news.columns:
                week_news["FinBERT_Label"] = week_news["Predicted_Label"].map(label_map)
        except Exception as e:
            st.error(f"❌ Error loading news data: {e}")

    # ---- Load weekly aggregated sentiment scores ----
    weekly_sent_path = Path("data/sentiment/weekly_sentiment.csv")
    weekly_sent = pd.DataFrame()
    if weekly_sent_path.exists():
        try:
            weekly_sent = pd.read_csv(weekly_sent_path)
            weekly_sent['date'] = pd.to_datetime(weekly_sent['date'], format='mixed', dayfirst=True, errors='coerce')
            weekly_sent = weekly_sent.sort_values('date')
        except:
            pass

    st.markdown("---")

    # ========== SECTION 1: Weekly Aggregated Scores ==========
    st.subheader("📈 Weekly Sentiment Scores")
    if not weekly_sent.empty:
        # Find the row closest to our week end
        week_end_date = pd.to_datetime(week_range['end'])
        closest_idx = (weekly_sent['date'] - week_end_date).abs().idxmin()
        latest_row = weekly_sent.loc[closest_idx]
        
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            val = latest_row.get('avg_sentiment_weekly', None)
            if val is not None and pd.notna(val):
                color = '🟢' if val > 0 else ('🔴' if val < 0 else '⚪')
                st.metric("FinBERT Avg Score", f"{val:.4f}")
                st.caption(f"{color} {'Bullish' if val > 0 else ('Bearish' if val < 0 else 'Neutral')}")
            else:
                st.metric("FinBERT Avg Score", "N/A")
        with sc2:
            pos_fb = int(latest_row.get('pos_count_weekly', 0))
            neg_fb = int(latest_row.get('neg_count_weekly', 0))
            neu_fb = int(latest_row.get('neu_count_weekly', 0))
            st.markdown("**FinBERT +/-/~**")
            st.markdown(
                f"<div style='display:flex;gap:6px;align-items:center;margin-top:4px;'>"
                f"<span style='background:#2ecc71;color:white;padding:4px 10px;border-radius:6px;font-weight:bold;font-size:1.1em;'>+{pos_fb}</span>"
                f"<span style='background:#e74c3c;color:white;padding:4px 10px;border-radius:6px;font-weight:bold;font-size:1.1em;'>-{neg_fb}</span>"
                f"<span style='background:#95a5a6;color:white;padding:4px 10px;border-radius:6px;font-weight:bold;font-size:1.1em;'>~{neu_fb}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        with sc3:
            val = latest_row.get('AI_avg_sentiment_weekly', None)
            if val is not None and pd.notna(val):
                color = '🟢' if val > 0 else ('🔴' if val < 0 else '⚪')
                st.metric("AI (Llama) Avg Score", f"{val:.4f}")
                st.caption(f"{color} {'Bullish' if val > 0 else ('Bearish' if val < 0 else 'Neutral')}")
            else:
                st.metric("AI (Llama) Avg Score", "N/A")
        with sc4:
            pos_ai = int(latest_row.get('AI_pos_count_weekly', 0))
            neg_ai = int(latest_row.get('AI_neg_count_weekly', 0))
            neu_ai = int(latest_row.get('AI_neu_count_weekly', 0))
            st.markdown("**AI +/-/~**")
            st.markdown(
                f"<div style='display:flex;gap:6px;align-items:center;margin-top:4px;'>"
                f"<span style='background:#2ecc71;color:white;padding:4px 10px;border-radius:6px;font-weight:bold;font-size:1.1em;'>+{pos_ai}</span>"
                f"<span style='background:#e74c3c;color:white;padding:4px 10px;border-radius:6px;font-weight:bold;font-size:1.1em;'>-{neg_ai}</span>"
                f"<span style='background:#95a5a6;color:white;padding:4px 10px;border-radius:6px;font-weight:bold;font-size:1.1em;'>~{neu_ai}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("No weekly aggregated sentiment data available.")

    st.markdown("---")

    # ========== SECTION 2: AI-Generated News Summary (loaded from pre-generated file) ==========
    st.subheader("🤖 AI Weekly News Summary")
    summary_csv_path = os.path.join(os.path.dirname(__file__), "data", "sentiment", "weekly_news_summary.csv")
    summary_loaded = False
    if os.path.exists(summary_csv_path):
        try:
            summary_df = pd.read_csv(summary_csv_path)
            week_end_str = str(week_range['end'])
            match = summary_df[summary_df['week_end'].astype(str).str.strip() == week_end_str.strip()]
            if not match.empty:
                summary_text = match.iloc[-1]['summary']
                headline_count = int(match.iloc[-1]['headline_count'])
                st.markdown(f"> {summary_text}")
                st.caption(f"Generated by Llama-4-maverick based on {headline_count} headlines")
                summary_loaded = True
        except Exception:
            pass
    if not summary_loaded:
        st.info("No AI summary available for this week. It will be generated during the next weekly update.")

    st.markdown("---")

    # ========== SECTION 3: Side-by-side Pie Charts ==========
    st.subheader("📊 Sentiment Distribution Comparison")
    pie1, pie2 = st.columns(2)

    with pie1:
        st.markdown("**🤖 AI Sentiment (Llama)**")
        if not week_news.empty and "AI_Predicted_Label" in week_news.columns:
            counts_ai = week_news["AI_Predicted_Label"].dropna().str.upper().value_counts().to_dict()
            pos_a = counts_ai.get("POSITIVE", 0)
            neu_a = counts_ai.get("NEUTRAL", 0)
            neg_a = counts_ai.get("NEGATIVE", 0)
            fig_ai = go.Figure(data=[go.Pie(
                labels=['Positive', 'Neutral', 'Negative'],
                values=[pos_a, neu_a, neg_a],
                marker=dict(colors=['#2ecc71', '#95a5a6', '#e74c3c']),
                textinfo='label+percent'
            )])
            fig_ai.update_layout(height=350, margin=dict(t=10, b=10))
            st.plotly_chart(fig_ai, use_container_width=True, key="news_ai_pie")
            avg_ai = week_news["AI_Predicted_Label"].dropna().map({"POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0}).mean()
            st.metric("AI Avg Sentiment", f"{avg_ai:.2f}" if pd.notna(avg_ai) else "N/A")
            st.caption(f"{week_news['AI_Predicted_Label'].notna().sum()} articles analysed")
        else:
            st.info("No AI sentiment data for this week.")

    with pie2:
        st.markdown("**📊 FinBERT Sentiment**")
        if not week_news.empty and "FinBERT_Label" in week_news.columns:
            counts_fb = week_news["FinBERT_Label"].dropna().str.upper().value_counts().to_dict()
            pos_f = counts_fb.get("POSITIVE", 0)
            neu_f = counts_fb.get("NEUTRAL", 0)
            neg_f = counts_fb.get("NEGATIVE", 0)
            fig_fb = go.Figure(data=[go.Pie(
                labels=['Positive', 'Neutral', 'Negative'],
                values=[pos_f, neu_f, neg_f],
                marker=dict(colors=['#2ecc71', '#95a5a6', '#e74c3c']),
                textinfo='label+percent'
            )])
            fig_fb.update_layout(height=350, margin=dict(t=10, b=10))
            st.plotly_chart(fig_fb, use_container_width=True, key="news_finbert_pie")
            avg_fb = week_news["FinBERT_Label"].dropna().map({"POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0}).mean()
            st.metric("FinBERT Avg Sentiment", f"{avg_fb:.2f}" if avg_fb is not None and pd.notna(avg_fb) else "N/A")
            st.caption(f"{week_news['FinBERT_Label'].notna().sum()} articles analysed")
        else:
            st.info("No FinBERT data for this week.")

    st.markdown("---")

    # ========== SECTION 4: Word Cloud ==========
    st.subheader("☁️ News Headlines Word Cloud")
    if not week_news.empty and "Title" in week_news.columns:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        all_titles = " ".join(week_news["Title"].dropna().tolist())
        stopwords = set(WordCloud().stopwords)
        stopwords.update(["bitcoin", "btc", "crypto", "cryptocurrency", "s", "t", "will", "new", "says", "could", "one", "us"])

        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            colormap="viridis",
            max_words=100,
            stopwords=stopwords,
            collocations=False,
        ).generate(all_titles)

        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)
        plt.close(fig_wc)
    else:
        st.info("No headlines available to generate word cloud.")

    st.markdown("---")

    # ========== SECTION 5: News by Source ==========
    st.subheader("📰 News by Source")
    if not week_news.empty and "Source" in week_news.columns:
        source_counts = week_news["Source"].value_counts().head(10).to_dict()
        fig_src = go.Figure(data=[go.Bar(
            x=list(source_counts.keys()),
            y=list(source_counts.values()),
            marker=dict(color='#3498db')
        )])
        fig_src.update_layout(
            title="Top 10 News Sources",
            xaxis_title="Source", yaxis_title="Count",
            height=350, showlegend=False
        )
        st.plotly_chart(fig_src, use_container_width=True, key="news_source_bar")

    st.markdown("---")

    # ========== SECTION 6: Articles Table with both labels ==========
    st.subheader(f"📄 News Articles for Recent Week ({len(week_news)} articles)")
    if not week_news.empty:
        # Show both sentiment columns side by side
        cols_to_show = ["Published", "Title", "Source"]
        if "AI_Predicted_Label" in week_news.columns:
            cols_to_show.append("AI_Predicted_Label")
        if "FinBERT_Label" in week_news.columns:
            cols_to_show.append("FinBERT_Label")
        display_df = week_news[cols_to_show].copy()
        display_df = display_df.rename(columns={"AI_Predicted_Label": "AI Sentiment", "FinBERT_Label": "FinBERT Sentiment"})
        st.dataframe(display_df.sort_values("Published", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ No news data available for the selected week.")

# ========== TAB 2: LATEST DATA ===========
elif selected_tab == "📈 Latest data":
    st.header("📈 Latest data (Newest Completed Week)")

    # Load data first
    from app_code.app_utils import get_week_comparison_data
    data = load_all_data()

    # Display Bitcoin weekly summary statistics at the top
    btc_raw = data['raw'].get('btc', None)
    if btc_raw is not None:
        st.subheader("🪙 Bitcoin Weekly Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        # Get raw numeric values
        close_val = float(btc_raw.get('Close', 0))
        open_val = float(btc_raw.get('Open', 0))
        high_val = float(btc_raw.get('High', 0))
        low_val = float(btc_raw.get('Low', 0))
        weekly_change = close_val - open_val
        weekly_change_pct = (weekly_change / open_val * 100) if open_val != 0 else 0
        col1.metric("Close", f"${close_val:,.2f}", f"{weekly_change_pct:+.2f}%")
        col2.metric("Open", f"${open_val:,.2f}")
        col3.metric("High", f"${high_val:,.2f}")
        col4.metric("Low", f"${low_val:,.2f}")
        col5.metric("Week Range", f"${high_val - low_val:,.2f}")
        newest_week = get_newest_completed_week()
        week_end_str = newest_week.strftime('%Y-%m-%d') if newest_week and hasattr(newest_week, 'strftime') else '2025-08-29'
        st.info(f"📅 Week ending: {week_end_str}")

    st.markdown("---")

    # ===== SECTION: DYNAMIC WEEK COMPARISON =====
    comparison_data = get_week_comparison_data()
    if comparison_data:
        st.markdown(f"**Week:** {comparison_data['week_range']['start']} to {comparison_data['week_range']['end']}")
        st.markdown("---")

        # Display in a nice table format
        st.subheader("All Variables - Raw Values & Week-over-Week Change")
        st.caption("Shows the newest completed week's raw values, percentage changes from the previous available data, and direction indicators.")

        # Organize variables by category
        btc_vars = {k: v for k, v in comparison_data['variables'].items() if k.startswith('BTC')}
        macro_vars = {k: v for k, v in comparison_data['variables'].items() if k.startswith('Macro')}
        more_vars = {k: v for k, v in comparison_data['variables'].items() if k.startswith('More')}
        sentiment_vars = {k: v for k, v in comparison_data['variables'].items() if k in ['FinBERT Sentiment', 'AI Sentiment']}

        # BTC Variables
        if btc_vars:
            st.subheader("🪙 Bitcoin (BTC) Data")
            cols = st.columns([2, 2, 2, 2])
            for col, (var_name, var_data) in zip(cols, list(btc_vars.items())[:4]):
                with col:
                    raw_val = var_data['raw_value']
                    pct_chg = var_data['pct_change']
                    symbol = var_data['symbol']
                    color = "🟢" if var_data['direction'] == 'up' else "🔴"

                    st.metric(
                        label=var_name,
                        value=f"{raw_val:,.2f}" if isinstance(raw_val, float) else raw_val,
                        delta=f"{symbol} {pct_chg:+.2f}%" if var_data['prev_value'] else "N/A",
                    )

            # Detailed table for BTC
            btc_table_data = []
            for var_name, var_data in btc_vars.items():
                btc_table_data.append({
                    'Variable': var_name,
                    'Raw Value': f"{var_data['raw_value']:,.2f}" if isinstance(var_data['raw_value'], float) else var_data['raw_value'],
                    'Data Date': var_data['raw_date'],
                    'Previous Value': f"{var_data['prev_value']:,.2f}" if var_data['prev_value'] else 'N/A',
                    'Change %': f"{var_data['pct_change']:+.2f}%" if var_data['prev_value'] else 'N/A',
                    '↑↓': var_data['symbol'],
                    'Calculation': var_data['change_desc']
                })
            if btc_table_data:
                st.dataframe(pd.DataFrame(btc_table_data), use_container_width=True)

        st.markdown("---")

        # Macro Variables
        if macro_vars:
            st.subheader("📊 Macro Economic Data")
            macro_table_data = []
            for var_name, var_data in macro_vars.items():
                macro_table_data.append({
                    'Variable': var_name,
                    'Raw Value': f"{var_data['raw_value']:,.4f}" if isinstance(var_data['raw_value'], float) else var_data['raw_value'],
                    'Data Date': var_data['raw_date'],
                    'Previous Value': f"{var_data['prev_value']:,.4f}" if var_data['prev_value'] else 'N/A',
                    'Change %': f"{var_data['pct_change']:+.2f}%" if var_data['prev_value'] else 'N/A',
                    '↑↓': var_data['symbol'],
                    'Calculation': var_data['change_desc']
                })
            if macro_table_data:
                st.dataframe(pd.DataFrame(macro_table_data), use_container_width=True)
            else:
                st.info("No macro data available for this week")

        st.markdown("---")

        # More Variables (Blockchain & Transaction Data)
        if more_vars:
            st.subheader("⛓️ Blockchain & Transaction Data")
            more_table_data = []
            for var_name, var_data in more_vars.items():
                more_table_data.append({
                    'Variable': var_name,
                    'Raw Value': f"{var_data['raw_value']:,.2f}" if isinstance(var_data['raw_value'], float) else var_data['raw_value'],
                    'Data Date': var_data['raw_date'],
                    'Previous Value': f"{var_data['prev_value']:,.2f}" if var_data['prev_value'] else 'N/A',
                    'Change %': f"{var_data['pct_change']:+.2f}%" if var_data['prev_value'] else 'N/A',
                    '↑↓': var_data['symbol'],
                    'Calculation': var_data['change_desc']
                })
            if more_table_data:
                st.dataframe(pd.DataFrame(more_table_data), use_container_width=True)
            else:
                st.info("No blockchain data available for this week")

        st.markdown("---")

        # Sentiment Variables - All columns from weekly_sentiment.csv
        st.subheader("💬 Sentiment Analysis")
        
        try:
            # Get newest week date
            newest_week = get_newest_completed_week()
            if newest_week:
                from app_code.app_utils import get_week_range_from_friday
                week_start, week_end = get_week_range_from_friday(newest_week)
                week_date_obj = week_end
            else:
                week_date_obj = None
            
            # Load sentiment data
            sent_df = pd.read_csv(Path("data/sentiment/weekly_sentiment.csv"))
            sent_df['date'] = pd.to_datetime(sent_df['date'], format='mixed', dayfirst=True, errors='coerce').dt.date
            
            if week_date_obj and week_date_obj in sent_df['date'].values:
                sent_row = sent_df[sent_df['date'] == week_date_obj].iloc[0]
                
                # Separate into FinBERT and AI columns
                finbert_cols = [col for col in sent_df.columns if col != 'date' and not col.startswith('AI_')]
                ai_cols = [col for col in sent_df.columns if col.startswith('AI_')]
                
                # Display FinBERT Sentiment Table
                st.subheader("📊 FinBERT Sentiment (Weekly)")
                finbert_data = []
                for col in finbert_cols:
                    val = sent_row[col]
                    finbert_data.append({
                        'Metric': col,
                        'Value': f"{val:.4f}" if isinstance(val, (int, float)) and pd.notna(val) else str(val)
                    })
                finbert_df = pd.DataFrame(finbert_data)
                st.dataframe(finbert_df, use_container_width=True, hide_index=True)
                
                # Display AI Sentiment Table
                st.subheader("🤖 AI Sentiment (Weekly) - llama-4-maverick")
                ai_data = []
                for col in ai_cols:
                    # Remove 'AI_' prefix for display
                    display_col = col.replace('AI_', '')
                    val = sent_row[col]
                    ai_data.append({
                        'Metric': display_col,
                        'Value': f"{val:.4f}" if isinstance(val, (int, float)) and pd.notna(val) else str(val)
                    })
                ai_df = pd.DataFrame(ai_data)
                st.dataframe(ai_df, use_container_width=True, hide_index=True)
            else:
                st.info("No sentiment data available for this week")
        except Exception as e:
            st.warning(f"Could not load sentiment data: {e}")
    else:
        st.warning("Unable to load week comparison data. Please check the data files.")

    st.markdown("---")

# ========== TAB 3: LOGISTIC MODELS ==========
elif selected_tab == "🔵 Logistic Regression":
    st.markdown("<span style='color:gray'><b>Note:</b> Accuracy, AUC, and F1 Score are calculated based on the test period: <b>2024-09-13 to 2025-08-31</b> for each model variant.</span>", unsafe_allow_html=True)
    st.header("🔵 Logistic Regression Models")
    st.markdown("**Framework:** Logistic Regression (Classification)")
    st.markdown("---")


    metrics = load_metrics()

    # Base Model
    st.subheader("Base Model (BTC Lags Only)")
    if 'LOC_BASE' in metrics.get('models', {}):
        display_model_metrics(metrics['models']['LOC_BASE'], "Base Model",
                              feature_importance_path="/Users/charlotteho/Desktop/fyp/figure_lin/lin_base_feature_importance.png")
    else:
        st.warning("Base model metrics not available")

    st.markdown("---")

    # Model 1
    st.subheader("Model 1 (Macro Features Only)")
    if 'LOC_MODEL1' in metrics.get('models', {}):
        display_model_metrics(metrics['models']['LOC_MODEL1'], "Model 1",
                              feature_importance_path="/Users/charlotteho/Desktop/fyp/figure_lin/lin_model1_feature_importance.png")
    else:
        st.warning("Model 1 metrics not available")

    st.markdown("---")

    # Model 2
    st.subheader("Model 2 (BTC + Macro + More Market Data)")
    if 'LOC_MODEL2' in metrics.get('models', {}):
        display_model_metrics(metrics['models']['LOC_MODEL2'], "Model 2",
                              feature_importance_path="/Users/charlotteho/Desktop/fyp/figure_lin/lin_model2_feature_importance.png")
    else:
        st.info("Model 2 metrics not available (check if model was trained)")

    st.markdown("---")

    # Model 3
    st.subheader("Model 3 (BTC + Macro + More + Sentiment)")
    if 'LOC_MODEL3' in metrics.get('models', {}):
        display_model_metrics(metrics['models']['LOC_MODEL3'], "Model 3",
                              feature_importance_path="/Users/charlotteho/Desktop/fyp/figure_lin/lin_model3_feature_importance.png")
    else:
        st.info("Model 3 metrics not available (check if model was trained)")

    st.markdown("---")

    # Generated figures section
    st.subheader("📊 Model Visuals")
    st.write("**Model Comparison Plot**")
    fig_path = Path("/Users/charlotteho/Desktop/fyp/figure_lin/lin_model_comparison.png")
    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)
    else:
        st.info("Figure not yet generated. Run `python generate_figures.py --only lin`")

    st.write("**Confusion Matrices (All Variants)**")
    cm_cols = st.columns(4)
    for i, (vk, vlabel) in enumerate([("base", "Base"), ("model1", "Model 1"), ("model2", "Model 2"), ("model3", "Model 3")]):
        with cm_cols[i]:
            st.caption(vlabel)
            cm_path = Path(f"/Users/charlotteho/Desktop/fyp/figure_lin/lin_{vk}_confusion_matrix.png")
            if cm_path.exists():
                st.image(str(cm_path), use_container_width=True)
            else:
                st.info("Not yet generated")

    # Walk-Forward Analysis
    with st.expander("📊 Advanced: Walk-Forward Rolling Window Analysis", expanded=False):
        st.write("""
        **What is Walk-Forward Analysis?**

        This analysis tests model performance across different training window sizes:
        - Trains on windows: 52, 78, 104, 130, 156, 182 weeks
        - Evaluates on next 51 weeks (test set)
        - Shows how performance scales with more training data
        """)
        show_walkforward = st.checkbox("📈 Show Rolling Window Plots", value=False, key="wf_lin_checkbox")  # Change key for each tab: wf_lin_checkbox, wf_rf_checkbox, wf_xgb_checkbox
        if show_walkforward:
            st.write("**Logistic Model Walk-Forward**")  # Change label for each tab: Logistic, Random Forest, XGBoost
            try:
                # NOTE: This is just the older saved figure, not the actual correct walk-forward comparison plot.
                fig_path = Path("/Users/charlotteho/Desktop/fyp/???/figure_lin/lin_walkforward_comparison.png")
                if fig_path.exists():
                    st.image(str(fig_path), width=400)
                else:
                    st.info("Figure not yet generated")
            except:
                st.info("Figures will be available after model training")

# ========== TAB 4: RANDOM FOREST MODELS ==========
elif selected_tab == "🌳 Random Forest":
    st.markdown("<span style='color:gray'><b>Note:</b> Accuracy, AUC, and F1 Score are calculated based on the test period: <b>2024-09-13 to 2025-08-31</b> for each model variant.</span>", unsafe_allow_html=True)
    st.header("🌳 Random Forest Models")
    st.markdown("**Framework:** Random Forest Classifier")
    st.markdown("---")

    metrics = load_metrics()

    # Base Model
    st.subheader("Base Model (BTC Lags Only)")
    if 'rf_base' in metrics.get('models', {}):
        display_model_metrics(metrics['models']['rf_base'], "Base Model",
                              feature_importance_path="/Users/charlotteho/Desktop/fyp/figure_rf/rf_base_feature_importance.png")
    else:
        st.warning("Base model metrics not available")

    st.markdown("---")

    # Model 1
    st.subheader("Model 1 (Macro Features Only)")
    if 'rf_model1' in metrics.get('models', {}):
        display_model_metrics(metrics['models']['rf_model1'], "Model 1",
                              feature_importance_path="/Users/charlotteho/Desktop/fyp/figure_rf/rf_model1_feature_importance.png")
    else:
        st.warning("Model 1 metrics not available")

    st.markdown("---")

    # Model 2
    st.subheader("Model 2 (BTC + Macro + More Market Data)")
    if 'rf_model2' in metrics.get('models', {}):
        display_model_metrics(metrics['models']['rf_model2'], "Model 2",
                              feature_importance_path="/Users/charlotteho/Desktop/fyp/figure_rf/rf_model2_feature_importance.png")
    else:
        st.info("Model 2 metrics not available (check if model was trained)")

    st.markdown("---")

    # Model 3
    st.subheader("Model 3 (BTC + Macro + More + Sentiment)")
    if 'rf_model3' in metrics.get('models', {}):
        display_model_metrics(metrics['models']['rf_model3'], "Model 3",
                              feature_importance_path="/Users/charlotteho/Desktop/fyp/figure_rf/rf_model3_feature_importance.png")
    else:
        st.info("Model 3 metrics not available (check if model was trained)")

    st.markdown("---")

    # Generated figures section
    st.subheader("📊 Model Visuals")
    st.write("**Model Comparison Plot**")
    fig_path = Path("/Users/charlotteho/Desktop/fyp/figure_rf/rf_model_comparison.png")
    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)
    else:
        st.info("Figure not yet generated. Run `python generate_figures.py --only rf`")

    st.write("**Confusion Matrices (All Variants)**")
    cm_cols = st.columns(4)
    for i, (vk, vlabel) in enumerate([("base", "Base"), ("model1", "Model 1"), ("model2", "Model 2"), ("model3", "Model 3")]):
        with cm_cols[i]:
            st.caption(vlabel)
            cm_path = Path(f"/Users/charlotteho/Desktop/fyp/figure_rf/rf_{vk}_confusion_matrix.png")
            if cm_path.exists():
                st.image(str(cm_path), use_container_width=True)
            else:
                st.info("Not yet generated")

    # Walk-Forward Analysis (Model2, Rolling Window)
    with st.expander("📊 Advanced: Walk-Forward Rolling Window Analysis", expanded=False):
        st.write("""
        **What is Walk-Forward Analysis?**

        This analysis tests Model2 performance across different training window sizes:
        - Trains on windows: 52, 78, 104, 130, 156, 182 weeks
        - Evaluates on next 51 weeks (test set)
        - Shows how performance scales with more training data
        """)
        show_walkforward = st.checkbox("📈 Show Rolling Window Plots", value=False, key="wf_rf_checkbox")
        if show_walkforward:
            st.write("**Random Forest Model2 Walk-Forward**")
            try:
                # NOTE: This is just the older saved figure, not the actual correct walk-forward comparison plot.
                fig_path = Path("/Users/charlotteho/Desktop/fyp/???/figure_rf/rf_walkforward_comparison.png")
                if fig_path.exists():
                    st.image(str(fig_path), use_container_width=True)
                else:
                    st.info("Figure not yet generated")
            except:
                st.info("Figures will be available after model training")

# ========== TAB 5: XGBOOST MODELS ==========
elif selected_tab == "⚡ XGBoost":
    st.markdown("<span style='color:gray'><b>Note:</b> Accuracy, AUC, and F1 Score are calculated based on the test period: <b>2024-09-13 to 2025-08-31</b> for each model variant.</span>", unsafe_allow_html=True)
    st.header("⚡ XGBoost Models")
    st.markdown("**Framework:** XGBoost Classifier")
    st.markdown("---")

    metrics = load_metrics()

    # Base Model
    st.subheader("Base Model (BTC Lags Only)")
    if 'xgb_base' in metrics.get('models', {}):
        display_model_metrics(metrics['models']['xgb_base'], "Base Model",
                              feature_importance_path="/Users/charlotteho/Desktop/fyp/figure_xgb/xgb_base_feature_importance.png")
    else:
        st.warning("Base model metrics not available")

    st.markdown("---")

    # Model 1
    st.subheader("Model 1 (Macro Features Only)")
    if 'xgb_model1' in metrics.get('models', {}):
        display_model_metrics(metrics['models']['xgb_model1'], "Model 1",
                              feature_importance_path="/Users/charlotteho/Desktop/fyp/figure_xgb/xgb_model1_feature_importance.png")
    else:
        st.warning("Model 1 metrics not available")

    st.markdown("---")

    # Model 2
    st.subheader("Model 2 (BTC + Macro + More Market Data)")
    if 'xgb_model2' in metrics.get('models', {}):
        display_model_metrics(metrics['models']['xgb_model2'], "Model 2",
                              feature_importance_path="/Users/charlotteho/Desktop/fyp/figure_xgb/xgb_model2_feature_importance.png")
    else:
        st.info("Model 2 metrics not available (check if model was trained)")

    st.markdown("---")

    # Model 3
    st.subheader("Model 3 (BTC + Macro + More + Sentiment)")
    if 'xgb_model3' in metrics.get('models', {}):
        display_model_metrics(metrics['models']['xgb_model3'], "Model 3",
                              feature_importance_path="/Users/charlotteho/Desktop/fyp/figure_xgb/xgb_model3_feature_importance.png")
    else:
        st.info("Model 3 metrics not available (check if model was trained)")

    st.markdown("---")

    # Generated figures section
    st.subheader("📊 Model Visuals")
    st.write("**Model Comparison Plot**")
    fig_path = Path("/Users/charlotteho/Desktop/fyp/figure_xgb/xgb_model_comparison.png")
    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)
    else:
        st.info("Figure not yet generated. Run `python generate_figures.py --only xgb`")

    st.write("**Confusion Matrices (All Variants)**")
    cm_cols = st.columns(4)
    for i, (vk, vlabel) in enumerate([("base", "Base"), ("model1", "Model 1"), ("model2", "Model 2"), ("model3", "Model 3")]):
        with cm_cols[i]:
            st.caption(vlabel)
            cm_path = Path(f"/Users/charlotteho/Desktop/fyp/figure_xgb/xgb_{vk}_confusion_matrix.png")
            if cm_path.exists():
                st.image(str(cm_path), use_container_width=True)
            else:
                st.info("Not yet generated")

    # Walk-Forward Analysis (Model2, Rolling Window)
    with st.expander("📊 Advanced: Walk-Forward Rolling Window Analysis", expanded=False):
        st.write("""
        **What is Walk-Forward Analysis?**

        This analysis tests Model2 performance across different training window sizes:
        - Trains on windows: 52, 78, 104, 130, 156, 182 weeks
        - Evaluates on next 51 weeks (test set)
        - Shows how performance scales with more training data
        """)
        show_walkforward = st.checkbox("📈 Show Rolling Window Plots", value=False, key="wf_xgb_checkbox")
        if show_walkforward:
            st.write("**XGBoost Model2 Walk-Forward**")
            try:
                # NOTE: This is just the older saved figure, not the actual correct walk-forward comparison plot.
                fig_path = Path("/Users/charlotteho/Desktop/fyp/???/figure_xgb/xgb_walkforward_comparison.png")
                if fig_path.exists():
                    st.image(str(fig_path), use_container_width=True)
                else:
                    st.info("Figure not yet generated")
            except:
                st.info("Figures will be available after model training")
                
# ========== TAB: LSTM ==========
elif selected_tab == "🧠 LSTM":
    st.markdown("<span style='color:gray'><b>Note:</b> Accuracy, AUC, and F1 Score are calculated based on the test period: <b>2024-09-13 to 2025-08-31</b> for each model variant.</span>", unsafe_allow_html=True)
    st.header("🧠 LSTM (Long Short-Term Memory)")
    st.markdown("**Framework:** Bidirectional LSTM + Multi-Head Attention")
    st.markdown("---")

    # Architecture description
    st.subheader("Model Architecture")
    st.markdown("""
    - **Input**: 12-week lookback window × 5,186 features
    - **Layer 1**: Bidirectional LSTM (96 units) + Batch Normalization + Dropout (0.3)
    - **Layer 2**: Bidirectional LSTM (48 units) + Batch Normalization
    - **Attention**: Multi-Head Attention (4 heads) + Global Average Pooling
    - **Dense**: 48 units (ReLU) + Batch Normalization + Dropout (0.3)
    - **Output**: Dense (1, sigmoid) — binary classification
    - **Loss**: Binary Crossentropy
    - **Features**: BTC + Macro + More + Sentiment
    """)

    st.markdown("---")

    # Model metrics
    st.subheader("Model3 Performance (Best Accuracy)")
    metrics = load_metrics()
    if 'lstm_model3' in metrics.get('models', {}):
        display_model_metrics(metrics['models']['lstm_model3'], "LSTM Model3")
    else:
        # Show hardcoded metrics from metadata
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", "68.6%")
        col2.metric("AUC", "0.719")
        col3.metric("F1 Score", "0.556")
        col4.metric("Features", "5,186")

    st.markdown("---")

    # Predictions summary
    st.subheader("📊 Test Period Predictions")
    lstm_pred_path = Path("data/predictions/predictions_lstm.csv")
    if lstm_pred_path.exists():
        lstm_df = pd.read_csv(lstm_pred_path)
        lstm_df['date'] = pd.to_datetime(lstm_df['date'])

        total = len(lstm_df)
        buys = (lstm_df['trading_signal'] == 1).sum()
        sells = (lstm_df['trading_signal'] == -1).sum()
        holds = (lstm_df['trading_signal'] == 0).sum()
        correct = ((lstm_df['y_pred_binary'] == (lstm_df['y_actual'] > 0).astype(int))).sum()

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Test Weeks", total)
        col2.metric("Correct", f"{correct}/{total}")
        col3.metric("🟢 BUY", int(buys))
        col4.metric("🔴 SELL", int(sells))
        col5.metric("⚪ HOLD", int(holds))

        # Show predictions table
        with st.expander("View All Predictions", expanded=False):
            show_df = lstm_df.copy()
            show_df['Actual Direction'] = show_df['y_actual'].apply(lambda x: '🟢 UP' if x > 0 else '🔴 DOWN')
            show_df['Predicted'] = show_df['y_pred_binary'].apply(lambda x: '🟢 UP' if x == 1 else '🔴 DOWN')
            show_df['Signal'] = show_df['trading_signal'].map({1.0: '🟢 BUY', -1.0: '🔴 SELL', 0.0: '⚪ HOLD'})
            show_df['Correct'] = (show_df['y_pred_binary'] == (show_df['y_actual'] > 0).astype(int)).map({True: '✅', False: '❌'})
            show_df['Date'] = show_df['date'].dt.strftime('%Y-%m-%d')
            show_df['Confidence'] = show_df['confidence'].apply(lambda x: f"{x:.4f}")
            st.dataframe(show_df[['Date', 'Actual Direction', 'Predicted', 'Confidence', 'Signal', 'Correct']], use_container_width=True)
    else:
        st.warning("LSTM predictions not available. Run generate_lstm_predictions.py on a computer with TensorFlow 2.16+.")

    st.markdown("---")

    # Generated figures section
    st.subheader("📊 Model Visuals")
    st.write("**Model Comparison Plot**")
    fig_path = Path("/Users/charlotteho/Desktop/fyp/figure_lstm/lstm_model_comparison.png")
    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)
    else:
        st.info("🔄 To be updated — LSTM Base / Model1 / Model2 metrics not yet available.")

    st.write("**Confusion Matrices**")
    cm_cols = st.columns(4)
    for i, (vk, vlabel) in enumerate([("base", "Base"), ("model1", "Model 1"), ("model2", "Model 2"), ("model3", "Model 3")]):
        with cm_cols[i]:
            st.caption(vlabel)
            cm_path = Path(f"/Users/charlotteho/Desktop/fyp/figure_lstm/lstm_{vk}_confusion_matrix.png")
            if cm_path.exists():
                st.image(str(cm_path), use_container_width=True)
            else:
                st.info("🔄 To be updated")

# ========== TAB 6: MODEL COMPARISON ==========
elif selected_tab == "🎯 Model Comparison":
    st.header("🎯 Cross-Model Comparison")
    st.markdown("<span style='color:gray'><b>Note:</b> Accuracy, AUC, and F1 Score are calculated based on the test period: <b>2024-09-13 to 2025-08-31</b> for each model variant.</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Cross-Model Comparison Figures")
    compare_fig_dir = Path("/Users/charlotteho/Desktop/fyp/figure_compare")
    variant_keys = ["base", "model1", "model2", "model3"]
    variant_labels = ["Base Model", "Model 1", "Model 2", "Model 3"]
    for row in range(2):
        fig_cols = st.columns(2)
        for col in range(2):
            idx = row * 2 + col
            vk = variant_keys[idx]
            vlabel = variant_labels[idx]
            fig_path = compare_fig_dir / f"cross_model_{vk}_comparison.png"
            with fig_cols[col]:
                st.caption(vlabel)
                if fig_path.exists():
                    st.image(str(fig_path), use_container_width=True)
                else:
                    st.info("Figure not yet generated")

    metrics = load_metrics()

    # Build comparison dataframe
    comparison_data = []

    for model_key, model_metrics in metrics.get('models', {}).items():
        comparison_data.append({
            'Model': model_key.upper(),
            'Accuracy': model_metrics.get('accuracy'),
            'AUC': model_metrics.get('auc'),
            'F1 Score': model_metrics.get('f1'),
            'Type': model_metrics.get('model_type', 'Unknown'),
            'Features': model_metrics.get('features', 0)
        })

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)

        # Display table
        st.subheader("All Models Performance Metrics")
        st.dataframe(comparison_df, use_container_width=True)

        # Comparison charts
        col1, col2, col3 = st.columns(3)

        with col1:
            sorted_acc = comparison_df.sort_values('Accuracy', ascending=False)
            fig = px.bar(
                sorted_acc,
                x='Model',
                y='Accuracy',
                color='Model',
                title='Accuracy Comparison'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            sorted_auc = comparison_df.sort_values('AUC', ascending=False)
            fig = px.bar(
                sorted_auc,
                x='Model',
                y='AUC',
                color='Model',
                title='AUC Comparison'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            sorted_f1 = comparison_df.sort_values('F1 Score', ascending=False)
            fig = px.bar(
                sorted_f1,
                x='Model',
                y='F1 Score',
                color='Model',
                title='F1 Score Comparison'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No metrics available for comparison")

    st.markdown("---")

    # Advanced Section: Walk-Forward Analysis

# ========== TRADING SIMULATION TAB ==========
elif selected_tab == "💵 Trading Simulation":
    st.header("💵 Trading Simulation with ML Models")
    st.markdown("Compare trading simulations across different ML models.")

    # Load model predictions
    @st.cache_data
    def load_model_predictions():
        pred_files = {
            "Logistic Regression": "data/predictions/predictions_linear_regression.csv",
            "Random Forest": "data/predictions/predictions_random_forest.csv",
            "XGBoost": "data/predictions/predictions_xgboost.csv",
            "LSTM": "data/predictions/predictions_lstm.csv",
        }

        loaded_models = {}
        for model_name, filepath in pred_files.items():
            try:
                df = pd.read_csv(filepath)
                df['date'] = pd.to_datetime(df['date']).dt.date
                loaded_models[model_name] = df
            except:
                pass
        return loaded_models

    @st.cache_data
    def load_price_data():
        price_data = pd.read_csv("data/BTC-USD.csv", parse_dates=["DATE"])
        # Add date column for compatibility
        price_data['date'] = price_data['DATE'].dt.date
        return price_data

    model_predictions = load_model_predictions()
    price_data = load_price_data()

    # Load metrics and select best accuracy variant for each model type
    metrics = load_metrics()
    model_variant_map = {}
    best_acc = {}
    for key, m in metrics.get('models', {}).items():
        model_type = m.get('model_type', '').lower()
        acc = m.get('accuracy', 0)
        if 'logistic' in model_type:
            if 'Logistic Regression' not in best_acc or acc > best_acc['Logistic Regression']:
                best_acc['Logistic Regression'] = acc
                model_variant_map['Logistic Regression'] = key
        elif 'random forest' in model_type:
            if 'Random Forest' not in best_acc or acc > best_acc['Random Forest']:
                best_acc['Random Forest'] = acc
                model_variant_map['Random Forest'] = key
        elif 'xgboost' in model_type:
            if 'XGBoost' not in best_acc or acc > best_acc['XGBoost']:
                best_acc['XGBoost'] = acc
                model_variant_map['XGBoost'] = key
        elif 'lstm' in model_type:
            if 'LSTM' not in best_acc or acc > best_acc['LSTM']:
                best_acc['LSTM'] = acc
                model_variant_map['LSTM'] = key

    # Check which models are available
    available_models = list(model_predictions.keys())
    all_models = available_models

    # Track select-all state
    select_all = st.session_state.get("sim_select_all", False)

    # Model selection
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        if select_all:
            selected_models = st.multiselect(
                "Select Models to Compare",
                all_models,
                default=all_models,
                help="Choose which models to simulate."
            )
        else:
            selected_models = st.multiselect(
                "Select Models to Compare",
                all_models,
                default=[available_models[0]] if available_models else [],
                help="Choose which models to simulate."
            )
        select_all = st.checkbox("Select All Models", value=False, key="sim_select_all")

    with col2:
        initial_capital = st.number_input(
            "Initial Capital ($)",
            value=100000,
            min_value=1000,
            step=10000
        )

    with col3:
        st.markdown("")  # Spacing
        if st.button("🔄 Run Simulations", use_container_width=True):
            st.rerun()

    st.markdown("**Strategy:** Buy (signal=1) when predicted positive, Sell (signal=-1) when predicted negative.")
    st.markdown("---")

    # Show which variant is used for each model
    if selected_models:
        variant_info = []
        for model_name in selected_models:
            variant = model_variant_map.get(model_name, 'Unknown')
            variant_info.append(f"<b>{model_name}</b>: <span style='color:#1a73e8'>{variant}</span>")
        st.markdown("<b>Model Variant Used:</b> " + " | ".join(variant_info), unsafe_allow_html=True)

    if not selected_models:
        st.warning("Please select at least one model to simulate")
    else:
        try:
            from app_code.trading_simulator import simulate_trading

            sim_results_dict = {}

            # Run simulations for each selected model
            for model_name in selected_models:
                if model_name in model_predictions:
                    pred_df = model_predictions[model_name].copy()
                    pred_df['Predicted_Label'] = pred_df['trading_signal']
                    # Use same filtered price data as other models (Sep 13, 2024 - Aug 31, 2025)
                    test_start = pd.to_datetime("2024-09-13").date()
                    test_end = pd.to_datetime("2025-08-31").date()
                    price_data_filtered = price_data[
                        (price_data['DATE'] >= pd.to_datetime(test_start)) & 
                        (price_data['DATE'] <= pd.to_datetime(test_end))
                    ].copy()
                else:
                    continue  # Model not found

                # Run simulation
                try:
                    sim_result = simulate_trading(price_data_filtered, pred_df, initial_capital)
                    sim_results_dict[model_name] = sim_result
                except Exception as e:
                    st.error(f"Error simulating {model_name}: {e}")

            if sim_results_dict:
                # Create comparison table
                st.subheader("📊 Model Comparison")

                comparison_rows = []
                for model_name, results in sim_results_dict.items():
                    comparison_rows.append({
                        "Model": model_name,
                        "Final Value": f"${results['final_value']:,.0f}",
                        "ROI": f"{results['roi_percent']:.2f}%",
                        "Profit/Loss": f"${results['profit_loss']:,.0f}",
                        "vs Buy & Hold": f"{results['roi_percent'] - results['buy_hold_roi']:.2f}%",
                        "Trades": int(results['num_trades'])
                    })

                comparison_df = pd.DataFrame(comparison_rows)
                st.dataframe(comparison_df, use_container_width=True)

                # Model results (left) vs Buy & Hold (right)
                if sim_results_dict:
                    st.markdown("---")
                    left_col, right_col = st.columns([2, 1])

                    with left_col:
                        st.subheader("📊 Model Performance")
                        model_cols = st.columns(len(sim_results_dict))
                        for mc, (model_name, results) in zip(model_cols, sim_results_dict.items()):
                            with mc:
                                st.metric(model_name, f"${results['final_value']:,.0f}", f"{results['roi_percent']:.2f}% ROI")

                    with right_col:
                        first_result = list(sim_results_dict.values())[0]
                        st.subheader("📈 Buy & Hold")
                        st.metric("Final Value", f"${first_result['buy_hold_value']:,.0f}", f"{first_result['buy_hold_roi']:.2f}% ROI")
                    
                    # Test Period info
                    st.markdown("---")
                    start_price = first_result.get('price_range', {}).get('start', 'N/A')
                    end_price = first_result.get('price_range', {}).get('end', 'N/A')
                    try:
                        start_str = f"${float(start_price):,.0f}"
                        end_str = f"${float(end_price):,.0f}"
                    except (ValueError, TypeError):
                        start_str = str(start_price)
                        end_str = str(end_price)
                    st.info(f"**Test Period BTC Price:** {start_str} → {end_str}")

            # Detailed results for selected models
            st.markdown("---")
            st.subheader("📈 Detailed Performance")

            # Tabs for each model
            model_tabs = st.tabs(selected_models)

            for tab, model_name in zip(model_tabs, selected_models):
                with tab:
                    if model_name in sim_results_dict:
                        result = sim_results_dict[model_name]

                        # Key metrics
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Final Value", f"${result['final_value']:,.0f}", f"${result['profit_loss']:,.0f}")
                        with col2:
                            st.metric("ROI", f"{result['roi_percent']:.2f}%", f"{result['roi_percent'] - result['buy_hold_roi']:.2f}% vs B&H")
                        with col3:
                            st.metric("Total Trades", int(result['num_trades']))
                        with col4:
                            st.metric("Final BTC", f"{result['final_btc_holdings']:.4f}")
                        with col5:
                            st.metric("Final Cash", f"${result['final_cash']:,.0f}")

                        st.markdown("---")

                        # Portfolio value chart
                        if not result['portfolio_history'].empty:
                            portfolio_hist = result['portfolio_history'].copy()
                            portfolio_hist['date'] = pd.to_datetime(portfolio_hist['date'])

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=portfolio_hist['date'],
                                y=portfolio_hist['portfolio_value'],
                                mode='lines',
                                name='Portfolio Value (Strategy)',
                                line=dict(color='#1f77b4', width=2)
                            ))
                            
                            # Add BTC value over time (buy-and-hold benchmark)
                            if 'btc_value_over_time' in portfolio_hist.columns:
                                fig.add_trace(go.Scatter(
                                    x=portfolio_hist['date'],
                                    y=portfolio_hist['btc_value_over_time'],
                                    mode='lines',
                                    name='BTC Value (Buy & Hold)',
                                    line=dict(color='#ff7f0e', width=2, dash='dash')
                                ))

                            # Add BUY markers (actual executed buys only)
                            action_col = 'actual_action' if 'actual_action' in portfolio_hist.columns else 'prediction'
                            buys = portfolio_hist[portfolio_hist[action_col] == 1]
                            if not buys.empty:
                                fig.add_trace(go.Scatter(
                                    x=buys['date'],
                                    y=buys['portfolio_value'],
                                    mode='markers',
                                    name='BUY',
                                    marker=dict(symbol='triangle-up', size=12, color='#2ecc71', line=dict(width=1, color='#27ae60')),
                                    hovertemplate='BUY<br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
                                ))

                            # Add SELL markers (actual executed sells only)
                            sells = portfolio_hist[portfolio_hist[action_col] == -1]
                            if not sells.empty:
                                fig.add_trace(go.Scatter(
                                    x=sells['date'],
                                    y=sells['portfolio_value'],
                                    mode='markers',
                                    name='SELL',
                                    marker=dict(symbol='triangle-down', size=12, color='#e74c3c', line=dict(width=1, color='#c0392b')),
                                    hovertemplate='SELL<br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
                                ))
                            
                            fig.add_hline(
                                y=initial_capital,
                                line_dash="dash",
                                line_color="gray",
                                annotation_text="Initial Capital"
                            )
                            fig.update_layout(
                                title=f'{model_name} - Portfolio Value Over Time',
                                xaxis_title='Date',
                                yaxis_title='Value (USD)',
                                hovermode='x unified',
                                height=450
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # Trading action log
                        if not result['portfolio_history'].empty:
                            full_hist = result['portfolio_history'].copy()
                            full_hist['date'] = pd.to_datetime(full_hist['date']).dt.date

                            # Build actual action column (fallback if old simulator without actual_action)
                            if 'actual_action' not in full_hist.columns:
                                full_hist['actual_action'] = full_hist['prediction']

                            # Show trade actions only (BUY/SELL) and full history toggle
                            show_all = st.checkbox("Show all periods (including HOLD)", value=False, key=f"show_all_{model_name}")
                            if show_all:
                                display_df = full_hist.copy()
                            else:
                                display_df = full_hist[full_hist['prediction'] != 0].copy()

                            if not display_df.empty:
                                # Map actual executed actions (not just signals)
                                def format_action(row):
                                    sig = row['prediction']
                                    act = row['actual_action']
                                    if act == 1:
                                        return '🟢 BUY'
                                    elif act == -1:
                                        return '🔴 SELL'
                                    elif sig == 1:
                                        return '⚪ HOLD (no cash)'
                                    elif sig == -1:
                                        return '⚪ HOLD (no BTC)'
                                    else:
                                        return '⚪ HOLD'

                                display_df['action_label'] = display_df.apply(format_action, axis=1)
                                display_df = display_df[['date', 'price', 'action_label', 'btc_holdings', 'cash', 'portfolio_value']]
                                display_df.columns = ['Date', 'BTC Price', 'Action', 'BTC Holdings', 'Cash', 'Portfolio Value']
                                display_df['BTC Price'] = display_df['BTC Price'].apply(lambda x: f"${x:,.2f}")
                                display_df['BTC Holdings'] = display_df['BTC Holdings'].apply(lambda x: f"{x:.4f}")
                                display_df['Cash'] = display_df['Cash'].apply(lambda x: f"${x:,.2f}")
                                display_df['Portfolio Value'] = display_df['Portfolio Value'].apply(lambda x: f"${x:,.2f}")
                                st.dataframe(display_df, use_container_width=True)

                                # Summary of actual actions
                                total_buys = len(full_hist[full_hist['actual_action'] == 1])
                                total_sells = len(full_hist[full_hist['actual_action'] == -1])
                                total_holds = len(full_hist) - total_buys - total_sells
                                total_skipped_buys = len(full_hist[(full_hist['prediction'] == 1) & (full_hist['actual_action'] == 0)])
                                total_skipped_sells = len(full_hist[(full_hist['prediction'] == -1) & (full_hist['actual_action'] == 0)])
                                summary = f"Total: 🟢 {total_buys} BUY · 🔴 {total_sells} SELL · ⚪ {total_holds} HOLD"
                                if total_skipped_buys > 0 or total_skipped_sells > 0:
                                    summary += f" (skipped: {total_skipped_buys} buy, {total_skipped_sells} sell — insufficient funds/holdings)"
                                st.caption(summary)
                            else:
                                st.info("No trades were executed.")

        except Exception as e:
            st.error(f"Error running simulation: {e}")
            import traceback
            st.write(traceback.format_exc())


# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #95a5a6; font-size: 0.9rem;">
    <p>📊 Weekly Bitcoin Price Analysis | Last Updated: 2026-02-05</p>
    <p>Framework: Logistic Regression, Random Forest, XGBoost | Mode: Classification</p>
</div>
""", unsafe_allow_html=True)
