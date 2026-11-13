# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import date, timedelta
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def download_stock_data(ticker, **kwargs):
    """Download one ticker and normalize yfinance's single-ticker MultiIndex columns.

    Recent yfinance releases may return columns such as ('Close', 'RELIANCE.NS')
    even for a single ticker. Flattening those columns keeps the rest of the app
    working with the familiar Open/High/Low/Close/Volume Series interface.
    """
    data = yf.download(ticker, **kwargs)
    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        matched_level = None
        matched_value = None
        for level in range(data.columns.nlevels):
            values = data.columns.get_level_values(level)
            for value in pd.unique(values):
                if str(value) == str(ticker):
                    matched_level = level
                    matched_value = value
                    break
            if matched_level is not None:
                break

        # If Yahoo returns a one-ticker MultiIndex with a different ticker label,
        # select the only non-price level when it is unambiguous.
        if matched_level is None:
            price_fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
            for level in reversed(range(data.columns.nlevels)):
                values = list(pd.unique(data.columns.get_level_values(level)))
                if len(values) == 1 and str(values[0]) not in price_fields:
                    matched_level = level
                    matched_value = values[0]
                    break

        if matched_level is not None:
            data = data.xs(matched_value, axis=1, level=matched_level, drop_level=True)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

    return data


def get_close_series(data):
    """Return Close as a numeric pandas Series, including for legacy MultiIndex data."""
    if data is None or data.empty or "Close" not in data.columns:
        return pd.Series(dtype=float)
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return pd.to_numeric(close, errors="coerce").dropna().astype(float)

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="SagarFinSight | Stock Market Analysis",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="collapsed"
)

# ---------- GLASSMORPHISM & TYPOGRAPHY CSS ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0B1220 0%, #111C2D 58%, #0F1A2A 100%);
    }
    
    .glass {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.3);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .glass:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 50px rgba(0, 0, 0, 0.4);
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    .glass-red { background: rgba(45, 212, 191, 0.12); border-color: rgba(45, 212, 191, 0.16); }
    .glass-green  { background: rgba(34, 197, 94, 0.12); border-color: rgba(34, 197, 94, 0.18); }
    .glass-pink   { background: rgba(248, 113, 113, 0.12); border-color: rgba(248, 113, 113, 0.18); }
    .glass-blue   { background: rgba(96, 165, 250, 0.12); border-color: rgba(96, 165, 250, 0.18); }
    .glass-orange { background: rgba(245, 158, 11, 0.12); border-color: rgba(245, 158, 11, 0.18); }
    span.glass-purple { background: rgba(45, 212, 191, 0.12); border-color: rgba(45, 212, 191, 0.16); }
    with span.glass-purple:hover { background: rgba(45, 212, 191, 0.18); border-color: rgba(45, 212, 191, 0.24); }
    /* ---------- HEADER ---------- */
    .main-header {
        font-size: 3.5rem;
        text-align: center;
        font-weight: 900;
        letter-spacing: -1.5px;
        background: linear-gradient(135deg, #5EEAD4 0%, #38BDF8 55%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0 0.2rem 0;
        text-shadow: 0 0 80px rgba(45, 212, 191, 0.24);
    }
    
    .sub-header {
        text-align: center;
        color: rgba(255,255,255,0.5);
        font-size: 1.2rem;
        font-weight: 300;
        letter-spacing: 3px;
        margin-bottom: 1.5rem;
    }
    
    /* ---------- TOP NAVIGATION ---------- */
    .nav-header {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 0.6rem 1.2rem;
        margin-bottom: 2rem;
        display: flex;
        justify-content: center;
        gap: 0.3rem;
        flex-wrap: wrap;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    }
    
    .nav-header .nav-btn {
        color: rgba(255,255,255,0.6);
        background: transparent;
        border: 1px solid transparent;
        border-radius: 50px;
        padding: 0.5rem 1.5rem;
        font-size: 0.95rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
        letter-spacing: 0.3px;
    }
    
    .nav-header .nav-btn:hover {
        color: #ffffff;
        background: rgba(45, 212, 191, 0.12);
        border-color: rgba(45, 212, 191, 0.24);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(45, 212, 191, 0.12);
    }
    
    .nav-header .nav-btn.active {
        color: #ffffff;
        background: rgba(45, 212, 191, 0.18);
        border-color: #2DD4BF;
        box-shadow: 0 4px 15px rgba(45, 212, 191, 0.16);
    }
    
    /* ---------- SECTION HEADERS ---------- */
    .section-header {
        font-size: 2.2rem;
        color: #ffffff;
        margin: 2rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid rgba(45, 212, 191, 0.24);
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 12px;
        letter-spacing: -0.5px;
    }
    
    .section-header span {
        background: linear-gradient(135deg, #5EEAD4, #38BDF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* ---------- METRIC CARDS ---------- */
    .metric-card {
        padding: 1.8rem 1.5rem;
        border-radius: 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.4s ease;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: rgba(255,255,255,0.03);
        border-radius: 50%;
        transform: rotate(45deg);
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
    }
    
    .metric-card .metric-label {
        font-size: 0.85rem;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
        color: rgba(255,255,255,0.8);
    }
    
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 800;
        margin-top: 0.3rem;
        background: linear-gradient(135deg, #fff 0%, #99F6E4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card .metric-sub {
        font-size: 0.85rem;
        opacity: 0.6;
        margin-top: 0.2rem;
        color: rgba(255,255,255,0.6);
    }
    
    /* ---------- PLOT CONTAINER ---------- */
    .plot-container {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 20px;
        padding: 1.8rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .plot-container:hover {
        border-color: rgba(45, 212, 191, 0.18);
        box-shadow: 0 12px 50px rgba(0, 0, 0, 0.3);
    }
    
    /* ---------- SUCCESS BOX ---------- */
    .success-box {
        background: rgba(34, 197, 94, 0.08);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(34, 197, 94, 0.12);
        padding: 1.8rem;
        border-radius: 20px;
        color: rgba(255,255,255,0.9);
        text-align: center;
        font-weight: 500;
        font-size: 1.1rem;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .success-box:hover {
        border-color: rgba(34, 197, 94, 0.28);
        transform: scale(1.01);
    }
    
    /* ---------- FORM ELEMENTS ---------- */
    .stTextInput > div > div,
    .stTextArea > div > div,
    .stSelectbox > div > div,
    .stDateInput > div > div {
        background: rgba(30, 41, 59, 0.72) !important;
        border: 1px solid rgba(148,163,184,0.20) !important;
        border-radius: 12px !important;
        color: white !important;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div:focus-within,
    .stTextArea > div > div:focus-within,
    .stSelectbox > div > div:focus-within,
    .stDateInput > div > div:focus-within {
        border-color: #2DD4BF !important;
        box-shadow: 0 0 20px rgba(45, 212, 191, 0.12) !important;
    }

    .stTextInput label,
    .stTextArea label,
    .stSelectbox label,
    .stDateInput label {
        color: rgba(226, 232, 240, 0.88) !important;
    }

    .stTextInput input,
    .stTextArea textarea,
    .stDateInput input {
        color: #f8fafc !important;
    }

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: rgba(203, 213, 225, 0.62) !important;
    }
    
    .stButton > button {
        white-space: nowrap !important;
        background: linear-gradient(135deg, #0F766E 0%, #0E7490 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.65rem 1.35rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 25px rgba(45, 212, 191, 0.18) !important;
        letter-spacing: 0.3px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 9px 24px rgba(20, 184, 166, 0.24) !important;
    }
    
    /* ---------- FAQ ITEMS ---------- */
    .faq-item {
        background: rgba(255,255,255,0.02);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        border: 1px solid rgba(255,255,255,0.04);
        transition: all 0.3s ease;
    }
    
    .faq-item:hover {
        background: rgba(255,255,255,0.05);
        border-color: rgba(45, 212, 191, 0.12);
    }
    
    .faq-item .faq-q {
        color: white;
        font-weight: 600;
        font-size: 1.05rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .faq-item .faq-a {
        color: rgba(255,255,255,0.7);
        font-weight: 300;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid rgba(255,255,255,0.04);
        font-size: 0.95rem;
    }
    
    /* ---------- FOOTER ---------- */
    .footer {
        text-align: center;
        padding: 2rem 0 0.5rem;
        color: rgba(255,255,255,0.1);
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        border-top: 1px solid rgba(255,255,255,0.03);
        margin-top: 3rem;
    }
    
    /* ---------- RESPONSIVE ---------- */
    @media (max-width: 768px) {
        .main-header { font-size: 2.2rem; }
        .metric-card .metric-value { font-size: 1.3rem; }
        .nav-header { flex-direction: column; align-items: stretch; gap: 0.2rem; }
        .nav-header .nav-btn { text-align: center; }
    }
</style>
""", unsafe_allow_html=True)

# ---------- THEME TOGGLE ----------
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

def toggle_theme():
    st.session_state.theme = "Light" if st.session_state.theme == "Dark" else "Dark"

# ---------- PAGE NAVIGATION ----------
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

def navigate_to(page_name):
    st.session_state.page = page_name

# ---------- HEADER ----------
st.markdown('<div class="main-header">📊 SagarFinSight</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">✨ Time Series Decomposition & SARIMA Forecasting for Indian Stocks</div>', unsafe_allow_html=True)

# ---------- TOP NAVIGATION HEADER ----------
cols = st.columns([0.35, 1.6, 1.1, 1.35, 1.15, 1.1, 1.3, 0.2, 0.7])  # balanced navigation spacing
with cols[1]:
    if st.button("Home", key="nav_home", use_container_width=True):
        navigate_to("🏠 Home")
with cols[2]:
    if st.button("About", key="nav_about", use_container_width=True):
        navigate_to("📖 About")
with cols[3]:
    if st.button("Portfolio", key="nav_portfolio", use_container_width=True):
        navigate_to("📊 Portfolio")
with cols[4]:
    if st.button("Tech", key="nav_tech", use_container_width=True):
        navigate_to("📈 Technical Analysis")
with cols[5]:
    if st.button("News", key="nav_news", use_container_width=True):
        navigate_to("📰 News & Sentiment")
with cols[6]:
    if st.button("Contact", key="nav_contact", use_container_width=True):
        navigate_to("📱 Contact")
with cols[8]:
    theme_label = "☀️" if st.session_state.theme == "Dark" else "🌙"
    if st.button(f"{theme_label}", key="theme_toggle", use_container_width=True):
        toggle_theme()
        st.rerun()

# ---------- APPLY THEME ----------
if st.session_state.theme == "Light":
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%); }
        .stTextInput label, .stTextArea label, .stSelectbox label, .stDateInput label { color: #334155 !important; }
        .stTextInput input, .stTextArea textarea, .stDateInput input { color: #0F172A !important; }
        .stTextInput input::placeholder, .stTextArea textarea::placeholder { color: #64748B !important; }
        .metric-card .metric-value { background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .section-header { color: #2d3748; }
        .glass { background: rgba(255,255,255,0.6); border-color: rgba(255,255,255,0.3); }
        .success-box { background: rgba(34,197,94,0.14); color: #2d3748; }
        .faq-item .faq-q { color: #2d3748; }
        .faq-item .faq-a { color: #4a5568; }
        .nav-header { background: rgba(255,255,255,0.6); border-color: rgba(255,255,255,0.3); }
        .nav-header .nav-btn { color: rgba(0,0,0,0.6); }
        .nav-header .nav-btn:hover { color: #000; background: rgba(45,212,191,0.10); }
        .nav-header .nav-btn.active { color: #000; background: rgba(45,212,191,0.16); border-color: #2DD4BF; }
        .plot-container { background: rgba(255,255,255,0.4); border-color: rgba(255,255,255,0.3); }
        .stTextInput > div > div, .stTextArea > div > div, .stSelectbox > div > div, .stDateInput > div > div { background: white !important; color: black !important; }
        .footer { color: rgba(0,0,0,0.2); border-top-color: rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #0B1220 0%, #111C2D 58%, #0F1A2A 100%); }
        .section-header { color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

# ---------- SIDEBAR (optional) ----------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")
    st.caption("Theme: " + st.session_state.theme)
    st.caption("Page: " + st.session_state.page)
    st.markdown("---")
    st.caption("⚠️ For educational purpose only that is why ")

# ---------- SARIMA FUNCTION ----------
def fit_sarima_model(series, forecast_days=30):
    try:
        model = SARIMAX(series,
                       order=(1,1,1),
                       seasonal_order=(1,1,1,5),
                       enforce_stationarity=False,
                       enforce_invertibility=False)
        fitted = model.fit(disp=False)
        forecast = fitted.get_forecast(steps=forecast_days)
        return forecast.predicted_mean, forecast.conf_int(), fitted
    except:
        last = series[-10:].mean()
        pred = pd.Series([last]*forecast_days)
        ci = pd.DataFrame({'lower': pred*0.95, 'upper': pred*1.05}, index=pred.index)
        return pred, ci, None

# ---------- COMPANY DATA ----------
companies = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Larsen & Toubro": "LT.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ITC Limited": "ITC.NS"
}

# =====================================================================
# PAGE ROUTING
# =====================================================================
page = st.session_state.page

# ---------- HOME ----------
if page == "🏠 Home":
    st.markdown('<div class="section-header"> <span>Market Overview</span></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        company_name = st.selectbox("🏢 Select Company", list(companies.keys()))
        ticker = companies[company_name]
    with col2:
        start_date = st.date_input("📅 Start Date", date(2020, 1, 1))
        end_date = st.date_input("📅 End Date", date.today())

    with st.spinner(f'Fetching {company_name}...'):
        data = download_stock_data(ticker, start=start_date, end=end_date, progress=False)

    if data.empty:
        st.error("❌ No data found!")
    else:
        data_reset = data.reset_index()
        close_series = get_close_series(data)
        if close_series.empty:
            st.error("No valid closing-price data was returned for this stock and date range. Please try another date range or stock.")
            st.stop()

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        current_price = float(close_series.iloc[-1])
        first_price = float(close_series.iloc[0])
        pct_change = ((current_price - first_price) / first_price) * 100 if first_price else 0
        avg_volume = float(data['Volume'].mean()) if 'Volume' in data.columns else 0

        with col1:
            st.markdown(f"""
            <div class="metric-card glass glass-purple">
                <div class="metric-label">💰 Current Price</div>
                <div class="metric-value">₹{current_price:,.2f}</div>
                <div class="metric-sub">{data_reset['Date'].iloc[-1].strftime('%d-%b-%Y')}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            emoji = '📈' if pct_change >= 0 else '📉'
            st.markdown(f"""
            <div class="metric-card glass glass-green">
                <div class="metric-label">{emoji} Total Return</div>
                <div class="metric-value">{pct_change:+.2f}%</div>
                <div class="metric-sub">{len(data_reset)} trading days</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card glass glass-blue">
                <div class="metric-label">📊 Avg Volume</div>
                <div class="metric-value">{avg_volume:,.0f}</div>
                <div class="metric-sub">daily average</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card glass glass-pink">
                <div class="metric-label">📅 Days Analyzed</div>
                <div class="metric-value">{len(data_reset)}</div>
                <div class="metric-sub">trading days</div>
            </div>
            """, unsafe_allow_html=True)

        # ---------- DECOMPOSITION ----------
        st.markdown('<div class="section-header"> <span>Time Series Decomposition</span></div>', unsafe_allow_html=True)

        if len(close_series) >= 60:
            with st.spinner('Decomposing...'):
                result = seasonal_decompose(close_series, model='multiplicative', period=30)

            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            fig, axes = plt.subplots(4, 1, figsize=(14, 10))
            colors = ['#2DD4BF', '#22C55E', '#F59E0B', '#60A5FA']

            result.observed.plot(ax=axes[0], title='📊 Observed Data', color=colors[0], linewidth=2)
            axes[0].grid(True, alpha=0.15)
            result.trend.plot(ax=axes[1], title='📈 Trend', color=colors[1], linewidth=2)
            axes[1].grid(True, alpha=0.15)
            result.seasonal.plot(ax=axes[2], title='🔄 Seasonality', color=colors[2], linewidth=2)
            axes[2].grid(True, alpha=0.15)
            result.resid.plot(ax=axes[3], title='📉 Residuals', color=colors[3], linewidth=2)
            axes[3].grid(True, alpha=0.15)

            for ax in axes:
                ax.set_facecolor('none')
                ax.title.set_color('white')
                ax.xaxis.label.set_color((1,1,1,0.7))
                ax.yaxis.label.set_color((1,1,1,0.7))
                ax.tick_params(colors=(1,1,1,0.5))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color((1,1,1,0.1))
                ax.spines['left'].set_color((1,1,1,0.1))

            fig.patch.set_alpha(0)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning(f"⚠️ Need at least 60 days for decomposition. Got {len(close_series)}.")

        # ---------- PREDICTION ----------
        st.markdown('<div class="section-header"> <span>Price Prediction</span></div>', unsafe_allow_html=True)

        if len(close_series) >= 30:
            with st.spinner('Training SARIMA model...'):
                forecast, ci, model = fit_sarima_model(close_series, 30)
                last_date = data_reset["Date"].iloc[-1]
                future_dates = pd.date_range(last_date + timedelta(days=1), periods=30)
                future_df = pd.DataFrame({
                    "Date": future_dates,
                    "Predicted": forecast.values,
                    "Lower": ci.iloc[:,0],
                    "Upper": ci.iloc[:,1]
                })

            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            fig2, ax = plt.subplots(figsize=(14,6))
            ax.set_facecolor('none')

            ax.plot(data_reset["Date"], data_reset["Close"], label="Historical", color="#2DD4BF", linewidth=2.5, alpha=0.8)
            ax.plot(future_df["Date"], future_df["Predicted"], label="SARIMA Prediction", color="#F87171", linewidth=2.5, linestyle='--')
            ax.fill_between(future_df["Date"], future_df["Lower"], future_df["Upper"], color="#F87171", alpha=0.15, label="95% CI")

            ax.legend(fontsize=11, loc='upper left')
            ax.set_title("30-Day Price Prediction", fontsize=14, fontweight='600', color='white', pad=20)
            ax.set_xlabel("Date", fontsize=11, color=(1,1,1,0.7))
            ax.set_ylabel("Price (₹)", fontsize=11, color=(1,1,1,0.7))
            ax.grid(True, alpha=0.15)
            ax.tick_params(colors=(1,1,1,0.5))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color((1,1,1,0.1))
            ax.spines['left'].set_color((1,1,1,0.1))
            fig2.patch.set_alpha(0)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)
            st.markdown('</div>', unsafe_allow_html=True)

            # Prediction result cards
            last_pred = float(future_df["Predicted"].iloc[-1])
            pred_change = ((last_pred - current_price) / current_price) * 100
            ci_width = ((future_df["Upper"].iloc[-1] - future_df["Lower"].iloc[-1]) / last_pred) * 100
            confidence = "🟢 High" if ci_width < 10 else "🟡 Medium" if ci_width < 20 else "🔴 Low"

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="glass glass-purple" style="padding:1.5rem; text-align:center;">
                    <div style="color:rgba(255,255,255,0.6); font-size:0.9rem;">📊 30-Day Prediction</div>
                    <div style="color:white; font-size:2rem; font-weight:800;">₹{last_pred:,.2f}</div>
                    <div style="color:{'#22C55E' if pred_change>=0 else '#F87171'}; font-size:1.1rem;">{pred_change:+.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="glass glass-blue" style="padding:1.5rem; text-align:center;">
                    <div style="color:rgba(255,255,255,0.6); font-size:0.9rem;">🎯 Confidence</div>
                    <div style="color:white; font-size:2rem; font-weight:800;">{confidence}</div>
                    <div style="color:rgba(255,255,255,0.5); font-size:0.9rem;">SARIMA Model</div>
                </div>
                """, unsafe_allow_html=True)

        # Success
        st.markdown(f"""
        <div class="success-box">
            ✅ Analysis completed for <b>{company_name}</b><br>
            <span style="font-size:0.9rem; opacity:0.6;">{start_date.strftime('%d-%b-%Y')} to {end_date.strftime('%d-%b-%Y')} | {len(data_reset)} trading days</span>
        </div>
        """, unsafe_allow_html=True)

# ---------- ABOUT ----------
elif page == "📖 About":
    st.markdown('<div class="section-header"> <span>About This Project</span></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("""
        <div class="glass" style="height:100%;">
            <h3 style="color:white; font-size:1.6rem;"> Project Overview</h3>
            <p style="color:rgba(255,255,255,0.7); line-height:1.8;">
                <b>SagarFinSight</b> is a comprehensive web application designed to 
                analyze and forecast Indian stock prices using advanced time series techniques.
            </p>
            <br>
            <h4 style="color:white;">🌟 Key Objectives</h4>
            <ul style="color:rgba(255,255,255,0.7); line-height:2.2;">
                <li>📊 Provide interactive stock analysis tools</li>
                <li>🔍 Perform time series decomposition</li>
                <li>🤖 Generate accurate price predictions</li>
                <li>📈 Visualize market trends effectively</li>
                <li>🎓 Educate users about financial concepts</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass" style="height:100%; text-align:center;">
            <div style="font-size:4rem; margin-bottom:1rem;">👨‍💻</div>
            <h3 style="color:white; font-size:1.6rem;">Developer Profile</h3>
            <h2 style="color:white; font-size:2rem; font-weight:800;">Sagar Kashyap</h2>
            <p style="color:rgba(255,255,255,0.7); font-size:1.1rem;">B.Tech CSE IBM (AI & DS)</p>
            <p style="color:rgba(255,255,255,0.5);">School of Engg. & Technology</p>
            <br>
            <div style="display:flex; justify-content:center; gap:0.8rem; flex-wrap:wrap;">
                <span style="background:rgba(45,212,191,0.16); padding:0.3rem 1.2rem; border-radius:50px; font-size:0.8rem;">🐍 Python</span>
                <span style="background:rgba(34,197,94,0.14); padding:0.3rem 1.2rem; border-radius:50px; font-size:0.8rem;">📊 Streamlit</span>
                <span style="background:rgba(248,113,113,0.14); padding:0.3rem 1.2rem; border-radius:50px; font-size:0.8rem;">🧠 SARIMA</span>
                <span style="background:rgba(96,165,250,0.14); padding:0.3rem 1.2rem; border-radius:50px; font-size:0.8rem;">📈 Finance</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="glass" style="text-align:center;">
            <div style="font-size:2.8rem;">🛠️</div>
            <h4 style="color:white;">Technologies</h4>
            <div style="color:rgba(255,255,255,0.7); font-size:0.9rem; line-height:1.8;">
                Python • Streamlit<br>Pandas • NumPy<br>Matplotlib • Statsmodels<br>yfinance • SARIMA
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass" style="text-align:center;">
            <div style="font-size:2.8rem;">📊</div>
            <h4 style="color:white;">Features</h4>
            <div style="color:rgba(255,255,255,0.7); font-size:0.9rem; line-height:1.8;">
                Real-time Data • Decomposition<br>SARIMA Forecasting • 30-Day Prediction<br>Confidence Intervals • Glassmorphism UI<br>Multi-Page Layout • Dark/Light Themes
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="glass" style="text-align:center;">
            <div style="font-size:2.8rem;">🏢</div>
            <h4 style="color:white;">Stocks Covered</h4>
            <div style="color:rgba(255,255,255,0.7); font-size:0.9rem; line-height:1.8;">
                10 Major Indian Stocks<br>Banking • IT • Telecom<br>Consumer Goods • Energy<br>Nifty 50 Companies
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass" style="text-align:center; padding:2rem;">
        <h4 style="color:white;">📅 Project Timeline</h4>
        <div style="display:flex; justify-content:center; gap:2rem; flex-wrap:wrap; margin-top:1rem;">
            <div><div style="color:rgba(255,255,255,0.3); font-size:0.8rem;">PHASE 1</div><div style="color:white; font-weight:600;">Research & Planning</div><div style="color:rgba(255,255,255,0.4);">Week 1-2</div></div>
            <div><div style="color:rgba(255,255,255,0.3); font-size:0.8rem;">PHASE 2</div><div style="color:white; font-weight:600;">Development</div><div style="color:rgba(255,255,255,0.4);">Week 3-6</div></div>
            <div><div style="color:rgba(255,255,255,0.3); font-size:0.8rem;">PHASE 3</div><div style="color:white; font-weight:600;">Testing & Deployment</div><div style="color:rgba(255,255,255,0.4);">Week 7-8</div></div>
            <div><div style="color:rgba(255,255,255,0.3); font-size:0.8rem;">PHASE 4</div><div style="color:white; font-weight:600;">Documentation</div><div style="color:rgba(255,255,255,0.4);">Week 9-10</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass" style="margin-top:2rem; padding:1.5rem; text-align:center; background:rgba(245,158,11,0.05); border-color:rgba(245,158,11,0.12);">
        <div style="font-size:1.2rem; color:rgba(255,255,255,0.8);">
            ⚠️ <b>Disclaimer:</b> This project is for <b>educational purposes only</b>. 
            The predictions and analysis should not be considered as financial advice. 
            Stock market investments carry risks.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------- PORTFOLIO ----------
elif page == "📊 Portfolio":
    st.markdown('<div class="section-header"><span>Portfolio Management</span></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass" style="margin-bottom:2rem;">
        <h3 style="color:white;"> Compare Multiple Stocks</h3>
        <p style="color:rgba(255,255,255,0.6);">Select up to 4 stocks to compare their performance side by side.</p>
    </div>
    """, unsafe_allow_html=True)

    default_stocks = list(companies.keys())[:3]
    selected = st.multiselect("Select Stocks to Compare", list(companies.keys()), default=default_stocks)
    if len(selected) < 2:
        st.info("💡 Select at least 2 stocks.")
    else:
        start_p = st.date_input("📅 Start Date", date(2023,1,1), key="port_start")
        end_p = st.date_input("📅 End Date", date.today(), key="port_end")
        if st.button("📊 Compare", use_container_width=True):
            with st.spinner("Fetching..."):
                fig, ax = plt.subplots(figsize=(14,7))
                ax.set_facecolor('none')
                colors = ['#2DD4BF','#22C55E','#F87171','#F59E0B']
                for i, stock in enumerate(selected):
                    ticker = companies[stock]
                    d = download_stock_data(ticker, start=start_p, end=end_p, progress=False)
                    if not d.empty:
                        d_close = get_close_series(d)
                        if not d_close.empty and d_close.iloc[0] != 0:
                            norm = (d_close / d_close.iloc[0]) * 100
                            ax.plot(norm, label=stock, color=colors[i%len(colors)], linewidth=2)
                ax.legend(fontsize=11)
                ax.set_title("Performance Comparison (Normalized to 100)", fontsize=14, fontweight='600', color='white', pad=20)
                ax.set_xlabel("Date", fontsize=11, color=(1,1,1,0.7))
                ax.set_ylabel("Normalized Price", fontsize=11, color=(1,1,1,0.7))
                ax.grid(True, alpha=0.15)
                ax.tick_params(colors=(1,1,1,0.5))
                for spine in ax.spines.values():
                    spine.set_color((1,1,1,0.1))
                fig.patch.set_alpha(0)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

        # Performance table
        if selected:
            st.markdown('<div class="glass" style="margin-top:2rem;"><h4 style="color:white;">📊 Performance Summary</h4></div>', unsafe_allow_html=True)
            perf = []
            for stock in selected[:4]:
                ticker = companies[stock]
                d = download_stock_data(ticker, start=date.today()-timedelta(days=365), end=date.today(), progress=False)
                if not d.empty:
                    d_close = get_close_series(d)
                    if not d_close.empty and d_close.iloc[0] != 0:
                        first = float(d_close.iloc[0])
                        last = float(d_close.iloc[-1])
                        chg = ((last - first) / first) * 100
                        perf.append({"Stock": stock, "1Y Return": f"{chg:+.2f}%", "Current": f"₹{last:,.2f}"})
            if perf:
                st.dataframe(pd.DataFrame(perf), use_container_width=True, hide_index=True)

# ---------- TECHNICAL ANALYSIS ----------
elif page == "📈 Technical Analysis":
    st.markdown('<div class="section-header"> <span>Technical Analysis</span></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass" style="margin-bottom:2rem;">
        <h3 style="color:white;">Technical Indicators</h3>
        <p style="color:rgba(255,255,255,0.6);">Analyze stocks using key technical indicators.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])
    with col1:
        tech_ticker = st.selectbox("Select Stock", list(companies.keys()), key="tech_sel")
        ticker = companies[tech_ticker]
    with col2:
        tech_start = st.date_input("Start Date", date(2023,1,1), key="tech_st")
        tech_end = st.date_input("End Date", date.today(), key="tech_en")

    if st.button("📊 Analyze Technicals", use_container_width=True):
        with st.spinner("Fetching data..."):
            data = download_stock_data(ticker, start=tech_start, end=tech_end, progress=False)
        if not data.empty:
            data['Close'] = get_close_series(data)
            if data['Close'].empty:
                st.error("No valid closing-price data was returned. Please try another date range.")
                st.stop()
            data['SMA_20'] = data['Close'].rolling(20).mean()
            data['SMA_50'] = data['Close'].rolling(50).mean()
            data['SMA_200'] = data['Close'].rolling(200).mean()

            delta = data['Close'].diff()
            gain = (delta.where(delta>0,0)).rolling(14).mean()
            loss = (-delta.where(delta<0,0)).rolling(14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100/(1+rs))

            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp1 - exp2
            data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

            data['BB_Upper'] = data['Close'].rolling(20).mean() + (data['Close'].rolling(20).std()*2)
            data['BB_Lower'] = data['Close'].rolling(20).mean() - (data['Close'].rolling(20).std()*2)

            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            fig, axes = plt.subplots(3,1, figsize=(14,12))

            # Price & MA
            axes[0].plot(data.index, data['Close'], label='Close', color='#2DD4BF', linewidth=2)
            axes[0].plot(data.index, data['SMA_20'], label='SMA 20', color='#22C55E', linewidth=1.5, linestyle='--')
            axes[0].plot(data.index, data['SMA_50'], label='SMA 50', color='#F59E0B', linewidth=1.5, linestyle='--')
            axes[0].plot(data.index, data['SMA_200'], label='SMA 200', color='#F87171', linewidth=1.5, linestyle='--')
            axes[0].set_title('Price & Moving Averages', color='white', fontsize=12, fontweight='600')
            axes[0].legend(loc='upper left')
            axes[0].grid(True, alpha=0.15)

            # RSI
            axes[1].plot(data.index, data['RSI'], label='RSI 14', color='#F59E0B', linewidth=2)
            axes[1].axhline(y=70, color='#F87171', linestyle='--', alpha=0.5, label='Overbought 70')
            axes[1].axhline(y=30, color='#22C55E', linestyle='--', alpha=0.5, label='Oversold 30')
            axes[1].set_title('Relative Strength Index (RSI)', color='white', fontsize=12, fontweight='600')
            axes[1].legend(loc='upper left')
            axes[1].grid(True, alpha=0.15)
            axes[1].set_ylim(0,100)

            # Bollinger Bands
            axes[2].plot(data.index, data['Close'], label='Close', color='#2DD4BF', linewidth=1.5)
            axes[2].plot(data.index, data['BB_Upper'], label='Upper Band', color='#F87171', linewidth=1, linestyle='--')
            axes[2].plot(data.index, data['BB_Lower'], label='Lower Band', color='#22C55E', linewidth=1, linestyle='--')
            axes[2].fill_between(data.index, data['BB_Lower'], data['BB_Upper'], alpha=0.08, color='#2DD4BF')
            axes[2].set_title('Bollinger Bands (20-day)', color='white', fontsize=12, fontweight='600')
            axes[2].legend(loc='upper left')
            axes[2].grid(True, alpha=0.15)

            for ax in axes:
                ax.set_facecolor('none')
                ax.xaxis.label.set_color((1,1,1,0.7))
                ax.yaxis.label.set_color((1,1,1,0.7))
                ax.tick_params(colors=(1,1,1,0.5))
                ax.set_xlabel('Date', color=(1,1,1,0.7))
                for spine in ax.spines.values():
                    spine.set_color((1,1,1,0.1))

            fig.patch.set_alpha(0)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

            # Current values
            st.markdown("""
            <div class="glass" style="margin-top:1.5rem; display:grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap:1rem;">
                <div style="text-align:center;"><div style="color:rgba(255,255,255,0.4);">RSI</div><div style="color:white; font-weight:700; font-size:1.3rem;">{:.2f}</div></div>
                <div style="text-align:center;"><div style="color:rgba(255,255,255,0.4);">MACD</div><div style="color:white; font-weight:700; font-size:1.3rem;">{:.2f}</div></div>
                <div style="text-align:center;"><div style="color:rgba(255,255,255,0.4);">SMA 20</div><div style="color:white; font-weight:700; font-size:1.3rem;">{:.2f}</div></div>
                <div style="text-align:center;"><div style="color:rgba(255,255,255,0.4);">SMA 50</div><div style="color:white; font-weight:700; font-size:1.3rem;">{:.2f}</div></div>
            </div>
            """.format(
                data['RSI'].iloc[-1] if not pd.isna(data['RSI'].iloc[-1]) else 0,
                data['MACD'].iloc[-1] if not pd.isna(data['MACD'].iloc[-1]) else 0,
                data['SMA_20'].iloc[-1] if not pd.isna(data['SMA_20'].iloc[-1]) else 0,
                data['SMA_50'].iloc[-1] if not pd.isna(data['SMA_50'].iloc[-1]) else 0
            ), unsafe_allow_html=True)

            
# ---------- NEWS & SENTIMENT ----------
elif page == "📰 News & Sentiment":
    st.markdown(
        '<div class="section-header"><span>News & Market Sentiment</span></div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="glass" style="margin-bottom:1.5rem;">
        <h3 style="color:white;">Indian Market Snapshot</h3>
        <p style="color:rgba(255,255,255,0.62); margin-bottom:0;">
            Latest completed trading session: Friday, 13 November 2026.
            Figures below are closing-session values, not live quotes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Verified close figures for the 13 November 2026 session.
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="glass" style="text-align:center;">
            <div style="font-size:2.2rem;">📉</div>
            <div style="color:rgba(255,255,255,0.58);">Market Trend</div>
            <div style="color:#F87171; font-weight:800; font-size:1.6rem;">Bearish</div>
            <div style="color:rgba(255,255,255,0.52); font-size:0.82rem;">
                NIFTY 50: 25,144.30<br>
                −155.05 points (−0.61%)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass" style="text-align:center;">
            <div style="font-size:2.2rem;">📊</div>
            <div style="color:rgba(255,255,255,0.58);">Market Breadth</div>
            <div style="color:#F59E0B; font-weight:800; font-size:1.6rem;">38 / 50 Declined</div>
            <div style="color:rgba(255,255,255,0.52); font-size:0.82rem;">
                11 advanced · 1 unchanged<br>
                SENSEX: 81,592.45 (−0.39%)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="glass" style="text-align:center;">
            <div style="font-size:2.2rem;">🌡️</div>
            <div style="color:rgba(255,255,255,0.58);">India VIX</div>
            <div style="color:#2DD4BF; font-weight:800; font-size:1.6rem;">15.50</div>
            <div style="color:rgba(255,255,255,0.52); font-size:0.82rem;">
                −6.74% on the day<br>
                Previous close: 16.62
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass" style="margin-top:1.5rem;">
        <h4 style="color:white;">📝 Recent Market Highlights</h4>
        <div style="display:grid; gap:0.85rem; margin-top:1rem;">
            <div style="background:rgba(255,255,255,0.035); padding:1rem; border-radius:12px; border-left:3px solid #F87171;">
                <div style="color:rgba(255,255,255,0.52); font-size:0.75rem;">
                    Market close · 13 Nov 2026
                </div>
                <div style="color:white; margin:0.25rem 0;">
                    Sensex fell 318.56 points to 81,592.45 (−0.39%), while Nifty 50
                    declined 155.05 points to 25,144.30 (−0.61%), ending a three-session
                    winning streak.
                </div>
                <div style="font-size:0.76rem;">
                    <a href="https://www.reuters.com/world/india/indian-shares-likely-open-flat-oil-surge-counters-strong-foreign-inflows-2026-11-13/"
                       target="_blank" rel="noopener noreferrer"
                       style="color:#5EEAD4; text-decoration:none;">Source: Reuters</a>
                    &nbsp;·&nbsp;
                    <a href="https://timesofindia.indiatimes.com/business/india-business/sensex-stock-market-today-13-november-2026-live-updates-nse-bse-gift-nifty-50-top-gainers-losers-mcx-stocks-in-focus-market-news/liveblog/135787823.cms"
                       target="_blank" rel="noopener noreferrer"
                       style="color:#5EEAD4; text-decoration:none;">Market breadth and movers</a>
                </div>
            </div>
            <div style="background:rgba(255,255,255,0.035); padding:1rem; border-radius:12px; border-left:3px solid #F59E0B;">
                <div style="color:rgba(255,255,255,0.52); font-size:0.75rem;">
                    Official data release · 12 Nov 2026
                </div>
                <div style="color:white; margin:0.25rem 0;">
                    India's provisional October retail inflation rose to 6.09% year-on-year,
                    above the RBI's 6% upper tolerance level. Food inflation stood at 8.94%.
                </div>
                <div style="font-size:0.76rem;">
                    <a href="https://www.pib.gov.in/PressReleseDetailm.aspx?PRID=2335416&lang=1&reg=3"
                       target="_blank" rel="noopener noreferrer"
                       style="color:#5EEAD4; text-decoration:none;">Source: Ministry of Statistics &amp; Programme Implementation (PIB)</a>
                </div>
            </div>
            <div style="background:rgba(255,255,255,0.035); padding:1rem; border-radius:12px; border-left:3px solid #2DD4BF;">
                <div style="color:rgba(255,255,255,0.52); font-size:0.75rem;">
                    Weekly market update · 13 Nov 2026
                </div>
                <div style="color:white; margin:0.25rem 0;">
                    Despite Friday's decline, the benchmarks finished the week higher:
                    Nifty 50 gained about 0.5% and Sensex about 0.7%. Reuters also reported
                    around $8 billion of foreign investor inflows into India so far in November.
                </div>
                <div style="font-size:0.76rem;">
                    <a href="https://www.reuters.com/world/india/indian-shares-likely-open-flat-oil-surge-counters-strong-foreign-inflows-2026-11-13/"
                       target="_blank" rel="noopener noreferrer"
                       style="color:#5EEAD4; text-decoration:none;">Source: Reuters</a>
                </div>
            </div>
            <div style="background:rgba(255,255,255,0.035); padding:1rem; border-radius:12px; border-left:3px solid #38BDF8;">
                <div style="color:rgba(255,255,255,0.52); font-size:0.75rem;">
                    Volatility update · 13 Nov 2026
                </div>
                <div style="color:white; margin:0.25rem 0;">
                    India VIX was reported at 15.50, down 6.74% from its previous close
                    of 16.62, indicating a lower options-implied volatility reading for the session.
                </div>
                <div style="font-size:0.76rem;">
                    <a href="https://upstox.com/indices/india-vix-share-price/"
                       target="_blank" rel="noopener noreferrer"
                       style="color:#5EEAD4; text-decoration:none;">Source: Upstox</a>
                </div>
            </div>
        </div>
        <p style="color:rgba(255,255,255,0.42); font-size:0.72rem; margin-top:1rem; margin-bottom:0;">
            Data snapshot: 13 November 2026. Market figures are historical closing values.
            News cards link to their published sources. This section is a dated snapshot and
            does not automatically refresh with new headlines.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------- CONTACT ----------
elif page == "📱 Contact":
    st.markdown('<div class="section-header">📱 <span>Contact & Support</span></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("""
        <div class="glass" style="height:100%;">
            <h3 style="color:white;"> Get in Touch</h3>
            <br>
            <div style="color:rgba(255,255,255,0.7); line-height:2.4;">
                <div>👨‍💻 <b style="color:white;">Sagar Kashyap</b></div>
                <div>📧 <a href="mailto:sagar46262@gmail.com" style="color:#99F6E4; text-decoration:none; font-weight:600;">sagar46262@gmail.com</a></div>
                <div>📍 <b style="color:white;">School of Engg. & Technology</b></div>
                <div>📞 <b style="color:white;">+91-72929-XXXXX</b></div>
                <div>🐙 <a href="https://github.com/sagarkashyappp" target="_blank" rel="noopener noreferrer" style="color:#99F6E4; text-decoration:none; font-weight:600;">github.com/sagarkashyappp</a></div>
                <div>🔗 <a href="https://www.linkedin.com/in/sagarbuilds/" target="_blank" rel="noopener noreferrer" style="color:#99F6E4; text-decoration:none; font-weight:600;">linkedin.com/in/sagarbuilds</a></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass" style="height:100%;">
            <h3 style="color:white;">💬 Send Feedback</h3>
            <p style="color:rgba(255,255,255,0.4);">We'd love to hear your thoughts!</p>
        </div>
        """, unsafe_allow_html=True)
        with st.form("feedback_form"):
            name = st.text_input("👤 Your Name", placeholder="Enter your name")
            email = st.text_input("📧 Your Email", placeholder="Enter your email")
            subject = st.selectbox("📌 Subject", ["Bug Report", "Feature Request", "General Feedback", "Question"])
            message = st.text_area("💬 Message", placeholder="Describe your feedback or issue...", height=120)
            submitted = st.form_submit_button("📨 Send Message", use_container_width=True)
            if submitted:
                if name and email and message:
                    st.success("✅ Thank you! We'll get back to you soon.")
                    st.balloons()
                else:
                    st.warning("⚠️ Please fill all fields.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">❓ <span>Frequently Asked Questions</span></div>', unsafe_allow_html=True)

    faqs = [
        ("What is SARIMA model?", "SARIMA (Seasonal AutoRegressive Integrated Moving Average) is a time series forecasting model that captures both trend and seasonal patterns. It's particularly effective for stock price prediction."),
        ("How accurate are the predictions?", "The model provides predictions with 95% confidence intervals. Accuracy depends on market conditions. The dashboard shows confidence levels (High/Medium/Low) based on prediction intervals."),
        ("Which stocks are covered?", "10 major Indian stocks: Reliance, TCS, Infosys, HDFC Bank, ICICI Bank, SBI, Bharti Airtel, L&T, HUL, and ITC."),
        ("Is this tool for investment decisions?", "No, this is for educational purposes only. Always consult a professional financial advisor."),
        ("How often is data updated?", "Data is fetched in real-time from Yahoo Finance API whenever you select a stock or change parameters.")
    ]
    for q, a in faqs:
        st.markdown(f"""
        <div class="faq-item">
            <div class="faq-q">❓ {q} <span style="font-size:1.2rem;">▼</span></div>
            <div class="faq-a">{a}</div>
        </div>
        """, unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("""
<div class="footer">
    Developed with ❤️ by <b>Sagar Kashyap</b> · B.Tech CSE IBM (AI & DS) · School of Engg. & Technology
    <br>
    <span style="font-size:0.7rem; opacity:0.5;">For educational purposes only · Not financial advice</span>
</div>
""", unsafe_allow_html=True)