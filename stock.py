from datetime import date
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

# Setup
st.set_page_config(page_title="📈 Stock Forecast Pro", page_icon="💹", layout="wide")

# Navigation Menu
selected = option_menu(
    menu_title="📈 Stock Forecast Pro",
    options=["Home", "Analysis", "Forecast", "About"],
    icons=[
        "house-fill",
        "bar-chart-line-fill",
        "graph-up-arrow",
        "info-circle-fill",
    ],
    default_index=0,
    orientation="horizontal",
)

# Available Stocks
stocks = {
    "Apple Inc. (AAPL) 🇺🇸": "AAPL",
    "AngloGold Ashanti Limited (AU) 🇬🇭": "AU",
    "MTN Group Limited (MTNOY) 🇿🇦": "MTNOY",
    "Vodafone Group PLC (VOD) 🇬🇧": "VOD",
    "Unilever PLC (UL) 🇬🇧": "UL",
    "Airtel Africa PLC (AAF.L) 🇳🇬": "AAF.L",
}


# Load Yahoo Finance data
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    return df


# Get stock currency from Yahoo Finance metadata
@st.cache_data
def get_currency(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("currency", "Unknown Currency")
    except:
        return "Unknown Currency"


# Plotting Utility
def plot_line(data, x_col, y_cols, title):
    fig = go.Figure()
    colors = ["green", "blue", "orange", "purple"]
    for i, col in enumerate(y_cols):
        if col in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data[x_col],
                    y=data[col],
                    name=col,
                    line=dict(color=colors[i % len(colors)]),
                )
            )
    fig.update_layout(title=title, xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


# ----------------- HOME ------------------
if selected == "Home":
    st.title("💹 Welcome to Stock Forecast Pro")
    st.markdown(
        """
        Analyze and forecast stock prices with **machine learning & technical indicators**.

        #### ✅ Features:
        - 📈 Historical chart & candlestick views
        - 🔮 Prophet-powered forecasting with tuning
        - 🧠 ML accuracy metrics (Root Mean Squared Error & Mean Absolute Percentage Error)
        - 🧮 Technical indicators (RSI, MACD, SMA, EMA)
        - 📥 Download forecast data
        """
    )

# ----------------- ANALYSIS ------------------
if selected == "Analysis":
    st.title("📊 Historical Stock Analysis")
    selected_name = st.selectbox("Choose a Stock:", list(stocks.keys()))
    selected_stock = stocks[selected_name]
    currency = get_currency(selected_stock)

    start = st.date_input("Start Date", date(2015, 1, 1))
    end = date.today()

    data = load_data(selected_stock, start, end)

    st.subheader(f"📄 Raw Data: {selected_name}")
    st.caption(f"💱 All prices are shown in **{currency}**.")
    st.write(data.tail())

    # Line Chart
    plot_line(data, "Date", ["Open", "Close"], "📈 Opening & Closing Prices")

    # Candlestick Chart
    st.subheader("📉 Candlestick Chart (Last 30 Days)")
    required_cols = ["Open", "High", "Low", "Close"]
    last_30 = data[required_cols + ["Date"]].dropna().tail(30)

    if last_30.empty:
        st.warning("⚠️ No valid data to plot.")
    else:
        fig_candle = go.Figure(
            data=[
                go.Candlestick(
                    x=last_30["Date"],
                    open=last_30["Open"],
                    high=last_30["High"],
                    low=last_30["Low"],
                    close=last_30["Close"],
                    increasing_line_color="green",
                    decreasing_line_color="red",
                )
            ]
        )
        fig_candle.update_layout(
            title=f"Candlestick: {selected_name}", xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig_candle)

# ----------------- INDICATORS ------------------
if selected == "Indicators":
    st.title("📉 Technical Indicators")
    selected_name = st.selectbox("Select a Stock:", list(stocks.keys()))
    selected_stock = stocks[selected_name]
    currency = get_currency(selected_stock)

    start = st.date_input("Start Date", date(2018, 1, 1), key="ind_date")
    end = date.today()

    df = load_data(selected_stock, start, end)

    df["SMA_20"] = ta.sma(df["Close"], length=20)
    df["EMA_20"] = ta.ema(df["Close"], length=20)
    df["RSI"] = ta.rsi(df["Close"], length=14)

    macd = ta.macd(df["Close"])
    if (
        macd is not None
        and "MACD_12_26_9" in macd.columns
        and "MACDs_12_26_9" in macd.columns
    ):
        df["MACD"] = macd["MACD_12_26_9"]
        df["MACD_Signal"] = macd["MACDs_12_26_9"]
    else:
        df["MACD"] = np.nan
        df["MACD_Signal"] = np.nan

    st.subheader("📈 SMA & EMA")
    plot_line(df, "Date", ["Close", "SMA_20", "EMA_20"], "SMA & EMA")

    st.subheader("💪 RSI (Relative Strength Index)")
    plot_line(df, "Date", ["RSI"], "RSI Indicator")

    st.subheader("📊 MACD")
    plot_line(df, "Date", ["MACD", "MACD_Signal"], "MACD & Signal")

# ----------------- FORECAST ------------------
if selected == "Forecast":
    st.title("🔮 Forecast with Prophet")
    selected_name = st.selectbox("Select Stock for Forecast:", list(stocks.keys()))
    selected_stock = stocks[selected_name]
    currency = get_currency(selected_stock)

    n_years = st.slider("Years of Prediction:", 1, 5)
    period = n_years * 365

    start = st.date_input("Training Start Date", date(2015, 1, 1), key="train_date")
    end = date.today()

    st.info("📡 Loading data...")
    data = load_data(selected_stock, start, end)

    df_train = data[["Date", "Close"]].copy()
    df_train.columns = ["ds", "y"]
    df_train.dropna(inplace=True)

    st.subheader("⚙️ Prophet Settings")
    col1, col2 = st.columns(2)
    with col1:
        seasonality_mode = st.selectbox(
            "Seasonality Mode", ["additive", "multiplicative"]
        )
    with col2:
        changepoint_scale = st.slider("Changepoint Prior Scale", 0.01, 1.0, 0.1)

    with st.spinner("🔮 Training Prophet model..."):
        model = Prophet(
            seasonality_mode=seasonality_mode, changepoint_prior_scale=changepoint_scale
        )
        model.fit(df_train)
        future = model.make_future_dataframe(periods=period)
        forecast = model.predict(future)

    # Rename columns for clarity
    forecast = forecast.rename(
        columns={
            "ds": "Date",
            "yhat": "Predicted Price",
            "yhat_lower": "Lower Confidence Interval",
            "yhat_upper": "Upper Confidence Interval",
        }
    )

    st.subheader(f"📑 Forecast Preview (Prices in {currency})")
    st.write(
        forecast[
            [
                "Date",
                "Predicted Price",
                "Lower Confidence Interval",
                "Upper Confidence Interval",
            ]
        ].tail()
    )

    st.subheader("📈 Forecast Plot")
    fig1 = plot_plotly(
        model,
        forecast.rename(
            columns={
                "Date": "ds",
                "Predicted Price": "yhat",
                "Lower Confidence Interval": "yhat_lower",
                "Upper Confidence Interval": "yhat_upper",
            }
        ),
    )
    st.plotly_chart(fig1)

    st.subheader("📊 Components")
    fig2 = plot_components_plotly(
        model,
        forecast.rename(
            columns={
                "Date": "ds",
                "Predicted Price": "yhat",
                "Lower Confidence Interval": "yhat_lower",
                "Upper Confidence Interval": "yhat_upper",
            }
        ),
    )
    st.plotly_chart(fig2)

    # Accuracy Metrics (full names + explanation)
    st.subheader("📏 Forecast Accuracy (on historical data)")
    try:
        y_true = df_train["y"].values
        y_pred = forecast.loc[: len(y_true) - 1, "Predicted Price"].values
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        st.success(
            f"✅ Root Mean Squared Error (RMSE): {rmse:.2f} | Mean Absolute Percentage Error (MAPE): {mape:.2f}%"
        )
        st.caption(
            """
            **RMSE**: Measures the average size of prediction errors in the same units as the stock price, with larger errors having more influence.  
            **MAPE**: Shows the average prediction error as a percentage of the actual values, making it easier to interpret across different price levels.
            """
        )
    except Exception as e:
        st.warning(f"⚠️ Unable to compute accuracy metrics: {e}")

    # Download
    st.subheader("📥 Download Forecast CSV")
    csv = forecast.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Forecast",
        data=csv,
        file_name=f"{selected_stock}_forecast.csv",
        mime="text/csv",
    )

# ----------------- ABOUT ------------------
if selected == "About":
    st.title("ℹ️ About This Project")
    st.markdown(
        """
        This app was built to forecast stock prices and support data-driven decisions.

        **Developed by:** Paul Prosper Lawer  
        **Email:** [prospaul999@gmail.com](mailto:prospaul999@gmail.com)  
        **WhatsApp:** [+233594760444](https://wa.me/233594760444)

        **Stack Used:**  
        - 🧠 Facebook Prophet  
        - 📉 Yahoo Finance via yfinance  
        - 📊 Technical Indicators via pandas-ta  
        - 📦 Streamlit + Plotly for UI  
        """
    )
