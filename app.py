import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import logging
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(layout="wide")
st.markdown("---")
st.title("KPIs")

st.sidebar.header("Business Use Cases")
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2025-01-01"))
end_date = st.sidebar.date_input("End date", pd.to_datetime("today"))

@st.cache_data
def load_data(ticker, start, end):
    try:
        logger.info(f"Loading data for {ticker} from {start} to {end}")
        tkr = yf.Ticker(ticker)
        df = tkr.history(start=start, end=end)
        df.columns = df.columns.str.strip()
        logger.info(f"Loaded {len(df)} rows of data")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error("Failed to load data.")
        return pd.DataFrame()

data = load_data("BAC", start_date, end_date)

if data.empty:
    st.warning("No data loaded. Adjust dates or check ticker symbol.")
    st.stop()

returns = data["Close"].pct_change().dropna()
vol_window = st.sidebar.selectbox("Volatility window (days)", [7, 14, 30, 60], index=2)
rolling_vol = returns.rolling(window=vol_window).std() * np.sqrt(252)
total_return = (data["Close"].iloc[-1] / data["Close"].iloc[0] - 1) * 100

last_close = data["Close"].iloc[-1]
prev_close = data["Close"].iloc[-2]
daily_change = last_close - prev_close
daily_pct_change = (daily_change / prev_close) * 100

vol_ref = st.sidebar.slider("Reference volatility (%)", 10, 50, 25) / 100
vol_current = rolling_vol.iloc[-1]
vol_delta_pct = (vol_current - vol_ref) * 100

return_ref = 0
return_delta = total_return - return_ref

logger.info(f"Calculated KPIs: last_close=${last_close:.2f}, daily_pct_change={daily_pct_change:.2f}%, total_return={total_return:.2f}%")

def get_delta_color(current, previous):
    if current > previous:
        return "normal"
    elif current < previous:
        return "inverse"
    else:
        return "off"

cols1 = st.columns(3)

cols1[0].metric(
    label="Last close",
    value=f"${last_close:.2f}",
    delta=f"{daily_pct_change:+.2f}%",
    delta_color="normal"
)

cols1[1].metric(
    label="Volatility (business use case window)",
    value=f"{vol_current*100:.2f}%",
    delta=f"{vol_delta_pct:+.2f}%",
    delta_color="inverse"
)

cols1[2].metric(
    label="Total return",
    value=f"{total_return:+.2f}%",
    delta=f"{return_delta:+.2f}%",
    delta_color="normal"
)

dividend_rows = data[data['Dividends'] > 0]

if len(dividend_rows) >= 2:
    last_div_date = dividend_rows.index[-1]
    last_div_amount = dividend_rows['Dividends'].iloc[-1]

    prev_div_date = dividend_rows.index[-2]
    prev_div_amount = dividend_rows['Dividends'].iloc[-2]

    today = pd.Timestamp.now(tz=last_div_date.tz).normalize()
    days_since_div = (today - last_div_date).days
    prev_days_since_div = (last_div_date - prev_div_date).days

    days_since_div_color = get_delta_color(days_since_div, prev_days_since_div)
    div_amount_color = get_delta_color(last_div_amount, prev_div_amount)

    data_since_div = data[data.index >= last_div_date]
    data_since_prev_div = data[data.index >= prev_div_date]

    price_col = "Adj Close" if "Adj Close" in data.columns else "Close"

    if len(data_since_div) > 2 and len(data_since_prev_div) > 2:
        data_since_div = data_since_div.copy()
        data_since_div["Return"] = data_since_div[price_col].pct_change()
        avg_return_since_div = data_since_div["Return"].mean() * 100

        data_since_prev_div = data_since_prev_div.copy()
        data_since_prev_div["Return"] = data_since_prev_div[price_col].pct_change()
        avg_return_since_prev_div = data_since_prev_div["Return"].mean() * 100

        avg_return_color = get_delta_color(avg_return_since_div, avg_return_since_prev_div)
    else:
        avg_return_since_div = None
        avg_return_color = "off"
else:
    last_div_date = None
    last_div_amount = None
    days_since_div = None
    days_since_div_color = "off"
    div_amount_color = "off"
    avg_return_since_div = None
    avg_return_color = "off"

cols2 = st.columns(3)

if last_div_date:
    cols2[0].metric(
        label="Days since last dividend",
        value=f"{days_since_div} days",
        delta=f"{days_since_div - prev_days_since_div:+d} days",
        delta_color=days_since_div_color
    )
else:
    cols2[0].write("No dividend data")

if last_div_date:
    cols2[1].metric(
        label="Last dividend amount",
        value=f"${last_div_amount:.2f}",
        delta=f"${last_div_amount - prev_div_amount:+.2f}",
        delta_color=div_amount_color
    )
else:
    cols2[1].write("No dividend data")

if avg_return_since_div is not None:
    cols2[2].metric(
        label="Avg daily return since dividend",
        value=f"{avg_return_since_div:.4f}%",
        delta=f"{(avg_return_since_div - avg_return_since_prev_div):+.4f}%",
        delta_color=avg_return_color
    )
else:
    cols2[2].write("Avg return since dividend: Not enough data")

st.write("Glance of the data")
st.write(data.head())

st.markdown("---")
st.subheader("Stock price and volatility over time")
st.markdown(
    "Root cause analysis - "
    "On why volatility is very high and stock price is very low. Reciprocal tariffs threat on around Apr 4 on all countries. "
    "Source: https://www.tradecomplianceresourcehub.com/2025/07/11/trump-2-0-tariff-tracker/"
)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data.index,
    y=data['Close'],
    name='Stock price',
    line=dict(color='blue'),
    yaxis='y1'
))

fig.add_trace(go.Scatter(
    x=rolling_vol.index,
    y=rolling_vol,
    name='Volatility',
    line=dict(color='red'),
    yaxis='y2'
))

fig.update_layout(
    title="Stock price and volatility over time",
    xaxis=dict(title="Date", showgrid=False),
    yaxis=dict(
        title=dict(text="Price", font=dict(color="blue")),
        tickfont=dict(color="blue"),
        side="left",
        showgrid=False
    ),
    yaxis2=dict(
        title=dict(text="Volatility", font=dict(color="red")),
        tickfont=dict(color="red"),
        overlaying="y",
        side="right",
        showgrid=False
    ),
    legend=dict(x=0, y=1.1, orientation="h"),
    height=500,
    margin=dict(l=50, r=50, t=80, b=50),
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Cumulative return")
st.markdown(
    "Root cause analysis - same goes for the dip in returns is because tariffs. "
    "Source: https://www.tradecomplianceresourcehub.com/2025/07/11/trump-2-0-tariff-tracker/"
)

cum_return = (1 + returns).cumprod() - 1
cum_return_df = pd.DataFrame({"Cumulative return": cum_return}, index=returns.index)
fig_cum = px.line(cum_return_df, x=cum_return_df.index, y="Cumulative return", title="Cumulative return over time")
st.plotly_chart(fig_cum, use_container_width=True)

st.markdown("---")
st.subheader("Time series modeling")

price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
ts_series = data[price_col].dropna()
ts_diff = ts_series.diff().dropna()

tab1, tab2, tab3, tab4 = st.tabs([
    "ACF & PACF",
    "Stationarity",
    "Forecasting",
    "Residual analysis"
])

with tab1:
    logger.info("Plotting ACF and PACF")
    st.write("Autocorrelation and partial autocorrelation of original and differenced series.")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    plot_acf(ts_series, ax=axs[0, 0])
    axs[0, 0].set_title("ACF - Original")

    plot_pacf(ts_series, ax=axs[0, 1])
    axs[0, 1].set_title("PACF - Original")

    plot_acf(ts_diff, ax=axs[1, 0])
    axs[1, 0].set_title("ACF - Differenced")

    plot_pacf(ts_diff, ax=axs[1, 1])
    axs[1, 1].set_title("PACF - Differenced")

    st.pyplot(fig)

with tab2:
    logger.info("Performing Augmented Dickey-Fuller test")
    st.write("Augmented Dickey-Fuller test:")
    adf_result = adfuller(ts_series)
    st.write(f"ADF Statistic: {adf_result[0]:.4f}")
    st.write(f"p-value: {adf_result[1]:.4f}")
    if adf_result[1] < 0.05:
        st.success("Reject H0: Series is stationary.")
    else:
        st.warning("Fail to reject H0: Series is non-stationary.")
    st.line_chart(ts_diff)

with tab3:
    logger.info("Starting ARIMA forecasting")
    st.subheader("Forecasting using ARIMA and GARCH models")

    arima_order = st.selectbox("Select ARIMA order (p,d,q):", [(1, 1, 1), (2, 1, 2), (3, 1, 3)], index=0)

    ts_series.index = pd.to_datetime(ts_series.index)
    full_index = pd.date_range(start=ts_series.index.min(), end=ts_series.index.max(), freq='B')
    ts_series = ts_series.reindex(full_index).ffill()

    arima_model = ARIMA(ts_series, order=arima_order)
    arima_res = arima_model.fit()
    logger.info(f"ARIMA model fitted with order {arima_order}")

    n_forecast = st.slider("Forecast horizon (days):", min_value=5, max_value=30, value=10)

    arima_forecast_res = arima_res.get_forecast(steps=n_forecast)
    forecast_df = arima_forecast_res.summary_frame()
    forecast_mean = forecast_df["mean"]
    ci_lower = forecast_df["mean_ci_lower"]
    ci_upper = forecast_df["mean_ci_upper"]

    st.markdown(f"""
    **ARIMA model summary**
    - Observations: {len(returns)}
    - AIC: {arima_res.aic:.2f}
    - BIC: {arima_res.bic:.2f}
    """)

    fig_arima = go.Figure()

    fig_arima.add_trace(go.Scatter(
        x=ts_series.index,
        y=ts_series,
        mode="lines",
        name="Observed",
        line=dict(color="blue")
    ))

    fig_arima.add_trace(go.Scatter(
        x=forecast_mean.index,
        y=forecast_mean,
        mode="lines",
        name="Forecast",
        line=dict(color="red", dash="dash")
    ))

    fig_arima.add_trace(go.Scatter(
        x=forecast_mean.index.tolist() + forecast_mean.index[::-1].tolist(),
        y=ci_upper.tolist() + ci_lower[::-1].tolist(),
        fill="toself",
        fillcolor="rgba(255,0,0,0.2)",
        line=dict(color="rgba(255,0,0,0)"),
        hoverinfo="skip",
        name="95% CI"
    ))

    fig_arima.update_layout(
        title="ARIMA forecast",
        xaxis_title="Date",
        yaxis_title="Value",
        height=500
    )

    st.plotly_chart(fig_arima, use_container_width=True)
    logger.info("ARIMA forecast plotted")

with tab4:
    logger.info("Starting residual analysis")
    st.write("Residual diagnostics of ARIMA model.")
    residuals = arima_res.resid.dropna()

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    plot_acf(residuals, ax=axs[0, 0], lags=40)
    axs[0, 0].set_title("Residual ACF")

    sm.qqplot(residuals, line='s', ax=axs[0, 1])
    axs[0, 1].set_title("Q-Q Plot")

    axs[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axs[1, 0].set_title("Histogram of residuals")
    axs[1, 0].set_xlabel("Residual")
    axs[1, 0].set_ylabel("Frequency")

    lb_test = acorr_ljungbox(residuals, lags=20, return_df=True)
    lb_pvalue = lb_test['lb_pvalue']

    axs[1, 1].scatter(lb_pvalue.index, lb_pvalue, color='steelblue', s=50)
    axs[1, 1].axhline(y=0.05, color='red', linestyle='--', label='Significance level (0.05)')
    axs[1, 1].set_title("Ljung-Box test p-values (first 20 lags)")
    axs[1, 1].set_xlabel("Lag")
    axs[1, 1].set_ylabel("P-value")
    axs[1, 1].set_ylim(0, 1.5)
    axs[1, 1].legend()

    st.pyplot(fig)
    logger.info("Residual analysis complete")

st.sidebar.header("Key Takeaways")
st.sidebar.markdown(" 1- Basic Standard Practice of Logging")
st.sidebar.markdown(" 2 - Followed Statistician Analysis Practices")
st.sidebar.markdown(" 3 - Followed Time Series Analysis Workflow")
