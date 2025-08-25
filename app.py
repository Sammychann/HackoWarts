# app.py

# Importing required libraries
import streamlit as st 
import numpy as np 
import pandas as pd 
import plotly.graph_objects as go 
import plotly.express as px 
from datetime import timedelta 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose 
import matplotlib.pyplot as plt
import pydeck as pdk  
import os, joblib

# =======================
# Page Configuration
# =======================
st.set_page_config(
    layout="wide",
    page_title="Water Quality Dashboard",
    page_icon="💧",
    initial_sidebar_state="expanded"
)

st.title("💧 Water Quality Dashboard")
st.caption("Real-time monitoring, analysis, and forecasting of water quality metrics.")

# =======================
# Sample Data Generator
# =======================
date_range = pd.date_range(start="2024-01-01", periods=365, freq='D')
np.random.seed(42)
filtered_data = pd.DataFrame({
    'Date': date_range,
    'Temperature (ºC)': np.random.normal(25, 3, len(date_range)),
    'D.O. (mg/l)': np.random.normal(7, 1, len(date_range)),
    'pH': np.random.normal(7, 0.5, len(date_range)),
    'Turbidity (NTU)': np.random.normal(3, 1, len(date_range)),
}).set_index('Date')

metrics = {
    'pH': filtered_data['pH'],
    'Turbidity (NTU)': filtered_data['Turbidity (NTU)'],
    'D.O. (mg/l)': filtered_data['D.O. (mg/l)'],
    'Temperature (ºC)': filtered_data['Temperature (ºC)'],
}

# =======================
# Sidebar: Location
# =======================
st.sidebar.header("📍 Select Location")
selected_location = st.sidebar.selectbox(
    'Choose Location', ['Delhi', 'Varanasi', 'Kolkata']
)

locations = {
    'Delhi': [28.7041, 77.1025],
    'Varanasi': [25.3176, 82.9739],
    'Kolkata': [22.5726, 88.3639]
}

# =======================
# Tabs for Dashboard
# =======================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Metrics", "🗺 Map", "📈 Trends", "🔮 Forecasts", "⚙️ Tools"
])

# =======================
# Tab 1: Metrics Gauges
# =======================
with tab1:
    st.subheader("📊 Water Quality Metrics")
    for metric, values in metrics.items():
        value = values.iloc[-1]
        max_value = 14 if 'pH' in metric else values.max()
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            gauge={
                'axis': {'range': [0, max_value]},
                'steps': [
                    {'range': [0, max_value*0.5], 'color': '#99d98c'},
                    {'range': [max_value*0.75, max_value], 'color': '#d62828'}
                ],
            },
            title={'text': f"{metric}", 'font': {'size': 14}}
        ))
        st.plotly_chart(fig, use_container_width=True)

# =======================
# Tab 2: Map
# =======================
with tab2:
    st.subheader("🗺 Location Map")
    view_state = pdk.ViewState(
        latitude=locations[selected_location][0],
        longitude=locations[selected_location][1],
        zoom=10, pitch=30
    )
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame([{
            "latitude": locations[selected_location][0],
            "longitude": locations[selected_location][1]
        }]),
        get_position='[longitude, latitude]',
        get_color='[200, 30, 0, 160]',
        get_radius=50000,
    )
    r = pdk.Deck(layers=[layer], initial_view_state=view_state)
    st.pydeck_chart(r)

# =======================
# Tab 3: Trends & ETS
# =======================
with tab3:
    st.subheader("📈 Trend Over Time")
    selected_metric = st.selectbox('Choose Metric', list(metrics.keys()))
    line_fig = px.line(filtered_data, x=filtered_data.index, y=selected_metric, 
                       title=f"{selected_metric} Over Time", height=350)
    st.plotly_chart(line_fig)

    st.subheader("🔍 ETS Decomposition")
    decomposition = seasonal_decompose(metrics[selected_metric], model='additive', period=30)
    fig, ax = plt.subplots(4, 1, figsize=(8, 10))
    decomposition.observed.plot(ax=ax[0], title='Observed')
    decomposition.trend.plot(ax=ax[1], title='Trend')
    decomposition.seasonal.plot(ax=ax[2], title='Seasonal')
    decomposition.resid.plot(ax=ax[3], title='Residual')
    plt.tight_layout()
    st.pyplot(fig)

# =======================
# Tab 4: Forecasts
# =======================
with tab4:
    st.subheader("🔮 7-Day Predictions (ARIMA)")

    def arima_predictions(data, metric, days=7):
        model = ARIMA(data[metric], order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=days)
        return forecast

    predictions = {metric: arima_predictions(filtered_data, metric) for metric in metrics.keys()}
    prediction_dates = [filtered_data.index[-1] + timedelta(days=i) for i in range(1, 8)]
    predictions_df = pd.DataFrame(predictions, index=prediction_dates)
    st.table(predictions_df)

    # Prophet Forecast (only pH)
    st.subheader("🔮 Prophet Forecast (pH)")
    prophet_model_path = os.path.join("models", "prophet_pH_model.pkl")
    if os.path.exists(prophet_model_path):
        model = joblib.load(prophet_model_path)
        future = model.make_future_dataframe(periods=7, freq="D")
        forecast = model.predict(future)

        # Use Plotly instead of st.line_chart
        fig = px.line(
            forecast,
            x="ds",
            y=["yhat", "yhat_lower", "yhat_upper"],
            labels={"ds": "Date", "value": "pH"},
            title="Prophet Forecast (pH)"
        )
        fig.update_layout(
            yaxis=dict(range=[5, 9], title="pH Level"),
            xaxis=dict(title="Date"),
            legend_title="Prediction"
        )
        fig.add_hline(y=7, line_dash="dash", line_color="green", annotation_text="Neutral pH (7)")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(7))
    else:
        st.warning("⚠️ Prophet model not found. Place `prophet_pH_model.pkl` inside `/models` folder.")

# =======================
# Tab 5: Tools (Chatbot & Alerts)
# =======================
with tab5:
    st.subheader("🤖 Chatbot")
    st.text_input("Ask me anything about water quality metrics!")

    st.subheader("🚨 Twilio Alerts")
    st.markdown(
        "<div style='background-color: #f8f9fa; border: 1px solid #e0e0e0; "
        "box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1); padding: 12px; border-radius: 8px;'>"
        "<strong>Stay Informed!</strong> Receive real-time water quality alerts on your phone.",
        unsafe_allow_html=True
    )


