import streamlit as st 
import numpy as np 
import pandas as pd 
import plotly.graph_objects as go 
import plotly.express as px 
from datetime import datetime, timedelta 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose 
import matplotlib.pyplot as plt
from twilio.rest import Client  
import pydeck as pdk  
import requests
import json

# Page Configuration
st.set_page_config(
    layout="wide",
    page_title="Water Quality Dashboard",
    page_icon="💧",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
        [data-testid="stToolbar"] {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# Left Column - Select Location and Metrics Dashboard
left_col, right_col = st.columns([1, 2])
with left_col:
    st.markdown("### 📍 Select Location") 
    selected_location = st.selectbox('Choose Location', ['Delhi', 'Varanasi', 'Kolkata'])
    
    # Generate sample water quality data
    date_range = pd.date_range(start="2024-01-01", periods=365, freq='D')
    np.random.seed(42)
    temp = np.random.normal(25, 3, len(date_range))  
    do = np.random.normal(7, 1, len(date_range))  
    ph = np.random.normal(7, 0.5, len(date_range))  
    turbidity = np.random.normal(3, 1, len(date_range))  
    filtered_data = pd.DataFrame({
        'Date': date_range,
        'Temperature (˚C)': temp,
        'D.O. (mg/l)': do,
        'pH': ph,
        'Turbidity (NTU)': turbidity,
    }).set_index('Date')

    # Extract Metrics
    metrics = {
        'pH': filtered_data['pH'], 
        'Turbidity (NTU)': filtered_data['Turbidity (NTU)'], 
        'D.O. (mg/l)': filtered_data['D.O. (mg/l)'], 
        'Temperature (˚C)': filtered_data['Temperature (˚C)'], 
    }

    st.markdown("### 📊 Water Quality Metrics")
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
                    {'range': [max_value*0.5, max_value*0.75], 'color': '#fcbf49'},
                    {'range': [max_value*0.75, max_value], 'color': '#d62828' }
                ],
            },
            title={'text': f"{metric}", 'font': {'size': 12}},  
            domain={'x': [0, 1], 'y': [0, 0.4]}  
        )) 
        st.plotly_chart(fig, use_container_width=True)

# Right Column - Interactive Map Visualization
with right_col:
    st.markdown("### 🗺 Location Map")

    # Coordinates for selected locations
    locations = {
        'Delhi': [28.7041, 77.1025],
        'Varanasi': [25.3176, 82.9739],
        'Kolkata': [22.5726, 88.3639]
    }

    # Set the map view based on the selected location
    view_state = pdk.ViewState(
        latitude=locations[selected_location][0],
        longitude=locations[selected_location][1],
        zoom=10,
        pitch=30
    )

    # Layer for the selected location
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

    # Render the map in Streamlit
    r = pdk.Deck(layers=[layer], initial_view_state=view_state)
    st.pydeck_chart(r)

# Create a new row for the line chart, ETS decomposition, and alerts
with right_col:
    st.markdown("## 📈 Trend Over Time")
    selected_metric = st.selectbox('Choose Metric for Line Chart', list(metrics.keys()), key="line_chart")
    line_fig = px.line(filtered_data, x=filtered_data.index, y=selected_metric, title=f"{selected_metric} Over Time", height=300)
    st.plotly_chart(line_fig)

    st.markdown("## 🔍 ETS Decomposition")
    decomposition = seasonal_decompose(metrics[selected_metric], model='additive', period=30)
    fig, ax = plt.subplots(4, 1, figsize=(8, 10))
    decomposition.observed.plot(ax=ax[0], title='Observed')
    decomposition.trend.plot(ax=ax[1], title='Trend')
    decomposition.seasonal.plot(ax=ax[2], title='Seasonal')
    decomposition.resid.plot(ax=ax[3], title='Residual')
    plt.tight_layout()
    st.pyplot(fig)

# ARIMA Predictions for the Next 7 Days
def arima_predictions(data, metric, days=7):
    model = ARIMA(data[metric], order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=days)
    return forecast

predictions = {metric: arima_predictions(filtered_data, metric) for metric in metrics.keys()}
prediction_dates = [filtered_data.index[-1] + timedelta(days=i) for i in range(1, 8)]
predictions_df = pd.DataFrame(predictions, index=prediction_dates)

st.markdown("## 🗓 7-Day Predictions")
st.table(predictions_df)

# Report Button
if st.button("Generate Report"):
    report = f"""
    ## Water Quality Report for {selected_location}
    **Latest Metrics:**
    - Temperature: {filtered_data['Temperature (˚C)'].iloc[-1]:.2f}˚C
    - D.O.: {filtered_data['D.O. (mg/l)'].iloc[-1]:.2f} mg/l
    - pH: {filtered_data['pH'].iloc[-1]:.2f}
    - Turbidity: {filtered_data['Turbidity (NTU)'].iloc[-1]:.2f} NTU
    """
    st.download_button(label="Download Report", data=report, file_name="water_quality_report.txt", mime="text/plain")
    st.success("Report generated successfully!")
      
    # Chatbot Integration
st.markdown("### 🤖 AI Chatbot for Water Pollution Awareness")
user_query = st.text_input("Ask me anything about water pollution:")
if st.button("Ask Chatbot"):
    if user_query:
        api_url = "https://api.worqhat.com/api/ai/content/v4"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer wh_m764aw5hSXgSxTc9kEmCDk7RptQQJjBnjzjt0cGaD"
        }
        payload = {
            "question": user_query,
            "model": "aicon-v4-nano-160824",
            "randomness": 0.5,
            "stream_data": False,
            "training_data": "You are an environmental scientist helping people understand water pollution and its effects.",
            "response_type": "text"
        }

        try:
            response = requests.post(api_url, headers=headers, json=payload)
            if response.status_code == 200:
                chatbot_response = response.json().get("content", "Sorry, I couldn't fetch a response.")
                st.markdown(f"**Chatbot:** {chatbot_response}")
            else:
                st.error("Error fetching chatbot response. Try again later.")
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")

# Twilio Alerts UI Element
st.markdown("### 🚨 Twilio Alerts")
st.markdown(
    "<div style='background-color: #ffffff; border: 1px solid #e0e0e0; "
    "box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1); padding: 10px; border-radius: 8px;'>"
    "<strong style='color: black;'>Stay Informed!</strong> "
    "<span style='color: black;'>Receive real-time water quality alerts on your phone.</span>"
    "</div>",
    unsafe_allow_html=True
)
