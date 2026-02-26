import streamlit as st
import requests

API_URL = "http://localhost:5000"

st.title("📈 Stock Price Predictor")

# Fetch company names
@st.cache_data
def get_companies():
    response = requests.get(f"{API_URL}/companies")
    return response.json()['companies']

companies = get_companies()
company = st.selectbox("Select Company", companies)
date = st.date_input("Select Prediction Date")

if st.button("Predict"):
    payload = {"company": company, "date": str(date)}
    response = requests.post(f"{API_URL}/predict", json=payload)

    if response.status_code == 200:
        result = response.json()
        st.success(f"📊 Predicted Close Price for {company} on {date}: ${result['prediction']}")
    else:
        st.error(f"❌ Error: {response.json()['error']}")
