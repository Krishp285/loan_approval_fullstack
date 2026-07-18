import streamlit as st
import requests

st.title("Loan Approval Prediction System")


# User Inputs
no_of_dependents = st.number_input("no of dependents", min_value=0)

income_annum = st.number_input("income annum", min_value=0)

loan_amount = st.number_input("loan amount", min_value=0)

loan_term = st.number_input("loan term", min_value=0)

cibil_score = st.number_input("cibil score", min_value=0)

total_assets_value = st.number_input("Total Assets Value", min_value=0)


# Predict Button
if st.button("Predict Loan Status"):

    payload = {
        "no_of_dependents": no_of_dependents,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "total_assets_value": total_assets_value,
    }

    # API Call
    response = requests.post("http://127.0.0.1:8000/predict", json=payload)

    result = response.json()

    st.subheader(f"Prediction: {result['prediction']}")

    st.write(f"Approval Probability: {result['approval_probability']}")
