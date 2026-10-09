import os
import requests
import streamlit as st

# Configure Page Title & Layout
st.set_page_config(
    page_title="Loan Approval Deep Learning System",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Approval Prediction System")
st.markdown("Enter applicant details below to analyze loan approval probability using our **PyTorch Deep Learning Model**.")

# Retrieve Backend URL from environment variable or default to local URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

# Form layout using columns
col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, value=2, step=1)
    income_annum = st.number_input("Annual Income ($ / ₹)", min_value=0, value=5000000, step=50000)
    loan_amount = st.number_input("Loan Amount Requested ($ / ₹)", min_value=0, value=2000000, step=25000)

with col2:
    loan_term = st.number_input("Loan Term (Months)", min_value=1, value=36, step=6)
    cibil_score = st.number_input("CIBIL / Credit Score (300 - 900)", min_value=300, max_value=900, value=750, step=5)
    total_assets_value = st.number_input("Total Assets Value ($ / ₹)", min_value=0, value=10000000, step=100000)

st.markdown("---")

# Predict Button
if st.button("🚀 Analyze & Predict Loan Status", use_container_width=True):
    payload = {
        "no_of_dependents": int(no_of_dependents),
        "income_annum": int(income_annum),
        "loan_amount": int(loan_amount),
        "loan_term": int(loan_term),
        "cibil_score": int(cibil_score),
        "total_assets_value": int(total_assets_value),
    }

    with st.spinner("Processing applicant profile through PyTorch Deep Learning Model..."):
        try:
            api_endpoint = f"{BACKEND_URL}/predict"
            response = requests.post(api_endpoint, json=payload, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction", "Unknown")
                probability = result.get("approval_probability", 0.0)

                st.subheader("Analysis Results:")
                if prediction == "Approved":
                    st.success(f"🎉 **Status:** {prediction}")
                else:
                    st.error(f"❌ **Status:** {prediction}")

                col_acc1, col_acc2 = st.columns(2)
                col_acc1.metric("Decision", prediction)
                col_acc2.metric("Approval Probability", f"{probability * 100:.2f}%")
            else:
                st.error(f"Error from API server (Status {response.status_code}): {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to backend server at `{BACKEND_URL}`. Details: {e}")
            st.info("If deploying on Render, please ensure the backend web service is active and `BACKEND_URL` environment variable is set.")

