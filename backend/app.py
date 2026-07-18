from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import asyncio

# Create FastAPI App
app = FastAPI()


# Load Model
model = joblib.load("D:/SoftCoding_practice/Loan_Application_Model/backend/model.pkl")

# Load Scaler
scaler = joblib.load("D:/SoftCoding_practice/Loan_Application_Model/backend/scaler.pkl")


# Request Body Schema
class LoanInput(BaseModel):
       
    no_of_dependents: int
    income_annum: int
    loan_amount: int
    loan_term: int
    cibil_score: int
    total_assets_value: int

# Home Route
@app.get("/")
async def home():
    return {"message": "Loan Prediction API Running"}


# Prediction Route
@app.post("/predict")
async def predict_loan(data: LoanInput):
    print("Received data: " , data)
    # Simulate async work
    await asyncio.sleep(0.1)

    # Convert to DataFrame
    input_data = pd.DataFrame(
        [
            {
                ' no_of_dependents': data.no_of_dependents,
                ' income_annum': data.income_annum,
                ' loan_amount': data.loan_amount,
                ' loan_term': data.loan_term,
                ' cibil_score': data.cibil_score,
                'total_assets_value': data.total_assets_value,    
            }
        ]
    )

    # Scale Data
    scaled_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_data)[0]

    # Probability
    probability = model.predict_proba(scaled_data)[0][1]

    # Final Result
    result = "Approved" if prediction == 1 else "Rejected"

    return {"prediction": result, "approval_probability": float(probability)}
