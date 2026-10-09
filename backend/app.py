import os
import asyncio
import pandas as pd
import joblib
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI App
app = FastAPI(title="Loan Application Deep Learning API")

# Define PyTorch Neural Network Architecture
class SimpleLoanMLP(nn.Module):
    def __init__(self, input_dim=6):
        super(SimpleLoanMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# Resolve paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Load Scaler
scaler = joblib.load(SCALER_PATH)

# Load PyTorch Model
model = SimpleLoanMLP(input_dim=6)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

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
    return {"message": "Loan Approval Deep Learning API is running successfully"}

# Prediction Route
@app.post("/predict")
async def predict_loan(data: LoanInput):
    print("Received prediction request:", data)

    # Convert incoming payload to pandas DataFrame
    input_df = pd.DataFrame([
        {
            'no_of_dependents': data.no_of_dependents,
            'income_annum': data.income_annum,
            'loan_amount': data.loan_amount,
            'loan_term': data.loan_term,
            'cibil_score': data.cibil_score,
            'total_assets_value': data.total_assets_value,
        }
    ])

    # Scale input data
    scaled_data = scaler.transform(input_df)

    # Convert to PyTorch Tensor
    input_tensor = torch.tensor(scaled_data, dtype=torch.float32)

    # Perform Deep Learning Inference
    with torch.no_grad():
        probability = model(input_tensor).item()

    prediction_result = "Approved" if probability >= 0.5 else "Rejected"

    return {
        "prediction": prediction_result,
        "approval_probability": round(float(probability), 4)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

