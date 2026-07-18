# Loan Approval Prediction System

A complete Machine Learning project using:

- **Scikit-learn** for ML Model
- **FastAPI** for Backend API
- **Streamlit** for Frontend UI

This project predicts whether a loan will be approved or rejected based on applicant details.

---

# Project Architecture

```text
Streamlit UI  --->  FastAPI Backend  --->  ML Model
```

---

# Features

- Loan approval prediction
- Interactive Streamlit form
- FastAPI async backend
- Machine Learning model integration
- Probability score for prediction
- Clean project structure

---

# Tech Stack

## Frontend

- Streamlit

## Backend

- FastAPI
- Uvicorn

## Machine Learning

- Scikit-learn
- Pandas
- Joblib

---

# Project Structure

```text
loan-prediction-project/
│
├── backend/
│   ├── app.py
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── requirements.txt
│
├── frontend/
│   ├── streamlit_app.py
│   ├── requirements.txt
│
├── dataset/
│   ├── loan-data.csv
│
├── train_model.py
│
└── README.md
```

---

# Step 1 — Clone Project

```bash
git clone <your-repository-url>
```

Move into project folder:

```bash
cd loan-prediction-project
```

---

# Step 2 — Create Virtual Environment

## Windows

```bash
python -m venv venv
```

Activate environment:

```bash
venv\Scripts\activate
```

---

## Linux / Mac

```bash
python3 -m venv venv
```

Activate environment:

```bash
source venv/bin/activate
```

---

# Step 3 — Install Backend Dependencies

Move to backend folder:

```bash
cd backend
```

Install packages:

```bash
pip install -r requirements.txt
```

---

# Backend Requirements

## `backend/requirements.txt`

```txt
fastapi
uvicorn
scikit-learn
pandas
joblib
python-multipart
```

---

# Step 4 — Install Frontend Dependencies

Open another terminal.

Move to frontend folder:

```bash
cd frontend
```

Install packages:

```bash
pip install -r requirements.txt
```

---

# Frontend Requirements

## `frontend/requirements.txt`

```txt
streamlit
requests
```

---

# Step 5 — Train Machine Learning Model

Move to project root folder:

```bash
cd ..
```

Run training script:

```bash
python train_model.py
```

After successful training:

- `model.pkl` will be generated
- `scaler.pkl` will be generated

These files are used by FastAPI for prediction.

---

# Step 6 — Start FastAPI Backend

Move to backend folder:

```bash
cd backend
```

Run server:

```bash
uvicorn app:app --reload
```

Backend will run on:

```text
http://127.0.0.1:8000
```

---

# FastAPI Swagger Documentation

Open browser:

```text
http://127.0.0.1:8000/docs
```

You can test APIs directly from Swagger UI.

---

# Step 7 — Start Streamlit Frontend

Open another terminal.

Move to frontend folder:

```bash
cd frontend
```

Run Streamlit app:

```bash
streamlit run streamlit_app.py
```

Frontend will run on:

```text
http://localhost:8501
```

---

# Application Workflow

```text
User fills Streamlit Form
            ↓
Frontend sends API request
            ↓
FastAPI receives request
            ↓
ML Model predicts result
            ↓
FastAPI sends response
            ↓
Streamlit displays prediction
```

---

# API Information

## Base URL

```text
http://127.0.0.1:8000
```

---

## Prediction Endpoint

```text
POST /predict
```

---

# Sample Request Body

```json
{
  "Dependents": 1,
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 2000,
  "LoanAmount": 120,
  "Loan_Amount_Term": 360,
  "Credit_History": 1
}
```

---

# Sample Response

```json
{
  "prediction": "Approved",
  "approval_probability": 0.87
}
```

---

# Important Concepts Used

## Machine Learning

- Data Cleaning
- Feature Scaling
- Logistic Regression
- Model Training
- Prediction

---

## FastAPI

- REST APIs
- Async Functions
- Request Validation
- JSON Response

---

## Streamlit

- Forms
- Input Fields
- API Integration
- Interactive UI

---

# Common Commands

## Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

# Stop Running Server

Press:

```bash
CTRL + C
```

---

# Common Errors

## Module Not Found Error

Install requirements again:

```bash
pip install -r requirements.txt
```

---

## Port Already In Use

Run server on another port:

```bash
uvicorn app:app --reload --port 8001
```

---

# Learning Outcomes

After completing this project, you will understand:

- Machine Learning workflow
- Model serialization
- API development
- Async backend architecture
- Frontend and backend integration
- End-to-end AI application development
